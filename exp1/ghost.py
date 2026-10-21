"""Two-pass Ghost Clipping using Opacus' exact sequence Linear norm sampler.

Only untied trainable LoRA A/B Linear weights are accepted. Frozen Qwen
attention, convolution and gated delta modules propagate ordinary autograd.
Tensor gradient hooks cooperate with non-reentrant activation checkpointing.
"""
import torch
from opacus.grad_sample.linear import compute_linear_norm_sample


class GhostClipper:
    def __init__(self, model, clip=1.0):
        self.model, self.clip = model, clip
        self.params = [p for p in model.parameters() if p.requires_grad]
        assert self.params and clip > 0
        self.layers = []
        covered = []
        for name, layer in model.named_modules():
            direct = [p for p in layer.parameters(recurse=False) if p.requires_grad]
            if direct:
                assert type(layer) is torch.nn.Linear and ("lora_A." in name or "lora_B." in name), f"Unsupported trainable module: {name}"
                assert layer.bias is None, f"LoRA bias unsupported: {name}"
                self.layers.append((name, layer))
                covered.extend(direct)
        assert len(covered) == len(self.params) and {id(p) for p in covered} == {id(p) for p in self.params}, "Tied or uncovered parameters"
        assert all(p.dtype in (torch.float32, torch.float64) for p in self.params)
        self.active = False
        self.handles = [layer.register_forward_hook(self._forward_hook) for _, layer in self.layers]

    def _forward_hook(self, layer, args, output):
        if not self.active or not torch.is_grad_enabled() or not output.requires_grad:
            return
        activation = args[0].detach()

        def backprop(grad):
            # This hook only counts actual backwards, not checkpoint recomputations.
            assert id(layer.weight) not in self.seen, "Reused/tied LoRA layer: cross-use terms require a different norm sampler"
            self.seen.add(id(layer.weight))
            with torch.no_grad(), torch.autocast(device_type=grad.device.type, enabled=False):
                dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
                ns = compute_linear_norm_sample(layer, [activation.to(dtype)], grad.detach().to(dtype))[layer.weight]
                assert ns.shape == (self.batch_size,) and torch.isfinite(ns).all(), "Invalid per-sample norm"
                self.norm_sq.add_(ns.double().square())
        output.register_hook(backprop)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def backward(self, batch, loss_fn):
        """Return detached losses/norms/coefficients and leave clipped SUM grads."""
        self.model.zero_grad(set_to_none=True)
        self.batch_size = len(batch["input_ids"])
        device = self.params[0].device
        self.norm_sq = torch.zeros(self.batch_size, device=device, dtype=torch.float64)
        self.seen = set()
        self.active = True
        sample_losses = loss_fn(self.model, batch)
        assert sample_losses.shape == (self.batch_size,)
        sample_losses.sum().backward()
        self.active = False
        assert self.seen == {id(p) for p in self.params}, "Some trainable parameters did not receive Ghost norms"
        norms = self.norm_sq.sqrt()
        assert torch.isfinite(norms).all()
        coeff = (self.clip / norms.clamp_min(1e-30)).clamp(max=1.0).detach()
        self.model.zero_grad(set_to_none=True)
        weighted = loss_fn(self.model, batch)
        assert torch.isfinite(weighted).all()
        (weighted * coeff.to(weighted.dtype)).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.params)
        return sample_losses.detach(), norms.detach(), coeff


def logical_gradients(model, examples, physical, batch_fn, loss_fn, clipper=None):
    """Accumulate in FP32; no optimizer/noise operation at microbatch boundaries."""
    params = [p for p in model.parameters() if p.requires_grad]
    sums = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    loss_sum = 0.0
    norms, coeffs = [], []
    for start in range(0, len(examples), physical):
        model.zero_grad(set_to_none=True)
        batch = batch_fn(examples[start:start + physical])
        if clipper is not None:
            ls, ns, cs = clipper.backward(batch, loss_fn)
            norms.extend(ns.cpu().tolist())
            coeffs.extend(cs.cpu().tolist())
        else:
            ls = loss_fn(model, batch)
            ls.sum().backward()
        loss_sum += float(ls.detach().sum())
        for acc, p in zip(sums, params):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            acc.add_(p.grad.float())
    model.zero_grad(set_to_none=True)
    return sums, {"loss": loss_sum / len(examples), "norms": norms, "coefficients": coeffs}


def assign_private_grads(params, sums, batch_size, noise_std, generator):
    """One independent Gaussian tensor per parameter, once per logical batch."""
    with torch.no_grad():
        for p, acc in zip(params, sums):
            if noise_std:
                acc.add_(torch.randn(acc.shape, dtype=acc.dtype, device=acc.device, generator=generator), alpha=noise_std)
            p.grad = (acc / batch_size).to(p.dtype)
