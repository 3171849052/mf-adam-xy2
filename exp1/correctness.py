"""Compare Ghost Clipping with independent one-record autograd."""
import torch

from .ghost import GhostClipper, logical_gradients


def relative_error(actual, expected):
    numerator = sum((a.double() - b.double()).square().sum() for a, b in zip(actual, expected))
    denominator = sum(b.double().square().sum() for b in expected)
    return float((numerator / denominator.clamp_min(1e-30)).sqrt())


def reference(model, examples, batch_fn, loss_fn, clip):
    params = [p for p in model.parameters() if p.requires_grad]
    sums = [torch.zeros_like(p) for p in params]
    norms, coeff = [], []
    for row in examples:
        model.zero_grad(set_to_none=True)
        loss_fn(model, batch_fn([row])).sum().backward()
        assert all(p.grad is not None for p in params)
        n = sum(p.grad.double().square().sum() for p in params).sqrt()
        c = (clip / n.clamp_min(1e-30)).clamp(max=1.0)
        norms.append(float(n))
        coeff.append(float(c))
        for acc, p in zip(sums, params):
            acc.add_(p.grad * c.to(p.dtype))
    model.zero_grad(set_to_none=True)
    return sums, torch.tensor(norms, dtype=torch.float64), torch.tensor(coeff, dtype=torch.float64)


def validate(model, examples, batch_fn, loss_fn, clip=1.0, rtol=0.03):
    assert len(examples) == 8, "Eight distinct fixtures are needed to verify physical batches 1/2/4/8"
    params = [p for p in model.parameters() if p.requires_grad]
    saved = [p.detach().clone() for p in params]
    was_training = model.training
    model.train()
    report = {"passed": False, "rtol": rtol, "examples": [x["id"] for x in examples], "cases": []}
    clipper = GhostClipper(model, clip)
    # Both the standard B=0 initial state and nonzero A/B gradients are required.
    for nonzero_b in (False, True):
        if nonzero_b:
            generator = torch.Generator(device=params[0].device).manual_seed(314159)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad and ".lora_B." in name:
                        p.copy_(torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator) * 0.02)
        expected, ref_norms, ref_coef = reference(model, examples, batch_fn, loss_fn, clip)
        # Test all microbatch sizes with active clipping, including whole-parameter sums.
        previous = None
        for physical in (1, 2, 4, 8):
            actual, values = logical_gradients(model, examples, physical, batch_fn, loss_fn, clipper)
            ns = torch.tensor(values["norms"], dtype=torch.float64)
            cs = torch.tensor(values["coefficients"], dtype=torch.float64)
            torch.testing.assert_close(ns, ref_norms, rtol=rtol, atol=1e-6)
            torch.testing.assert_close(cs, ref_coef, rtol=rtol, atol=1e-6)
            error = relative_error(actual, expected)
            assert error <= rtol, f"Clipped gradient error {error} > {rtol} at physical={physical}"
            # Explicitly check A and B, so a dominant B gradient cannot hide an A error.
            groups = {}
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            for group in ("lora_A", "lora_B"):
                indices = [i for i, (n, _) in enumerate(named) if group in n]
                groups[group] = relative_error([actual[i] for i in indices], [expected[i] for i in indices])
                assert groups[group] <= rtol, f"{group}: {groups[group]}"
            physical_error = relative_error(actual, previous) if previous is not None else 0.0
            assert physical_error <= rtol
            previous = [x.clone() for x in actual]
            report["cases"].append({"nonzero_B": nonzero_b, "physical_batch": physical,
                                    "gradient_relative_error": error, "group_errors": groups,
                                    "norm_max_abs_error": float((ns - ref_norms).abs().max()),
                                    "coefficient_max_abs_error": float((cs - ref_coef).abs().max()),
                                    "physical_batch_relative_error": physical_error,
                                    "reference_norms": ref_norms.tolist(), "reference_coefficients": ref_coef.tolist()})
    clipper.close()
    with torch.no_grad():
        for p, value in zip(params, saved):
            p.copy_(value)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    report["passed"] = True
    return report
