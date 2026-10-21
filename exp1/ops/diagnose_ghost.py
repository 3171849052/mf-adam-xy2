"""Real-weight diagnostics only: distinguish clipping and BF16 batch effects."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp1.common import EXP, cache_paths, write_json
cache_paths()
from exp1.correctness import reference, relative_error
from exp1.data import collate, encode, load_split
from exp1.ghost import GhostClipper, logical_gradients
from exp1.modeling import add_lora, load_model, losses, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", choices=("dynamic", "logical64"), default="dynamic")
    parser.add_argument("--math-attention", action="store_true")
    parser.add_argument("--full-reduction", action="store_true")
    parser.add_argument("--native-convolution", action="store_true")
    parser.add_argument("--cublas", action="store_true")
    parser.add_argument("--batch-preserving-linears", action="store_true")
    parser.add_argument("--fp32-delta-norm", action="store_true")
    parser.add_argument("--fp32-control", action="store_true", help="Diagnostic precision control only; never a production training configuration")
    parser.add_argument("--physicals", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.full_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.native_convolution:
        torch.backends.cudnn.enabled = False
    if args.cublas:
        torch.backends.cuda.preferred_blas_library("cublas")
    t = tokenizer()
    rows = sorted(load_split("gsm8k"), key=lambda x: len(t.encode(x["question"] + x["response"])))[:8]
    examples = [encode(x, t) for x in rows]
    width = ((max(len(x["input_ids"]) for x in examples) + 63) // 64) * 64
    def batch_fn(items):
        if args.padding == "logical64":
            items = [{**x, "input_ids": x["input_ids"] + [t.pad_token_id] * (width - len(x["input_ids"])),
                      "labels": x["labels"] + [-100] * (width - len(x["labels"]))} for x in items]
            # Keep original lengths for the mask rather than treating padding as text.
            batch = collate(items, t.pad_token_id, "cuda")
            batch["attention_mask"] = (batch["labels"] != -100).long()
            for i, original in enumerate(items):
                actual = next((len(e["input_ids"]) for e in examples if e["id"] == original["id"]))
                batch["attention_mask"][i, :actual] = 1
            return batch
        return collate(items, t.pad_token_id, "cuda")
    model = add_lora(load_model(42))
    if args.fp32_control:
        model.float()
        torch.cuda.empty_cache()
    if args.fp32_delta_norm:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        def promote(fn):
            def call(query, key, value, **kwargs):
                output, state = fn(query.float(), key.float(), value, **kwargs)
                return output.to(query.dtype), state
            return call
        for layer in model.modules():
            if isinstance(layer, q.Qwen3_5GatedDeltaNet):
                layer.chunk_gated_delta_rule = promote(q.torch_chunk_gated_delta_rule)
                layer.recurrent_gated_delta_rule = promote(q.torch_recurrent_gated_delta_rule)
    if args.batch_preserving_linears:
        def independent(self, x):
            if x.ndim == 3:
                result = torch.bmm(x, self.weight.t().unsqueeze(0).expand(x.shape[0], -1, -1))
                return result if self.bias is None else result + self.bias
            count = x.shape[0]
            aligned = ((count + 63) // 64) * 64
            padded = torch.nn.functional.pad(x, (0, 0, 0, aligned - count))
            return torch.nn.functional.linear(padded, self.weight, self.bias)[:count]
        for module in model.modules():
            if type(module) is torch.nn.Linear and module.weight.dtype == torch.bfloat16 and not module.weight.requires_grad:
                module.forward = types.MethodType(independent, module)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    output = {"settings": vars(args), "lengths": [len(e["input_ids"]) for e in examples], "logical_width": width, "cases": []}
    context = sdpa_kernel(SDPBackend.MATH) if args.math_attention else nullcontext()
    with context:
        expected, ref_norms, ref_coef = reference(model, examples, batch_fn, losses, 1.0)
        unweighted_ref, _ = logical_gradients(model, examples, 1, batch_fn, losses)
        clipper = GhostClipper(model, 1.0)
        for physical in args.physicals:
            actual, values = logical_gradients(model, examples, physical, batch_fn, losses, clipper)
            ns = torch.tensor(values["norms"], dtype=torch.float64)
            cs = torch.tensor(values["coefficients"], dtype=torch.float64)
            unweighted, _ = logical_gradients(model, examples, physical, batch_fn, losses)
            row = {"physical": physical, "clipped_error": relative_error(actual, expected),
                   "unclipped_error": relative_error(unweighted, unweighted_ref),
                   "norm_relative_error_max": float(((ns - ref_norms) / ref_norms.clamp_min(1e-30)).abs().max()),
                   "coef_relative_error_max": float(((cs - ref_coef) / ref_coef.clamp_min(1e-30)).abs().max()),
                   "norms": ns.tolist(), "reference_norms": ref_norms.tolist()}
            output["cases"].append(row)
            write_json(args.output, output)
            print(row, flush=True)
        # Two records from the same physical=2 graph: independent per-sample autograd.
        batch = batch_fn(examples[:2])
        clipper.active = False
        ls = losses(model, batch)
        direct = [torch.zeros_like(p) for p in params]
        direct_norms = []
        for i in range(2):
            grad = torch.autograd.grad(ls[i], params, retain_graph=i == 0)
            n = sum(g.double().square().sum() for g in grad).sqrt()
            c = (1.0 / n).clamp(max=1.0)
            direct_norms.append(float(n))
            for dest, g in zip(direct, grad):
                dest.add_(g * c.to(g.dtype))
        values, ns, cs = clipper.backward(batch, losses)
        row = {"same_graph_ghost_norms": ns.cpu().tolist(), "same_graph_direct_norms": direct_norms,
               "same_graph_clipped_error": relative_error([p.grad for p in params], direct)}
        output["same_graph"] = row
        print(row, flush=True)
        write_json(args.output, output)
        clipper.close()


if __name__ == "__main__":
    main()
