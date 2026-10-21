import unittest
import torch

from exp1.common import EXP, cache_paths, write_json
cache_paths()
from exp1.correctness import validate
from exp1.data import collate
from exp1.ghost import GhostClipper
from exp1.modeling import losses, token_mean_losses, underlying
from exp1.tests.fixtures import examples, tiny_qwen

torch.set_num_threads(1)


class GhostTests(unittest.TestCase):
    def test_hybrid_qwen_cpu_and_checkpointing(self):
        reports = []
        for cp in (False, True):
            model = tiny_qwen(checkpointing=cp)
            result = validate(model, examples(), lambda xs: collate(xs, 2, "cpu"), losses, clip=0.1, rtol=1e-4)
            reports.append({"checkpointing": cp, **result})
        write_json(EXP / "artifacts/tests/ghost_cpu.json", {"fixture": "random tiny Qwen3.5; not the 9B checkpoint", "checks": reports})

    def test_chunked_loss_and_gradients(self):
        model = tiny_qwen()
        # Multiple vocabulary-head chunks, unequal answer lengths and right padding.
        rows = [{"id": f"long/{i}", "input_ids": [3 + j % 100 for j in range(137 + i)],
                 "labels": [-100] * 20 + [3 + j % 100 for j in range(20, 137 + i)]} for i in range(2)]
        batch = collate(rows, 2, "cpu")
        chunked = losses(model, batch)
        ordinary = token_mean_losses(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits, batch["labels"])
        torch.testing.assert_close(chunked, ordinary, rtol=1e-6, atol=1e-6)
        params = [p for p in model.parameters() if p.requires_grad]
        a = torch.autograd.grad(chunked.sum(), params)
        b = torch.autograd.grad(ordinary.sum(), params)
        for aa, bb in zip(a, b):
            torch.testing.assert_close(aa, bb, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_bfloat16_hybrid_qwen_gpu(self):
        reports = []
        for cp in (False, True):
            model = tiny_qwen("cuda", cp)
            result = validate(model, examples(), lambda xs: collate(xs, 2, "cuda"), losses, clip=0.1, rtol=.03)
            reports.append({"checkpointing": cp, **result})
        write_json(EXP / "artifacts/tests/ghost_bf16_gpu.json", {"fixture": "random tiny Qwen3.5 on A800; not 9B checkpoint", "checks": reports})

    def test_reject_uncovered_trainable_parameters(self):
        with self.assertRaisesRegex(AssertionError, "Unsupported trainable"):
            GhostClipper(torch.nn.Linear(2, 3))

    def test_reject_shared_adapter_execution(self):
        class Reused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 3, bias=False)})
            def forward(self, x):
                return self.lora_A["default"](self.lora_A["default"](x))
        model = Reused()
        clip = GhostClipper(model)
        with self.assertRaisesRegex(AssertionError, "Reused/tied"):
            clip.backward({"input_ids": torch.ones(2, 3)}, lambda m, b: m(b["input_ids"]).square().mean(1))
        clip.close()


if __name__ == "__main__":
    unittest.main()
