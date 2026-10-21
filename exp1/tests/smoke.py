"""Small end-to-end CUDA fixture smoke, explicitly not a 9B model result.

This uses real Qwen3.5 hybrid modules at tiny random dimensions. The production
``exp1 run --config .../smoke.json`` separately requires the real pinned model.
"""
import gc
import json
import argparse

import torch

from exp1.common import EXP, ROOT, cache_paths, load_config, read_json, write_json
cache_paths()
from exp1.engine import evaluate, fit
from exp1.pipeline import paired, select
from exp1.tests.fixtures import tiny_qwen


class FixtureTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        # Short deterministic fixture encoding; not an experimental tokenizer.
        return [3 + sum(map(ord, word)) % 100 for word in text.split()]

    def __call__(self, text, return_tensors, add_special_tokens):
        from transformers import BatchEncoding
        ids = torch.tensor([self.encode(text)])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, tokens, skip_special_tokens):
        return "fixture generated tokens " + " ".join(map(str, tokens.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(EXP / "artifacts/fixture_smoke"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(EXP / "configs/smoke.json")
    from exp1.pipeline import checked_output
    out = checked_output(args.output)
    out.mkdir(exist_ok=False)
    tokenizer = FixtureTokenizer()
    rows = [{"id": f"fixture/{i}", "question": f"What is {i} plus 1?", "response": f"Add one.\n#### {i + 1}",
             "answer": str(i + 1)} for i in range(8)]
    model = tiny_qwen("cuda", checkpointing=True)
    base = evaluate(model, tokenizer, rows[:2], config, out / "base_validation.json")
    del model
    fits = []
    for dataset in ("gsm8k", "math10k"):
        for method in ("nonprivate", "dp"):
            model = tiny_qwen("cuda", checkpointing=True)
            result = fit(model, tokenizer, rows, rows[:2], config, dataset, method, 3e-4, 42,
                         out / dataset / method)
            fits.append(result)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    selection = select(fits)
    from peft import PeftModel
    reloaded = PeftModel.from_pretrained(tiny_qwen("cuda", adapters=False), out / "gsm8k/nonprivate/adapter", is_trainable=False)
    # One genuine adapter reload/generation check is part of the smoke.
    restored = evaluate(reloaded, tokenizer, rows[:2], config, out / "reloaded.json")
    before = read_json(out / "gsm8k/nonprivate/validation.json")
    assert [r["text"] for r in before["predictions"]] == [r["text"] for r in restored["predictions"]]
    assert all(x["training"]["completed_updates"] == 1 for x in fits)
    for dataset in ("gsm8k", "math10k"):
        values = [x for x in fits if x["dataset"] == dataset]
        assert values[0]["schedule_sha256"] == values[1]["schedule_sha256"]
    report = {"passed": True, "mode": "smoke", "fixture": "tiny randomly initialized Qwen3.5, synthetic questions and tokenizer",
              "is_9b_checkpoint_result": False, "official_test_accessed": False,
              "fits": fits, "selection": selection, "adapter_reload": True,
              "paired_fixture_check": paired(before, restored)}
    write_json(out / "result.json", report)
    print(json.dumps({"passed": True, "fixture": True, "result": str(out / "result.json")}))


if __name__ == "__main__":
    main()
