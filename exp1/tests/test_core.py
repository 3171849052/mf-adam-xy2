import math
import unittest

import numpy as np
import torch

from exp1.common import cache_paths, digest, EXP, load_config, read_json
cache_paths()
from exp1.data import canonical, collate, encode, lexical_matches, normalize, parse_answer
from exp1.ghost import GhostClipper, assign_private_grads
from exp1.modeling import token_mean_losses
from exp1.privacy import calibrate, compose_runs, fixed_schedule, gdp_delta, participation
from exp1.pipeline import paired, select

torch.set_num_threads(1)


class SchedulePrivacyTests(unittest.TestCase):
    def test_permutations_and_counts(self):
        s = fixed_schedule(73, 300, 64, 42)
        self.assertTrue(np.array_equal(s, fixed_schedule(73, 300, 64, 42)))
        self.assertFalse(np.array_equal(s, fixed_schedule(73, 300, 64, 43)))
        flat = s.ravel()
        for i in range(len(flat) // 73):
            self.assertEqual(sorted(flat[i * 73:(i + 1) * 73].tolist()), list(range(73)))
        audit = participation(s, 73)
        self.assertEqual(sum(audit["counts"]), 300 * 64)
        self.assertEqual(audit["max_participations"], math.ceil(300 * 64 / 73))

    def test_batch_boundary_duplicate_sensitivity(self):
        s = np.array([[0, 0, 1, 2], [0, 1, 1, 2]])
        audit = participation(s, 3)
        self.assertEqual(audit["counts"], [3, 3, 2])
        self.assertEqual(audit["squared_multiplicities"], [5, 5, 2])
        result = calibrate(audit)
        self.assertAlmostEqual(result["noise_multiplier"] * result["mu"], math.sqrt(5))
        self.assertAlmostEqual(result["epsilon"], 6, places=9)
        self.assertAlmostEqual(gdp_delta(6, result["mu"]), 1e-5, places=12)
        self.assertFalse(result["sampling_amplification"])

    def test_composition_and_noise_scaling(self):
        one = calibrate(participation(np.array([[0, 1, 2, 3]]), 4), clip=2)
        many = calibrate(participation(np.array([[0, 1, 2, 3]] * 9), 4), clip=2)
        self.assertAlmostEqual(many["noise_multiplier"], 3 * one["noise_multiplier"])
        self.assertAlmostEqual(one["noise_std_on_mean"], one["noise_std_on_sum"] / 4)
        self.assertGreater(compose_runs([one] * 3)["epsilon"], 6)

    def test_single_noise_on_sum_then_average(self):
        p = torch.nn.Parameter(torch.zeros(32))
        acc = torch.ones(32) * 8
        g1, g2 = (torch.Generator().manual_seed(234) for _ in range(2))
        expected = (acc + 2 * torch.randn(32, generator=g1)) / 8
        assign_private_grads([p], [acc], 8, 2, g2)
        torch.testing.assert_close(p.grad, expected)


class DataLossTests(unittest.TestCase):
    def test_normalization_and_near_decontamination(self):
        held = [{"id": "val/0", "question": "Mary has 12 red apples. She buys 5 more apples. How many apples now?"}]
        train = [{"id": "exact", "question": "MARY has 12 red apples! She buys 5 more apples; how many apples now?"},
                 {"id": "near", "question": "Mary has 13 red apples. She buys 5 more apples. How many apples now?"},
                 {"id": "other", "question": "What is the square root of 144?"}]
        matches = lexical_matches(train, held)
        self.assertEqual({x["train_id"] for x in matches}, {"exact", "near"})
        self.assertEqual(matches[0]["kind"], "normalized_exact")
        self.assertEqual(normalize("Ａ  １２!"), "a 12")

    def test_parser(self):
        self.assertEqual(parse_answer("reason 999\n#### 1,200.00"), "1200")
        self.assertEqual(parse_answer("#### -1.5"), "-3/2")
        self.assertEqual(parse_answer("#### 3/2"), "3/2")
        self.assertEqual(parse_answer("#### A"), "A")
        self.assertIsNone(parse_answer("The answer is 12"))
        self.assertIsNone(parse_answer("#### 12 and 13"))
        self.assertIsNone(parse_answer("#### 0\n#### nonsense"))
        self.assertIsNone(parse_answer("#### 1/0"))
        self.assertIsNone(parse_answer("#### 1.5/3"))

    def test_mask_and_no_silent_truncation(self):
        class Tokenizer:
            eos_token_id = 7
            def encode(self, text, add_special_tokens):
                return list(range(len(text)))
        row = {"id": "test", "question": "q", "response": "ans"}
        e = encode(row, Tokenizer())
        self.assertEqual(e["labels"][-4:], [0, 1, 2, 7])
        self.assertTrue(all(x == -100 for x in e["labels"][:-4]))
        with self.assertRaisesRegex(ValueError, "explicitly adjust"):
            encode(row, Tokenizer(), 3)

    def test_sample_mean_instead_of_token_mean(self):
        logits = torch.tensor([[[0., 0.], [0., 0.], [0., 0.]],
                               [[3., 0.], [3., 0.], [3., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, -100, 1], [-100, 0, 0]])
        losses = token_mean_losses(logits, labels)
        torch.testing.assert_close(losses[0], torch.tensor(math.log(2)))
        torch.testing.assert_close(losses[1], torch.log1p(torch.exp(torch.tensor(-3.))))
        losses.sum().backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[:, -1].abs().sum()), 0.)

    def test_config_modes_and_heldout_gate(self):
        for mode in ("smoke", "pilot", "full"):
            self.assertEqual(load_config(EXP / f"configs/{mode}.json")["mode"], mode)
        from exp1.pipeline import final
        import tempfile
        from pathlib import Path
        from exp1.common import write_json
        with tempfile.TemporaryDirectory(dir=EXP / "tests") as d:
            path = Path(d)
            body = {"files": {}, "configuration_frozen": True}
            write_json(path / "frozen.json", {**body, "sha256": digest(body)})
            write_json(path / "config.json", read_json(EXP / "configs/smoke.json"))
            with self.assertRaisesRegex(AssertionError, "frozen full"):
                final(path)


class SelectionTests(unittest.TestCase):
    def test_selection_uses_validation(self):
        rows = [{"dataset": ds, "method": method, "lr": lr, "path": str(lr),
                 "validation": {"accuracy": .3, "validation_loss": loss}}
                for ds in ("gsm8k", "math10k") for method in ("nonprivate", "dp")
                for lr, loss in ((1e-4, 2.0), (3e-4, 1.9))]
        self.assertTrue(all(x["lr"] == 3e-4 for x in select(rows).values()))

    def test_paired_correctness(self):
        a = {"predictions": [{"id": str(i), "correct": x} for i, x in enumerate([1, 1, 0, 1])]}
        b = {"predictions": [{"id": str(i), "correct": x} for i, x in enumerate([1, 0, 1, 0])]}
        result = paired(a, b)
        self.assertEqual((result["a_only_correct"], result["b_only_correct"]), (2, 1))
        self.assertEqual(result["difference"], .25)


if __name__ == "__main__":
    unittest.main()
