"""Pinned upstream data, conservative lexical decontamination and token audit."""
from collections import Counter
from fractions import Fraction
import json
import re
import unicodedata
import urllib.request

import numpy as np
from scipy.sparse import csr_matrix

from .common import EXP, ROOT, digest, read_json, sha256, write_json, write_jsonl

PROMPT = "Solve the following math problem. Show your reasoning. End with #### followed by the final answer.\n\nQuestion: {question}\n\nAnswer:\n"
NUMBER = r"[+-]?(?:\d[\d,]*/\d+|\d[\d,]*(?:\.\d+)?|\.\d+)"


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def canonical(value):
    value = str(value).strip().replace(",", "").replace("−", "-")
    if re.fullmatch(r"[A-E]", value):
        return value
    if not re.fullmatch(NUMBER, value):
        raise ValueError(f"Unsupported final answer: {value!r}")
    if "/" in value and int(value.rsplit("/", 1)[1]) == 0:
        raise ValueError(f"Zero answer denominator: {value!r}")
    return str(Fraction(value))


def parse_answer(text):
    """Strict shared rule: last #### marker, then a complete numeric/choice line.

    No last-number fallback: prose without the agreed marker is a parse failure.
    """
    if "####" not in text:
        return None
    line = text.rsplit("####", 1)[1].strip().split("\n")[0].strip()
    if re.fullmatch(NUMBER + r"|[A-E]", line) is None:
        return None
    if "/" in line and int(line.rsplit("/", 1)[1]) == 0:
        return None
    return canonical(line)


def prompt(question):
    return PROMPT.format(question=question.strip())


def gsm_rows(raw, split):
    rows = []
    for i, x in enumerate(raw):
        rationale, answer = x["answer"].rsplit("####", 1)
        answer = canonical(answer)
        rows.append({"id": f"gsm8k/{split}/{i}", "question": x["question"].strip(),
                     "response": rationale.strip() + "\n#### " + answer,
                     "answer": answer, "source": "GSM8K official " + split})
    return rows


def math_rows(raw):
    return [{"id": f"math10k/{i}", "question": (x["instruction"].strip() +
             ("\n" + x["input"].strip() if x["input"].strip() else "")),
             "response": x["output"].strip() + "\n#### " + canonical(x["answer"]),
             "answer": canonical(x["answer"]), "source": "LLM-Adapters math_10k; per-row provenance absent"}
            for i, x in enumerate(raw)]


def lexical_matches(train, held, threshold=0.80):
    """Exact normalized matches plus exhaustive word-set Jaccard >= threshold.

    Sparse multiplication evaluates every pair sharing at least one token.
    Numbers remain tokens. This intentionally over-excludes some templates.
    """
    sets = [set(normalize(x["question"]).split()) for x in train + held]
    vocab = {w: i for i, w in enumerate(sorted(set.union(*sets)))}
    rr, cc = [], []
    for i, words in enumerate(sets):
        rr.extend([i] * len(words))
        cc.extend(vocab[w] for w in words)
    mat = csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)), shape=(len(sets), len(vocab)))
    lengths = np.array([len(s) for s in sets])
    held_norm = [normalize(x["question"]) for x in held]
    norm_lookup = {}
    for j, key in enumerate(held_norm):
        norm_lookup.setdefault(key, []).append(j)
    matches = []
    for start in range(0, len(train), 128):
        intersections = (mat[start:min(start + 128, len(train))] @ mat[len(train):].T).tocoo()
        candidates = {}
        for r, col, inter in zip(intersections.row, intersections.col, intersections.data):
            i = start + int(r)
            score = float(inter / (lengths[i] + lengths[len(train) + col] - inter))
            if score >= threshold:
                candidates[(i, int(col))] = score
        for i in range(start, min(start + 128, len(train))):
            for j in norm_lookup.get(normalize(train[i]["question"]), []):
                candidates[(i, j)] = 1.0
        for (i, j), score in sorted(candidates.items()):
            matches.append({"train_id": train[i]["id"], "held_id": held[j]["id"],
                            "kind": "normalized_exact" if normalize(train[i]["question"]) == held_norm[j] else "near_jaccard",
                            "score": score})
    return matches


def prepare():
    lock = read_json(EXP / "configs/sources.lock.json")
    for item in lock["sources"].values():
        path = ROOT / item["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(item["url"], timeout=120) as f:
                path.write_bytes(f.read())
        if sha256(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
    with (ROOT / lock["sources"]["gsm8k_train"]["path"]).open() as f:
        train = gsm_rows([json.loads(x) for x in f], "train")
    with (ROOT / lock["sources"]["gsm8k_test"]["path"]).open() as f:
        test = gsm_rows([json.loads(x) for x in f], "test")
    idx = set(np.random.default_rng(20261020).permutation(len(train))[:512].tolist())
    validation = [x for i, x in enumerate(train) if i in idx]
    gsm = [x for i, x in enumerate(train) if i not in idx]
    math = math_rows(read_json(ROOT / lock["sources"]["math10k"]["path"]))
    matches = lexical_matches(math, validation + test)
    removed = {m["train_id"] for m in matches}
    filtered = [x for x in math if x["id"] not in removed]
    # GSM8K official split itself can contain lexical near duplicates. Record it.
    gsm_matches = lexical_matches(gsm, validation + test)
    gsm_norms = {normalize(x["question"]) for x in train}
    sources = Counter("matches_gsm8k_train" if normalize(x["question"]) in gsm_norms else
                      "multiple_choice_likely_AQuA" if "The options are:" in x["question"] else
                      "unverified_likely_MAWPS_or_modified_GSM8K" for x in math)
    report = {"split_seed": 20261020, "validation_size": 512,
              "counts": {"gsm8k_train": len(gsm), "validation": len(validation), "test": len(test),
                         "math10k_raw": len(math), "math10k_filtered": len(filtered), "math10k_removed": len(removed)},
              "near_duplicate_rule": "NFKC/casefold/word normalization; exhaustive word-set Jaccard >= 0.80",
              "math_matches": matches, "official_gsm_split_matches": gsm_matches,
              "provenance_heuristics": dict(sources),
              "within_dataset_normalized_duplicate_excess": {k: len(v) - len({normalize(x['question']) for x in v})
                                                          for k, v in [("gsm8k", gsm), ("math10k", filtered)]},
              "risks": ["Lexical audit cannot exclude semantic paraphrases, translations or pretraining contamination.",
                        "Math-10K includes GSM8K, MAWPS and 1000 AQuA per the pinned official README; no reliable per-row provenance.",
                        "Generated Math-10K rationales can contradict supplied labels; retained as upstream, with a final supplied-label marker.",
                        "Held-out questions are accessed for decontamination only; no official test predictions or metrics before freeze.",
                        "Public data-dependent deduplication is outside the per-fit fixed-slot DP guarantee.",
                        "Internal validation and official tests are public evaluation data, not protected training records."]}
    manifest = {"sources_lock_sha256": sha256(EXP / "configs/sources.lock.json"), "splits": {}}
    for name, rows in [("gsm8k", gsm), ("math10k", filtered), ("validation", validation), ("test", test)]:
        path = ROOT / f"data/processed/{name}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "count": len(rows)}
    write_json(EXP / "artifacts/data/audit.json", report)
    write_json(EXP / "artifacts/data/manifest.json", manifest)
    return report["counts"]


def load_split(name):
    manifest = read_json(EXP / "artifacts/data/manifest.json")
    assert manifest["sources_lock_sha256"] == sha256(EXP / "configs/sources.lock.json")
    item = manifest["splits"][name]
    path = ROOT / item["path"]
    assert sha256(path) == item["sha256"], f"Prepared data changed: {path}"
    with path.open() as f:
        return [json.loads(x) for x in f]


def encode(row, tokenizer, max_length=None):
    prefix = tokenizer.encode(prompt(row["question"]), add_special_tokens=False)
    answer = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prefix + answer
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens > max_length={max_length}; run audit and explicitly adjust config")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + answer, "id": row["id"], "target_tokens": len(answer)}


def length_audit(tokenizer, config):
    report = {"max_length": config["max_length"], "prompt_sha256": digest(PROMPT),
              "model_revision": read_json(EXP / "configs/sources.lock.json")["model"]["revision"],
              "data_manifest_sha256": sha256(EXP / "artifacts/data/manifest.json"), "splits": {}}
    # Official test is deliberately not tokenized until the final frozen stage.
    for split in ("gsm8k", "math10k", "validation"):
        rows = load_split(split)
        enc = [encode(x, tokenizer) for x in rows]
        lengths = [len(x["input_ids"]) for x in enc]
        over = [{"id": e["id"], "tokens": len(e["input_ids"]),
                 "lost_target_tokens": min(e["target_tokens"], len(e["input_ids"]) - config["max_length"])}
                for e in enc if len(e["input_ids"]) > config["max_length"]]
        report["splits"][split] = {"count": len(rows), "max_tokens": max(lengths),
                                    "quantiles": np.quantile(lengths, [0.5, 0.9, 0.95, 0.99, 1]).tolist(),
                                    "overlength_count": len(over), "truncation_fraction": len(over) / len(rows),
                                    "overlength": over}
    write_json(EXP / f"artifacts/data/lengths-{config['mode']}-{config['max_length']}.json", report)
    return report


def collate(rows, pad_id, device):
    import torch
    n = max(len(x["input_ids"]) for x in rows)
    ids = [x["input_ids"] + [pad_id] * (n - len(x["input_ids"])) for x in rows]
    labels = [x["labels"] + [-100] * (n - len(x["labels"])) for x in rows]
    mask = [[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in rows]
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": torch.tensor(mask, device=device),
            "labels": torch.tensor(labels, device=device)}
