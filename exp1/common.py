import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp1"


def cache_paths():
    os.environ["HF_HOME"] = str(ROOT / "model/.cache/huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(ROOT / "data/.cache")
    os.environ["TORCH_HOME"] = str(ROOT / "model/.cache/torch")
    os.environ["TRITON_CACHE_DIR"] = str(EXP / ".cache/triton")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment():
    import platform
    import torch
    packages = ["torch", "transformers", "peft", "opacus", "numpy", "scipy", "huggingface-hub", "safetensors"]
    return {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in packages},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_config(path):
    c = read_json(path)
    assert c["mode"] in ("smoke", "pilot", "full")
    assert c["physical_batch"] in (1, 2, 4, 8)
    assert c["logical_batch"] % c["physical_batch"] == 0
    assert c["max_length"] > 1 and c["steps"] > 0
    assert c["clip"] > 0 and c["epsilon"] > 0 and 0 < c["delta"] < 1
    if c["mode"] == "full":
        assert c["steps"] == 300 and c["logical_batch"] == 64
        assert len(c["seeds"]) == 3 and len(set(c["seeds"])) == 3
        assert c["learning_rates"] == [1e-4, 3e-4, 5e-4]
        assert c["eval_limit"] is None
    if c["mode"] == "smoke":
        assert c["steps"] <= 2 and c["eval_limit"] <= 8 and c["logical_batch"] <= 8
        assert 0 < c["train_limit"] <= 64
    if c["mode"] == "pilot":
        assert c["steps"] < 300 and 0 < c["eval_limit"] < 512
        assert c["train_limit"] is None
    return c
