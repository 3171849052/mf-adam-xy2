import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import EXP, ROOT, read_json, seed_all, sha256, write_json

MODEL_DIR = ROOT / "model/Qwen3.5-9B-Base"


def structure_audit():
    """Inspect actual 9B shapes/keys on meta; never claim weights were loaded."""
    from accelerate import init_empty_weights
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from .ghost import GhostClipper
    config = Qwen3_5Config.from_dict(read_json(MODEL_DIR / "config.json"))
    with init_empty_weights():
        model = select_torch_backend(Qwen3_5ForConditionalGeneration(config))
        expected = set(model.state_dict())
        present = set(read_json(MODEL_DIR / "model.safetensors.index.json")["weight_map"])
        assert not expected - present, f"Checkpoint misses model keys: {expected-present}"
        assert all(k.startswith("mtp.") for k in present - expected)
        total = sum(p.numel() for p in model.parameters())
        model.requires_grad_(False)
        model = add_lora(model.to(dtype=torch.bfloat16))
        clip = GhostClipper(model)
        report = {"weights_loaded": False, "method": "meta shape/key inspection against pinned official index",
                  "backbone_parameters": total, "unused_checkpoint_keys": sorted(present - expected),
                  "ghost_linear_modules": len(clip.layers), "trainable": trainable_manifest(model)}
        clip.close()
    write_json(EXP / "artifacts/model_structure.json", report)
    return {k: v for k, v in report.items() if k != "trainable"}


def select_torch_backend(model):
    """Select the tested upstream torch implementation explicitly, never auto-switch."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    for layer in model.modules():
        if isinstance(layer, q.Qwen3_5GatedDeltaNet):
            assert isinstance(layer.norm, q.Qwen3_5RMSNormGated), "This experiment certifies the torch gated norm; fused norm requires separate validation"
            layer.causal_conv1d_fn = None
            layer.causal_conv1d_update = q.torch_causal_conv1d_update
            layer.chunk_gated_delta_rule = q.torch_chunk_gated_delta_rule
            layer.recurrent_gated_delta_rule = q.torch_recurrent_gated_delta_rule
    return model


def download_model(transport="huggingface"):
    from huggingface_hub import snapshot_download
    lock = read_json(EXP / "configs/sources.lock.json")["model"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if transport == "huggingface":
        snapshot_download(lock["id"], revision=lock["revision"], local_dir=MODEL_DIR,
                          ignore_patterns=["*.md", ".gitattributes", "*preprocessor*", "LICENSE"], max_workers=4)
    elif transport == "modelscope":
        # Explicit byte transport for the SAME frozen weights, not another model.
        # Content hashes, including config/tokenizer, must match the HF revision.
        from concurrent.futures import ThreadPoolExecutor
        from .downloads import resume_ranges
        def fetch(item):
            name, expected = item
            path = MODEL_DIR / name
            if path.exists():
                assert sha256(path) == expected["sha256"], f"SHA256 mismatch: {path}"
                return
            url = f"https://www.modelscope.cn/models/Qwen/Qwen3.5-9B-Base/resolve/master/{name}"
            partial = path.with_suffix(path.suffix + ".partial")
            print(f"ModelScope byte transport: {name}; expected SHA256={expected['sha256']}", flush=True)
            resume_ranges(url, partial, expected["size"])
            assert sha256(partial) == expected["sha256"], f"Mirror differs from frozen HF content: {name}"
            partial.rename(path)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, lock["files"].items()))
    else:
        raise ValueError(f"Unknown download transport: {transport}")
    manifest = {}
    for path in sorted(MODEL_DIR.iterdir()):
        if path.is_file():
            value = sha256(path)
            if path.name in lock["files"]:
                assert value == lock["files"][path.name]["sha256"], f"Model SHA256 mismatch: {path}"
            manifest[path.name] = value
    assert all(name in manifest for name in lock["files"])
    write_json(EXP / "artifacts/model_manifest.json", {"id": lock["id"], "revision": lock["revision"],
               "transport": transport, "sha256": manifest})


def tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    t.pad_token = t.eos_token
    return t


def require_model():
    manifest_path = EXP / "artifacts/model_manifest.json"
    assert manifest_path.exists(), "Pinned 9B weights are not downloaded and SHA256-verified; run: python -m exp1 download"
    lock = read_json(EXP / "configs/sources.lock.json")["model"]
    manifest = read_json(manifest_path)
    assert manifest["revision"] == lock["revision"]
    for name, value in lock["files"].items():
        path = MODEL_DIR / name
        assert manifest["sha256"][name] == value["sha256"]
        assert path.exists() and path.stat().st_size == value["size"], f"Incomplete model: {name}; run python -m exp1 download"


def load_model(seed, checkpointing=False):
    from transformers import Qwen3_5ForConditionalGeneration
    assert torch.cuda.device_count() == 1, "Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0"
    assert torch.cuda.is_bf16_supported(), "BF16-capable CUDA GPU required"
    require_model()
    seed_all(seed)
    # Explicit multimodal checkpoint class; only its language path participates.
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        output_loading_info=True)
    assert not info["missing_keys"] and not info.get("mismatched_keys"), str(info)
    assert all(k.startswith("mtp.") for k in info["unexpected_keys"]), str(info)
    model = select_torch_backend(model)
    model.requires_grad_(False)
    model.config.text_config.use_cache = False
    model.to("cuda")
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def add_lora(model):
    from peft import LoraConfig, get_peft_model
    targets = [name for name, layer in model.named_modules()
               if isinstance(layer, torch.nn.Linear) and name.startswith("model.language_model.layers.")]
    assert targets and any("linear_attn" in x for x in targets) and any("self_attn" in x for x in targets)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                                          target_modules=targets, task_type="CAUSAL_LM"))
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert ".lora_A." in name or ".lora_B." in name, name
            # BF16 backbone and compute; FP32 LoRA master parameters/Adam states.
            p.data = p.data.float()
    return model


def underlying(model):
    from peft import PeftModel
    return model.get_base_model() if isinstance(model, PeftModel) else model


def token_mean_losses(logits, labels):
    """Reference loss for tests and small fixtures: each sample has equal weight."""
    shifted = labels[:, 1:]
    counts = (shifted != -100).sum(1)
    assert (counts > 0).all(), "Every sample must have a supervised answer token"
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), shifted, reduction="none", ignore_index=-100)
    return losses.sum(1) / counts


def losses(model, batch):
    """Response-only cross entropy, chunked over the frozen vocabulary head.

    Avoid a [B,T,248320] float32 allocation. Checkpointing the head chunks
    preserves exact loss/gradient semantics and uses PyTorch's implementation.
    """
    base = underlying(model)
    hidden = base.model.language_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                       use_cache=False).last_hidden_state[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    counts = mask.sum(1)
    assert (counts > 0).all()
    h = hidden[mask]
    target = labels[mask]
    owners = torch.arange(len(labels), device=labels.device)[:, None].expand_as(labels)[mask]
    sums = torch.zeros(len(labels), device=labels.device, dtype=torch.float32)

    def ce(chunk, gold):
        return F.cross_entropy(base.lm_head(chunk).float(), gold, reduction="none")

    for start in range(0, len(target), 64):
        hs, ys = h[start:start + 64], target[start:start + 64]
        values = checkpoint(ce, hs, ys, use_reentrant=False) if torch.is_grad_enabled() and hs.requires_grad else ce(hs, ys)
        sums = sums.index_add(0, owners[start:start + 64], values)
    return sums / counts


def trainable_manifest(model):
    return {"parameters": {n: {"shape": list(p.shape), "dtype": str(p.dtype)}
                           for n, p in model.named_parameters() if p.requires_grad},
            "count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "backbone_dtype": str(underlying(model).model.language_model.embed_tokens.weight.dtype),
            "checkpointing": underlying(model).is_gradient_checkpointing,
            "vision": "frozen, unused for text", "delta_backend": "explicit upstream torch reference (conv, chunk and recurrent)",
            "targets": "all language decoder nn.Linear, including gated delta projections"}
