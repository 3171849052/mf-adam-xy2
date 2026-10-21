import secrets
import time

import torch

from .common import EXP, ROOT, digest, environment, read_json, seed_all, write_json
from .data import collate, encode, parse_answer, prompt
from .ghost import GhostClipper, assign_private_grads, logical_gradients
from .modeling import losses, trainable_manifest, underlying
from .privacy import calibrate, fixed_schedule, participation


@torch.no_grad()
def evaluate(model, tokenizer, rows, config, output):
    model.eval()
    records = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        # Validation/test loss uses complete answers, regardless of training length.
        item = encode(row, tokenizer)
        loss = float(losses(model, collate([item], tokenizer.pad_token_id, "cuda"))[0])
        inputs = tokenizer(prompt(row["question"]), return_tensors="pt", add_special_tokens=False).to("cuda")
        generated = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
                                   use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id, logits_to_keep=1)
        continuation = generated[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        prediction = parse_answer(text)
        records.append({"id": row["id"], "gold": row["answer"], "prediction": prediction,
                        "correct": prediction == row["answer"], "parse_failed": prediction is None,
                        "loss": loss, "text": text, "generated_tokens": len(continuation),
                        "hit_generation_limit": len(continuation) == config["max_new_tokens"]})
    torch.cuda.synchronize()
    result = {"mode": config["mode"], "count": len(records),
              "accuracy": sum(x["correct"] for x in records) / len(records),
              "validation_loss": sum(x["loss"] for x in records) / len(records),
              "parse_failure_rate": sum(x["parse_failed"] for x in records) / len(records),
              "generation_limit_rate": sum(x["hit_generation_limit"] for x in records) / len(records),
              "seconds": time.perf_counter() - start,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(), "predictions": records}
    write_json(output, result)
    return result


def fit(model, tokenizer, rows, validation, config, dataset, method, lr, seed, output):
    assert method in ("nonprivate", "dp")
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    encoded = [encode(x, tokenizer, config["max_length"]) for x in rows]
    schedule = fixed_schedule(len(rows), config["steps"], config["logical_batch"], seed)
    audit = participation(schedule, len(rows))
    private = calibrate(audit, config["epsilon"], config["delta"], config["clip"]) if method == "dp" else None
    schedule_info = {"sample_ids": [x["id"] for x in rows], "indices": schedule.tolist(),
                     "sha256": digest(schedule.tolist()), **audit}
    write_json(output / "schedule.json", schedule_info)
    write_json(output / "config.json", {**config, "dataset": dataset, "method": method, "lr": lr, "seed": seed})
    write_json(output / "privacy.json", private if private else {"private": False})
    write_json(output / "environment.json", environment())
    write_json(output / "trainable.json", trainable_manifest(model))
    write_json(output / "data_manifest.json", read_json(EXP / "artifacts/data/manifest.json"))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    clipper = GhostClipper(model, config["clip"]) if private else None
    # Publishing the Gaussian seed would invalidate DP. Schedule/model seeds are public.
    noise = torch.Generator(device="cuda").manual_seed(secrets.randbits(63)) if private else None
    batch_fn = lambda items: collate(items, tokenizer.pad_token_id, "cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    token_count = 0
    logs = []
    actual_counts = [0] * len(rows)
    with (output / "train.jsonl").open("w") as log:
        import json
        model.train()
        for step, indices in enumerate(schedule):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            examples = [encoded[int(i)] for i in indices]
            sums, stats = logical_gradients(model, examples, config["physical_batch"], batch_fn, losses, clipper)
            assign_private_grads(params, sums, len(indices), private["noise_std_on_sum"] if private else 0.0, noise)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for i in indices:
                actual_counts[int(i)] += 1
            tokens = sum(len(x["input_ids"]) for x in examples)
            token_count += tokens
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            record = {"step": step + 1, "seconds": seconds, "samples_per_second": len(indices) / seconds,
                      "tokens_per_second": tokens / seconds, "peak_memory_bytes": torch.cuda.max_memory_allocated()}
            if not private:
                record["training_loss"] = stats["loss"]
            log.write(json.dumps(record) + "\n")
            log.flush()
            logs.append(record)
            print(f"{config['mode']} {dataset} {method} seed={seed} lr={lr} step={step+1}/{len(schedule)}", flush=True)
    assert actual_counts == audit["counts"]
    if clipper:
        clipper.close()
    seconds = time.perf_counter() - start
    throughput = {"seconds": seconds, "samples_per_second": schedule.size / seconds,
                  "tokens_per_second": token_count / seconds, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "completed_updates": len(logs)}
    write_json(output / "training.json", throughput)
    write_json(output / "participation_actual.json", {"sample_ids": schedule_info["sample_ids"], "counts": actual_counts})
    model.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)
    metrics = evaluate(model, tokenizer, validation, config, output / "validation.json")
    return {"path": str(output.relative_to(ROOT)), "dataset": dataset, "method": method, "lr": lr, "seed": seed,
            "validation": {k: v for k, v in metrics.items() if k != "predictions"}, "training": throughput,
            "schedule_sha256": schedule_info["sha256"], "privacy": private}
