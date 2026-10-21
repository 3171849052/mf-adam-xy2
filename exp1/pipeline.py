import gc
import statistics
from pathlib import Path

import torch
from scipy.stats import binomtest

from .common import EXP, ROOT, digest, environment, load_config, read_json, sha256, write_json
from .correctness import validate
from .data import collate, encode, length_audit, load_split
from .engine import evaluate, fit
from .modeling import add_lora, load_model, losses, require_model, tokenizer, trainable_manifest
from .privacy import calibrate, compose_runs, fixed_schedule, participation


def checked_output(path):
    path = Path(path).resolve()
    assert path.is_relative_to(EXP), "All experiment outputs must be inside exp1/"
    return path


def plan(config, output):
    """Materialize public schedules and privacy audits without loading a model."""
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    assert config["mode"] != "smoke", "Smoke uses explicitly length-filtered subsets inside run"
    audits = []
    for name in ("gsm8k", "math10k"):
        rows = load_split(name)
        for seed in [config["search_seed"]] + config["seeds"]:
            indices = fixed_schedule(len(rows), config["steps"], config["logical_batch"], seed)
            private = calibrate(participation(indices, len(rows)), config["epsilon"], config["delta"], config["clip"])
            write_json(output / f"{name}-{seed}.json", {"sample_ids": [r["id"] for r in rows], "indices": indices.tolist(),
                       "schedule_sha256": digest(indices.tolist()), "privacy": private})
            audits.append({"dataset": name, "seed": seed, **{k: v for k, v in private.items() if k not in ("counts", "squared_multiplicities")}})
    write_json(output / "summary.json", {"config": config, "audits": audits, "training_started": False})
    return audits


def check_ghost(config, output):
    require_model()
    t = tokenizer()
    rows = sorted(load_split("gsm8k"), key=lambda x: len(t.encode(x["question"] + x["response"])))[:8]
    examples = [encode(x, t, config["max_length"]) for x in rows]
    report = {"mode": config["mode"], "model_manifest_sha256": sha256(EXP / "artifacts/model_manifest.json"),
              "environment": environment(), "checks": []}
    model = add_lora(load_model(config["search_seed"]))
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        values = validate(model, examples, lambda xs: collate(xs, t.pad_token_id, "cuda"), losses, config["clip"])
        report["checks"].append({"checkpointing": checkpointing, **values})
    report["trainable"] = trainable_manifest(model)
    report["passed"] = all(x["passed"] for x in report["checks"])
    write_json(checked_output(output), report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def select(search):
    chosen = {}
    for dataset in ("gsm8k", "math10k"):
        for method in ("nonprivate", "dp"):
            candidates = [x for x in search if x["dataset"] == dataset and x["method"] == method]
            assert candidates
            winner = min(candidates, key=lambda x: (-x["validation"]["accuracy"], x["validation"]["validation_loss"], x["lr"]))
            chosen[f"{dataset}/{method}"] = {"lr": winner["lr"], "selected_from": winner["path"],
                                            "rule": "max validation answer accuracy; tie: min loss, then min lr"}
    return chosen


def freeze(run):
    config = load_config(run / "config.json")
    assert read_json(run / "ghost.json")["passed"], "Ghost correctness gate must pass before freezing"
    results = read_json(run / "fits.json")
    assert len(results) == 4 * len(config["seeds"])
    files = {str(p.relative_to(ROOT)): sha256(p) for p in [run / "config.json", run / "selection.json",
             run / "fits.json", run / "search.json", run / "privacy_composition.json", run / "base_validation.json", run / "ghost.json",
             EXP / "artifacts/data/manifest.json", EXP / "configs/sources.lock.json", EXP / "artifacts/model_manifest.json"]}
    # Include source and adapters in the freeze. Later evaluation verifies all hashes.
    for p in sorted(EXP.glob("*.py")):
        files[str(p.relative_to(ROOT))] = sha256(p)
    for fit_result in results:
        for p in sorted((ROOT / fit_result["path"] / "adapter").glob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = sha256(p)
    for key in ("gsm8k/nonprivate", "math10k/nonprivate"):
        ds = key.split("/")[0]
        for seed in config["seeds"]:
            pair = [x for x in results if x["dataset"] == ds and x["seed"] == seed]
            assert len(pair) == 2 and pair[0]["schedule_sha256"] == pair[1]["schedule_sha256"]
    body = {"mode": config["mode"], "files": files, "configuration_frozen": True,
            "official_test_evaluated": False, "selection": read_json(run / "selection.json"),
            "fit_count": len(results), "test_access": "questions only for decontamination before freeze"}
    write_json(run / "frozen.json", {**body, "sha256": digest(body)})
    return body


def run(config_path, output):
    config = load_config(config_path)
    output = checked_output(output)
    assert not output.exists(), "Choose a new run directory; existing runs are never overwritten"
    require_model()
    t = tokenizer()
    lengths = length_audit(t, config)
    if config["mode"] != "smoke":
        assert all(x["overlength_count"] == 0 for x in lengths["splits"].values()), "Length audit failed: explicitly raise max_length in a new config"
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    check_ghost(config, output / "ghost.json")
    validation = load_split("validation")
    if config["eval_limit"] is not None:
        validation = validation[:config["eval_limit"]]
    base = load_model(config["search_seed"])
    evaluate(base, t, validation, config, output / "base_validation.json")
    del base
    gc.collect()
    torch.cuda.empty_cache()
    datasets = {name: load_split(name) for name in ("gsm8k", "math10k")}
    if config["mode"] == "smoke":
        # Explicit small, complete answers. No truncation and no official test access.
        datasets = {name: [r for r in rows if len(encode(r, t)["input_ids"]) <= config["max_length"]][:config["train_limit"]]
                    for name, rows in datasets.items()}
        write_json(output / "smoke_subsets.json", {k: [r["id"] for r in v] for k, v in datasets.items()})
    search = []
    for dataset, rows in datasets.items():
        for method in ("nonprivate", "dp"):
            for lr in config["learning_rates"]:
                model = add_lora(load_model(config["search_seed"], config["gradient_checkpointing"]))
                result = fit(model, t, rows, validation, config, dataset, method, lr, config["search_seed"],
                             output / "search" / dataset / method / f"lr-{lr}")
                search.append(result)
                write_json(output / "search.json", search)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    selection = select(search)
    write_json(output / "selection.json", selection)
    fits = []
    for dataset, rows in datasets.items():
        for method in ("nonprivate", "dp"):
            lr = selection[f"{dataset}/{method}"]["lr"]
            for seed in config["seeds"]:
                model = add_lora(load_model(seed, config["gradient_checkpointing"]))
                result = fit(model, t, rows, validation, config, dataset, method, lr, seed,
                             output / "fits" / dataset / method / f"seed-{seed}")
                fits.append(result)
                write_json(output / "fits.json", fits)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    all_dp = [x["privacy"] for x in search + fits if x["method"] == "dp"]
    write_json(output / "privacy_composition.json", compose_runs(all_dp, config["delta"]))
    freeze(output)
    return {"run": str(output), "mode": config["mode"], "frozen": True, "official_test_evaluated": False}


def verify_frozen(run):
    f = read_json(run / "frozen.json")
    assert f["sha256"] == digest({k: v for k, v in f.items() if k != "sha256"})
    for path, value in f["files"].items():
        assert sha256(ROOT / path) == value, f"Frozen artifact changed: {path}"
    return f


def final(run_path):
    from peft import PeftModel
    run = checked_output(run_path)
    verify_frozen(run)
    config = load_config(run / "config.json")
    assert config["mode"] == "full", "Official GSM8K test is permitted only for a frozen full run"
    output = run / "final"
    output.mkdir(exist_ok=False)
    t = tokenizer()
    rows = load_split("test")
    model = load_model(config["search_seed"])
    evaluate(model, t, rows, config, output / "base.json")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for item in read_json(run / "fits.json"):
        model = PeftModel.from_pretrained(load_model(item["seed"]), ROOT / item["path"] / "adapter", is_trainable=False)
        evaluate(model, t, rows, config, output / f"{item['dataset']}-{item['method']}-{item['seed']}.json")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return summarize(run)


def paired(a, b):
    aa, bb = a["predictions"], b["predictions"]
    assert [x["id"] for x in aa] == [x["id"] for x in bb]
    wins = sum(x["correct"] and not y["correct"] for x, y in zip(aa, bb))
    losses_ = sum(not x["correct"] and y["correct"] for x, y in zip(aa, bb))
    return {"a_only_correct": wins, "b_only_correct": losses_, "difference": (wins - losses_) / len(aa),
            "mcnemar_exact_p": float(binomtest(wins, wins + losses_, 0.5).pvalue) if wins + losses_ else 1.0}


def summarize(run):
    verify_frozen(run)
    config = load_config(run / "config.json")
    assert config["mode"] == "full"
    groups, pairs = {}, {}
    metrics = {}
    for dataset in ("gsm8k", "math10k"):
        for method in ("nonprivate", "dp"):
            rows = [read_json(run / "final" / f"{dataset}-{method}-{seed}.json") for seed in config["seeds"]]
            metrics[dataset, method] = rows
            groups[f"{dataset}/{method}"] = {key: {"mean": statistics.mean(x[key] for x in rows),
                                                       "std": statistics.stdev(x[key] for x in rows)}
                                                for key in ("accuracy", "validation_loss", "parse_failure_rate", "generation_limit_rate")}
    for method in ("nonprivate", "dp"):
        pairs[method] = [{"seed": seed, **paired(a, b)} for seed, a, b in
                         zip(config["seeds"], metrics["math10k", method], metrics["gsm8k", method])]
    for ds in ("gsm8k", "math10k"):
        pairs[f"{ds}:dp_vs_nonprivate"] = [{"seed": seed, **paired(a, b)} for seed, a, b in
                         zip(config["seeds"], metrics[ds, "dp"], metrics[ds, "nonprivate"])]
    conclusions = {}
    for method in ("nonprivate", "dp"):
        deltas = [p["difference"] for p in pairs[method]]
        mean = statistics.mean(deltas)
        conclusions[method] = {"preferred_by_mean_accuracy": "math10k" if mean > 0 else "gsm8k" if mean < 0 else "tie",
                               "paired_seed_difference_mean": mean, "paired_seed_difference_std": statistics.stdev(deltas),
                               "consistent_across_three_seeds": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
                               "interpretation": "descriptive 3-seed comparison; lexical decontamination cannot guarantee absence of contamination"}
    fits = read_json(run / "fits.json")
    training = {}
    for ds in ("gsm8k", "math10k"):
        for method in ("nonprivate", "dp"):
            values = [x["training"] for x in fits if x["dataset"] == ds and x["method"] == method]
            training[f"{ds}/{method}"] = {k: {"mean": statistics.mean(x[k] for x in values), "std": statistics.stdev(x[k] for x in values)}
                                            for k in ("samples_per_second", "tokens_per_second", "peak_memory_bytes")}
    result = {"mode": "full", "groups": groups, "paired": pairs, "training": training,
              "base": {k: v for k, v in read_json(run / "final/base.json").items() if k != "predictions"},
              "dataset_conclusions": conclusions, "privacy_composition": read_json(run / "privacy_composition.json")}
    write_json(run / "results.json", result)
    return result
