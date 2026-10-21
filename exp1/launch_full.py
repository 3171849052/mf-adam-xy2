"""Persistent, observable launcher for the explicitly authorized full experiment.

It runs the existing entry points unchanged and stops on any nonzero exit.
Start in a detached session; status.json distinguishes download/preflight/full.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from .common import EXP, ROOT, load_config, read_json, sha256, write_json

CONDA = "/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/anaconda3/bin/conda"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def inside_exp(path):
    path = Path(path).resolve()
    assert path.is_relative_to(EXP), "Launcher artifacts and run outputs belong inside exp1/"
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", required=True)
    parser.add_argument("--smoke-output", default="exp1/runs/smoke-9b")
    parser.add_argument("--full-output", default="exp1/runs/full")
    parser.add_argument("--transport", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    launch = inside_exp(args.launch_dir)
    smoke = inside_exp(args.smoke_output)
    full = inside_exp(args.full_output)
    assert not launch.exists(), f"Launch directory already exists: {launch}"
    assert not smoke.exists() and not full.exists(), "Choose new run outputs; no runs are overwritten"
    assert Path(CONDA).is_file()
    smoke_config = load_config(EXP / "configs/smoke.json")
    full_config = load_config(EXP / "configs/full.json")
    assert full_config["mode"] == "full" and smoke_config["mode"] == "smoke"
    prefix = [CONDA, "run", "--no-capture-output", "-n", "pndp", "python", "-u"]
    entry = prefix + ["-m", "exp1"]
    stages = [
        ("download", entry + ["download", "--transport", args.transport]),
        ("unit_tests", prefix + ["-m", "unittest", "discover", "-s", "exp1/tests", "-v"]),
        ("smoke_9b", entry + ["run", "--config", str(launch / "smoke.config.json"), "--output", str(smoke)]),
        ("full", entry + ["run", "--config", str(launch / "full.config.json"), "--output", str(full)]),
        ("final", entry + ["final", "--run", str(full)]),
    ]
    if args.dry_run:
        print(json.dumps({"stages": stages, "smoke_config": smoke_config, "full_config": full_config}, indent=2))
        return
    (EXP / ".cache").mkdir(exist_ok=True)
    lock = (EXP / ".cache/full-experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    launch.mkdir(parents=True)
    write_json(launch / "smoke.config.json", smoke_config)
    write_json(launch / "full.config.json", full_config)
    inputs = list(EXP.glob("*.py")) + list((EXP / "tests").glob("*.py")) + [
        EXP / "configs/sources.lock.json", EXP / "artifacts/data/manifest.json",
        launch / "smoke.config.json", launch / "full.config.json",
    ]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in inputs}
    write_json(launch / "inputs.json", {"sha256": hashes, "commands": stages})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    status = {"status": "running", "stage": "initializing", "started_at": now(),
              "supervisor_pid": os.getpid(), "process_group": os.getpgrp(),
              "smoke_output": str(smoke.relative_to(ROOT)), "full_output": str(full.relative_to(ROOT)),
              "full_experiments_started": False, "official_test_evaluated": False, "history": []}

    def save():
        status["updated_at"] = now()
        model_files = ROOT / "model/Qwen3.5-9B-Base"
        expected = read_json(EXP / "configs/sources.lock.json")["model"]["files"]
        weights = {n: v for n, v in expected.items() if n.endswith(".safetensors")}
        def size(path):
            # A completed download atomically renames its partial during polling.
            try:
                return path.stat().st_size
            except FileNotFoundError:
                return 0
        status["weight_bytes"] = sum(min(v["size"], max(size(model_files / n), size(model_files / (n + ".partial"))))
                                      for n, v in weights.items())
        status["expected_weight_bytes"] = sum(v["size"] for v in weights.values())
        status["model_weights_verified"] = (EXP / "artifacts/model_manifest.json").exists()
        status["full_ghost_passed"] = (full / "ghost.json").exists()
        status["full_configuration_frozen"] = (full / "frozen.json").exists()
        status["official_test_evaluated"] = (full / "results.json").exists()
        temporary = launch / "status.tmp.json"
        write_json(temporary, status)
        temporary.replace(launch / "status.json")

    save()
    for name, command in stages:
        for path, expected in hashes.items():
            if sha256(ROOT / path) != expected:
                status.update(status="failed", error=f"Input changed after launch: {path}", finished_at=now())
                save()
                raise RuntimeError(status["error"])
        log_path = launch / f"{name}.log"
        status.update(stage=name, stage_started_at=now(), log=str(log_path.relative_to(ROOT)), command=command)
        status["full_experiments_started"] = name in ("full", "final")
        print(f"{now()} START {name}; log={log_path}", flush=True)
        with log_path.open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = child.pid
            save()
            while child.poll() is None:
                time.sleep(5)
                save()
            code = child.wait()
        status["history"].append({"stage": name, "returncode": code, "finished_at": now(),
                                  "log": str(log_path.relative_to(ROOT))})
        save()
        if code:
            status.update(status="failed", returncode=code, finished_at=now())
            save()
            print(f"{now()} FAILED {name}, exit={code}; later stages were not started", flush=True)
            raise SystemExit(code)
        print(f"{now()} DONE {name}", flush=True)
    status.update(status="complete", child_pid=None, finished_at=now())
    save()
    print(f"{now()} COMPLETE; results={full / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
