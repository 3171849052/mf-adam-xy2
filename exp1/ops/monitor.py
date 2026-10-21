"""Read-only experiment health monitor; writes only its own observations."""
import argparse
from collections import deque
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text())


def size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0  # A downloader may atomically rename a completed partial.


def process(pid):
    if pid is None:
        return {"alive": False, "pid": None}
    directory = Path("/proc") / str(pid)
    try:
        cmd = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        state = (directory / "stat").read_text().split(") ", 1)[1].split()[0]
    except FileNotFoundError:
        return {"alive": False, "pid": pid}
    return {"pid": pid, "alive": state != "Z", "state": state, "command": cmd}


def gpu():
    command = ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out after 15 seconds"}
    if result.returncode:
        return {"error": result.stderr.strip(), "returncode": result.returncode}
    name, memory, utilization, power = result.stdout.strip().split(",")
    return {"name": name.strip(), "memory_mib": float(memory), "utilization_percent": float(utilization), "power_w": float(power)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    path = Path(args.status).resolve()
    assert path.is_relative_to(ROOT / "exp1") and args.interval >= 5
    output = path.parent / "monitor"
    output.mkdir(exist_ok=True)
    lock = (output / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(__import__("os").getpid()) + "\n")
    lock.flush()
    samples = deque()
    last_progress = time.monotonic()
    previous_bytes, previous_event = None, None
    missing_supervisor, missing_child = 0, 0
    expected = read(ROOT / "exp1/configs/sources.lock.json")["model"]["files"]
    while True:
        status = read(path)
        tick = time.monotonic()
        supervisor = process(status["supervisor_pid"])
        child = process(status.get("child_pid"))
        missing_supervisor = 0 if supervisor["alive"] else missing_supervisor + 1
        missing_child = 0 if child["alive"] else missing_child + 1
        terminal = status["status"] in ("failed", "complete")
        issues = []
        if status["status"] == "failed":
            issues.append(f"Stage {status['stage']} failed; returncode={status.get('returncode')}; {status.get('error', '')}")
        if not terminal:
            age = datetime.now(ZoneInfo("Asia/Shanghai")) - datetime.fromisoformat(status["updated_at"])
            if age.total_seconds() > 90:
                issues.append(f"Supervisor heartbeat is {age.total_seconds():.0f} seconds old")
            if missing_supervisor >= 2:
                issues.append("Supervisor process is missing for two consecutive observations")
            if missing_child >= 2:
                issues.append("Stage child process is missing for two consecutive observations")
        weights = {}
        model = ROOT / "model/Qwen3.5-9B-Base"
        for name, item in expected.items():
            if name.endswith(".safetensors"):
                weights[name] = {"bytes": min(item["size"], max(size(model / name), size(model / (name + ".partial")))),
                                 "expected_bytes": item["size"], "complete_file_exists": (model / name).exists()}
        total = sum(v["bytes"] for v in weights.values())
        target = sum(v["expected_bytes"] for v in weights.values())
        if previous_bytes is None or total != previous_bytes or status["stage"] != "download":
            last_progress = tick
        previous_bytes = total
        samples.append((tick, total))
        while len(samples) > 1 and samples[0][0] < tick - 300:
            samples.popleft()
        seconds = tick - samples[0][0]
        rate = max(0, (total - samples[0][1]) / seconds) if seconds > 0 else 0
        eta = (target - total) / rate if rate > 0 and seconds >= 60 and status["stage"] == "download" else None
        if status["stage"] == "download" and not terminal and tick - last_progress > 900:
            issues.append("No new weight bytes observed for 15 minutes")
        log = ROOT / status["log"]
        with log.open("rb") as f:
            f.seek(max(0, size(log) - 65536))
            tail = f.read().decode(errors="replace")
        error_lines = [line.strip() for line in tail.splitlines()
                       if re.search(r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory|^AssertionError:|^RuntimeError:", line)]
        if error_lines:
            issues.append("Stage log contains an error: " + error_lines[-1][:400])
        steps = re.findall(r"((?:smoke|full) [^\r\n]*?step=\d+/\d+)", tail)
        metrics = gpu()
        if "error" in metrics:
            issues.append("GPU health poll failed: " + metrics["error"])
        observation = {"observed_at": now(), "launch_status": status["status"], "stage": status["stage"],
                       "supervisor": supervisor, "child": child, "gpu": metrics,
                       "weight_bytes": total, "expected_weight_bytes": target, "weight_percent": 100 * total / target,
                       "recent_download_bytes_per_second": rate, "estimated_download_seconds_remaining": eta,
                       "download_no_progress_seconds": tick - last_progress, "weight_files": weights,
                       "last_optimizer_update": steps[-1] if steps else None,
                       "full_ghost_passed": status["full_ghost_passed"], "full_configuration_frozen": status["full_configuration_frozen"],
                       "official_test_evaluated": status["official_test_evaluated"], "issues": issues}
        temporary = output / "latest.tmp.json"
        temporary.write_text(json.dumps(observation, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output / "latest.json")
        with (output / "observations.jsonl").open("a") as f:
            f.write(json.dumps(observation, ensure_ascii=False) + "\n")
        event = (status["stage"], status["status"], tuple(issues))
        if event != previous_event:
            message = {"observed_at": observation["observed_at"], "stage": status["stage"], "status": status["status"], "issues": issues}
            with (output / "events.jsonl").open("a") as f:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")
            print(json.dumps(message, ensure_ascii=False), flush=True)
            previous_event = event
        if terminal or missing_supervisor >= 2:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
