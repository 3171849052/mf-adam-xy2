import argparse
import json

from .common import EXP, cache_paths, environment, load_config


def main():
    cache_paths()
    p = argparse.ArgumentParser(description="exp1: explicit smoke/pilot/full LoRA and fixed-schedule GDP")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="download pinned data, split and decontaminate")
    dl = sub.add_parser("download", help="download and hash the pinned Qwen checkpoint")
    dl.add_argument("--transport", choices=("huggingface", "modelscope"), default="huggingface")
    sub.add_parser("env", help="print the actual software/GPU environment")
    sub.add_parser("structure", help="inspect 9B keys and shapes on meta, without loading weights")
    for name in ("audit", "plan", "check-ghost", "run"):
        q = sub.add_parser(name)
        q.add_argument("--config", required=True)
        if name in ("plan", "check-ghost", "run"):
            q.add_argument("--output", required=True)
    for name in ("final", "report"):
        q = sub.add_parser(name)
        q.add_argument("--run", required=True)
    a = p.parse_args()
    if a.command == "prepare":
        from .data import prepare
        result = prepare()
    elif a.command == "download":
        from .modeling import download_model
        download_model(a.transport)
        result = {"model_manifest": str(EXP / "artifacts/model_manifest.json")}
    elif a.command == "env":
        result = environment()
    elif a.command == "structure":
        from .modeling import structure_audit
        result = structure_audit()
    elif a.command == "audit":
        from .data import length_audit
        from .modeling import tokenizer
        result = length_audit(tokenizer(), load_config(a.config))
        result = {**result, "splits": {k: {kk: vv for kk, vv in v.items() if kk != "overlength"} for k, v in result["splits"].items()}}
    else:
        from .pipeline import check_ghost, checked_output, final, plan, run, summarize
        if a.command == "plan":
            result = plan(load_config(a.config), a.output)
        elif a.command == "check-ghost":
            result = check_ghost(load_config(a.config), a.output)
        elif a.command == "run":
            result = run(a.config, a.output)
        elif a.command == "final":
            result = final(a.run)
        else:
            result = summarize(checked_output(a.run))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
