from __future__ import annotations

import argparse
import json
import os

from reglu.config import PUBLIC_COMMANDS, RunConfig, load_run_config, normalize_public_command


def _default_single_visible_gpu(config: RunConfig) -> None:
    """避免多卡可见时 HF Trainer 走 DataParallel（backward 里 reduce_add_coalesced），并减少误占多卡。"""
    if str(config.runtime.device).lower() == "cpu":
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reglu", description="RegLU open-source CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in PUBLIC_COMMANDS:
        aliases: list[str] = []
        if command == "forget":
            aliases = ["unlearn"]
        elif command == "eval":
            aliases = ["evaluate"]
        subparser = subparsers.add_parser(command, aliases=aliases)
        subparser.set_defaults(command_name=command)
        subparser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    public_command = getattr(args, "command_name", normalize_public_command(args.command))
    config = load_run_config(args.config, public_command)
    if public_command == "finetune":
        _default_single_visible_gpu(config)
        from reglu.trainers import run_finetune

        result = run_finetune(config)
    elif public_command == "forget":
        _default_single_visible_gpu(config)
        from reglu.trainers import run_unlearn

        result = run_unlearn(config)
    else:
        from reglu.eval import run_eval

        result = run_eval(config)
    if isinstance(result, dict) and "command" in result:
        result["command"] = public_command
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
