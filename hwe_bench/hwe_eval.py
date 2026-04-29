from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class DesignSpec:
    name: str
    org: str
    repo: str
    dataset: Path
    pull_key: str


DESIGNS: dict[str, DesignSpec] = {
    "ibex": DesignSpec(
        name="ibex",
        org="lowRISC",
        repo="ibex",
        dataset=Path("datasets/lowRISC__ibex.jsonl"),
        pull_key="ibex",
    ),
    "cva6": DesignSpec(
        name="cva6",
        org="openhwgroup",
        repo="cva6",
        dataset=Path("datasets/openhwgroup__cva6.jsonl"),
        pull_key="cva6",
    ),
    "caliptra": DesignSpec(
        name="caliptra",
        org="chipsalliance",
        repo="caliptra-rtl",
        dataset=Path("datasets/chipsalliance__caliptra-rtl.jsonl"),
        pull_key="caliptra",
    ),
    "caliptra-rtl": DesignSpec(
        name="caliptra",
        org="chipsalliance",
        repo="caliptra-rtl",
        dataset=Path("datasets/chipsalliance__caliptra-rtl.jsonl"),
        pull_key="caliptra",
    ),
    "rocketchip": DesignSpec(
        name="rocketchip",
        org="chipsalliance",
        repo="rocket-chip",
        dataset=Path("datasets/chipsalliance__rocket-chip.jsonl"),
        pull_key="rocketchip",
    ),
    "rocket-chip": DesignSpec(
        name="rocketchip",
        org="chipsalliance",
        repo="rocket-chip",
        dataset=Path("datasets/chipsalliance__rocket-chip.jsonl"),
        pull_key="rocketchip",
    ),
    "xiangshan": DesignSpec(
        name="xiangshan",
        org="OpenXiangShan",
        repo="XiangShan",
        dataset=Path("datasets/OpenXiangShan__XiangShan.jsonl"),
        pull_key="xiangshan",
    ),
    "opentitan": DesignSpec(
        name="opentitan",
        org="lowRISC",
        repo="opentitan",
        dataset=Path("datasets/lowRISC__opentitan.jsonl"),
        pull_key="opentitan",
    ),
}

RESULTS_ROOT = Path("results")
DEFAULT_ATTEMPTS = 1
DEFAULT_RETRIES = 2
DEFAULT_AGENT_SETUP_TIMEOUT_MULTIPLIER = 3.0
REMOVED_GEN_PATCH_ARGS = {
    "--agent-setup-timeout-multiplier",
    "--attempts",
    "--dataset",
    "--design",
    "--dry-run",
    "--force",
    "--prepare-only",
    "--results-root",
    "--retries",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_design(name: str) -> DesignSpec:
    key = name.strip()
    try:
        return DESIGNS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DESIGNS))
        raise SystemExit(f"Unknown design {name!r}. Valid designs: {valid}") from exc


def _dataset_path(spec: DesignSpec, override: str | None) -> Path:
    path = Path(override) if override else spec.dataset
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _flatten_cases(raw_cases: Sequence[str] | None) -> list[str]:
    if not raw_cases:
        return []
    cases: list[str] = []
    for raw in raw_cases:
        for part in raw.split(","):
            part = part.strip()
            if part:
                cases.append(part)
    return cases


def _case_to_number(case: str) -> int:
    match = re.search(r"(?:^|pr-)(\d+)$", case.strip())
    if not match:
        match = re.search(r"(\d+)$", case.strip())
    if not match:
        raise SystemExit(
            f"Could not parse case {case!r}. Use a PR number, pr-1735, or ibex-pr-1735."
        )
    return int(match.group(1))


def _selected_numbers(cases: Sequence[str] | None, dataset: Path) -> list[int]:
    case_values = _flatten_cases(cases)
    if case_values:
        return sorted({_case_to_number(case) for case in case_values})
    return sorted({int(row["number"]) for row in _read_jsonl(dataset)})


def _task_name(repo: str, number: int) -> str:
    return f"{repo}-pr-{number}"


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._()+-]+", "_", value.strip())
    return safe.strip("_") or "default"


def _quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _run(cmd: Sequence[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(_quote_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), check=True, cwd=_repo_root(), env=env)


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_list(values: Sequence[str | int]) -> str:
    return "[" + ", ".join(_toml_str(str(value)) for value in values) + "]"


def _write_info(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[info]"]
    for key in [
        "design",
        "cases",
        "model",
        "effective_model",
        "agent",
        "base_url",
        "tag",
        "dataset",
    ]:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            lines.append(f"{key} = {_toml_list(value)}")
        else:
            lines.append(f"{key} = {_toml_str(str(value))}")
    lines.extend(["", "[paths]"])
    for key in ["run_dir", "tasks_dir", "job_dir", "patches_dir", "eval_dir"]:
        value = data.get(key)
        if value is not None:
            lines.append(f"{key} = {_toml_str(str(value))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _normalize_model(agent: str, model: str, base_url: str | None) -> str:
    if "/" in model:
        return model
    if agent == "openhands-sdk":
        lower = f"{model} {base_url or ''}".lower()
        if "dashscope" in lower or model.lower().startswith("qwen"):
            return f"dashscope/{model}"
        if "deepseek" in lower:
            return f"deepseek/{model}"
        if "z.ai" in lower or "bigmodel" in lower or model.lower().startswith("glm"):
            return f"zai/{model}"
    return model


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _gen_patch_run_dir(spec: DesignSpec, agent: str, model: str, tag: str | None) -> tuple[Path, str]:
    base = _repo_root() / RESULTS_ROOT / spec.name / agent / _safe_component(model)
    if tag:
        run_dir = base / tag
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Run directory already exists; choose a different --tag: {run_dir}")
        return run_dir, tag

    tag = _timestamp_tag()
    run_dir = base / tag
    suffix = 2
    while run_dir.exists():
        candidate_tag = f"{tag}_{suffix}"
        run_dir = base / candidate_tag
        suffix += 1
    return run_dir, run_dir.name


def _removed_gen_patch_arg(unknown: Sequence[str]) -> str | None:
    for arg in unknown:
        option = arg.split("=", 1)[0]
        if option in REMOVED_GEN_PATCH_ARGS:
            return option
    return None


def cmd_init(args: argparse.Namespace) -> int:
    spec = _resolve_design(args.design_name)
    dataset = _dataset_path(spec, None)
    script = _repo_root() / "scripts" / "pull_images.sh"
    cmd = [str(script), spec.pull_key, "--dataset", str(dataset)]
    _run(cmd)
    return 0


def cmd_gen_patch(args: argparse.Namespace, harbor_args: Sequence[str]) -> int:
    spec = _resolve_design(args.design_name)

    dataset = _dataset_path(spec, None)
    numbers = _selected_numbers(args.case, dataset)
    run_dir, tag = _gen_patch_run_dir(spec, args.agent, args.model, args.tag)
    tasks_dir = run_dir / "tasks"
    jobs_parent = run_dir / "jobs"
    job_name = "run"
    job_dir = jobs_parent / job_name
    patches_dir = run_dir / "patches"
    eval_dir = run_dir / "eval"

    run_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "workdir").mkdir(parents=True, exist_ok=True)
    (eval_dir / "logs").mkdir(parents=True, exist_ok=True)

    only = ",".join(str(number) for number in numbers)
    adapter_cmd = [
        sys.executable,
        "-m",
        "hwe_bench.harness.harbor.adapter",
        "--input",
        str(dataset),
        "--output",
        str(tasks_dir),
    ]
    if only:
        adapter_cmd.extend(["--only", only])

    effective_model = _normalize_model(args.agent, args.model, args.base_url)
    harbor_bin = shutil.which("harbor") or "harbor"
    harbor_cmd = [
        harbor_bin,
        "run",
        "--path",
        str(tasks_dir),
        "-a",
        args.agent,
        "-m",
        effective_model,
        "-k",
        str(DEFAULT_ATTEMPTS),
        "-r",
        str(DEFAULT_RETRIES),
        "--n-concurrent",
        str(args.jobs),
        "--no-delete",
        "--agent-setup-timeout-multiplier",
        str(DEFAULT_AGENT_SETUP_TIMEOUT_MULTIPLIER),
        "--jobs-dir",
        str(jobs_parent),
        "--job-name",
        job_name,
    ]
    for number in numbers:
        harbor_cmd.extend(["--include-task-name", _task_name(spec.repo, number)])

    api_key = args.api_key or os.environ.get("HWE_EVAL_API_KEY")
    if args.base_url:
        harbor_cmd.extend(["--ae", f"LLM_BASE_URL={args.base_url}"])
    if api_key:
        harbor_cmd.extend(["--ae", f"LLM_API_KEY={api_key}"])
    harbor_cmd.extend(harbor_args)

    _write_info(
        run_dir / "info.toml",
        {
            "design": spec.name,
            "cases": [_task_name(spec.repo, number) for number in numbers],
            "model": args.model,
            "effective_model": effective_model,
            "agent": args.agent,
            "base_url": args.base_url,
            "tag": tag,
            "dataset": str(dataset),
            "run_dir": str(run_dir),
            "tasks_dir": str(tasks_dir),
            "job_dir": str(job_dir),
            "patches_dir": str(patches_dir),
            "eval_dir": str(eval_dir),
        },
    )

    _run(adapter_cmd)

    safe_harbor_cmd = [
        "****" if str(part).startswith("LLM_API_KEY=") else part for part in harbor_cmd
    ]
    print(_quote_cmd(safe_harbor_cmd))
    env = os.environ.copy()
    subprocess.run(harbor_cmd, check=True, cwd=_repo_root(), env=env)

    print(f"Run directory: {run_dir}")
    return 0


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    spec = _resolve_design(args.design_name)
    return _repo_root() / RESULTS_ROOT / spec.name / args.agent / _safe_component(args.model) / args.tag


def cmd_eval(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args)
    info_path = run_dir / "info.toml"
    if not info_path.exists():
        raise SystemExit(f"info.toml not found: {info_path}")

    info = _read_info(info_path)
    info_data = info.get("info", {})
    paths = info.get("paths", {})
    dataset = Path(str(info_data["dataset"]))
    job_dir = Path(str(paths["job_dir"]))
    patches_dir = Path(str(paths.get("patches_dir", run_dir / "patches")))
    eval_dir = Path(str(paths.get("eval_dir", run_dir / "eval")))
    workdir = eval_dir / "workdir"
    logs_dir = eval_dir / "logs"

    verify_cmd = [
        sys.executable,
        "-m",
        "hwe_bench.harness.harbor.verify_bridge",
        "--harbor-job-dir",
        str(job_dir),
        "--output",
        str(patches_dir),
    ]
    evaluator_cmd = [
        sys.executable,
        "-m",
        "hwe_bench.harness.evaluator",
        "--workdir",
        str(workdir.resolve()),
        "--patch_files",
        str(patches_dir / "patches.jsonl"),
        "--dataset_files",
        str(dataset),
        "--output_dir",
        str(eval_dir),
        "--log_dir",
        str(logs_dir),
        "--stop_on_error",
        "false",
        "--max_workers",
        str(args.jobs),
    ]

    _run(verify_cmd)
    _run(evaluator_cmd)
    print(f"Eval directory: {eval_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hweEval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Pull all published images for a design.")
    init.add_argument("design_name")

    gen = subparsers.add_parser(
        "gen-patch",
        help="Generate Harbor tasks and run an agent to produce patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Harbor passthrough:\n"
            "  Any arguments not recognized by hweEval are appended verbatim to `harbor run`.\n"
            "  Example: hweEval gen-patch ibex --agent openhands --model gpt-5.5 \\\n"
            "           --ak reasoning_effort=high --ak max_iterations=500\n"
        ),
    )
    gen.add_argument("design_name")
    gen.add_argument("--model", required=True)
    gen.add_argument("--base-url", default="")
    gen.add_argument("--api-key", default="")
    gen.add_argument("--agent", default="openhands-sdk")
    gen.add_argument("--case", action="append", default=[])
    gen.add_argument("--jobs", type=int, default=1)
    gen.add_argument("--tag", default="")

    ev = subparsers.add_parser("eval", help="Extract patches and run offline scoring.")
    ev.add_argument("design_name")
    ev.add_argument("--agent", required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--tag", required=True)
    ev.add_argument("--jobs", type=int, default=4)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command == "init":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        return cmd_init(args)
    if args.command == "gen-patch":
        removed_arg = _removed_gen_patch_arg(unknown)
        if removed_arg:
            parser.error(
                f"{removed_arg} has been removed from hweEval gen-patch; "
                "the run directory is created automatically under results/<design>/<agent>/<model>/<tag>"
            )
        return cmd_gen_patch(args, unknown)
    if args.command == "eval":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        return cmd_eval(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
