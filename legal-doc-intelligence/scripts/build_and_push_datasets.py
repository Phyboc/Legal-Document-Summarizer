#!/usr/bin/env python3
"""Build the ten section-aligned IN-ABS datasets and upload each to the Hub.

This is a thin *orchestrator* over ``scripts/build_dataset.py``. It does not
reimplement any build or upload logic: for every configuration it constructs the
same argv the standalone CLI would receive, then calls the module's own
``parse_args`` -> ``run_build`` -> ``report`` -> ``push_to_hub``. That keeps one
source of truth for the schema, the quality gates and the upload guards:

* a stub-encoder build is refused before any upload;
* a directory without ``build_stats.json`` is refused on push-only;
* the dataset is read back from the Hub after upload to confirm it.

The ten configurations differ along one axis each (size, split, alignment knob,
evidence layer, PII redaction) so the resulting datasets are comparable. Two are
deliberately cheap (``sample10``, ``sample100``) for hand inspection; the rest
default to ``--full``. Use ``--fast`` to replace every ``--full`` with a small
sample while you are still smoke-testing the pipeline.

Usage
-----
    # dry run: show exactly what would be built and pushed, touch nothing
    python scripts/build_and_push_datasets.py --namespace <HF_USER> --dry-run

    # quick pass over all ten (whole IN-ABS train split sampled, not everything)
    python scripts/build_and_push_datasets.py --namespace <HF_USER> --fast --fast-n 50

    # the real run: full builds, pushed private (recommended until the source
    # dataset's licence is resolved - percins/IN-ABS declares none)
    python scripts/build_and_push_datasets.py --namespace <HF_USER> --private

    # build only, no upload
    python scripts/build_and_push_datasets.py --namespace <HF_USER> --skip-push

    # one dataset only
    python scripts/build_and_push_datasets.py --namespace <HF_USER> --only full

Authentication is read from the environment / the Hugging Face login cache
exactly as ``build_dataset.py`` does (``HF_TOKEN`` or ``hf auth login``). Nothing
is written into the repository.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Reuse the builder as a library. Add this file's own directory (scripts/) so
# ``import build_dataset`` resolves regardless of the CWD.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_dataset as bd  # noqa: E402

logger = logging.getLogger("build_and_push")


# --------------------------------------------------------------------------- #
# The ten configurations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetSpec:
    """One dataset: a name, the builder flags that define it, and why it exists."""

    name: str
    flags: tuple[str, ...]
    purpose: str


# Each spec varies exactly one thing from the baseline ``full`` so the datasets
# are directly comparable. ``--full`` specs are swapped for a sample under
# ``--fast`` (see ``resolve_flags``).
DEFAULT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("sample10",    ("--sample", "10"),            "smoke test; small enough to inspect by hand"),
    DatasetSpec("sample100",   ("--sample", "100"),           "timing + statistics check"),
    DatasetSpec("full",        ("--full",),                   "the canonical training set"),
    DatasetSpec("val100",      ("--sample", "100", "--split", "validation"), "100 validation documents only"),
    DatasetSpec("test-only",   ("--full", "--split", "test"),  "test split only"),
    DatasetSpec("evidence",    ("--full", "--evidence-mode"), "evidence-aware alignment (UNCALIBRATED)"),
    DatasetSpec("singlelabel", ("--full", "--no-multi-label"), "one label per summary sentence"),
    DatasetSpec("thr030",      ("--full", "--threshold", "0.30"), "lower alignment threshold"),
    DatasetSpec("thr055",      ("--full", "--threshold", "0.55"), "higher alignment threshold"),
    DatasetSpec("redacted",    ("--full", "--redact-names"),   "PERSON entities redacted (spaCy)"),
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def resolve_flags(spec: DatasetSpec, fast: bool, fast_n: int) -> list[str]:
    """Return the builder flags for one spec, applying ``--fast`` if requested."""
    flags = list(spec.flags)
    if fast and "--full" in flags:
        # Replace the single --full with a sample of fast_n documents.
        flags[flags.index("--full")] = "--sample"
        flags.insert(flags.index("--sample") + 1, str(fast_n))
    return flags


def select_specs(specs: tuple[DatasetSpec, ...], only: list[str] | None) -> list[DatasetSpec]:
    """Filter the spec list by ``--only`` names (case-insensitive), preserving order."""
    if not only:
        return list(specs)
    wanted = {name.strip().lower() for name in only}
    known = {spec.name.lower() for spec in specs}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(
            f"Unknown --only name(s): {unknown}. Known datasets: {[s.name for s in specs]}"
        )
    return [spec for spec in specs if spec.name.lower() in wanted]


def repo_id_for(namespace: str, prefix: str, name: str) -> str:
    return f"{namespace}/{prefix}-{name}"


# --------------------------------------------------------------------------- #
# Per-dataset run
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    name: str
    output_dir: str
    repo_id: str | None
    built: bool = False
    pushed: bool = False
    records: int | None = None
    error: str | None = None


def run_one(
    spec: DatasetSpec,
    *,
    namespace: str,
    prefix: str,
    output_root: Path,
    fast: bool,
    fast_n: int,
    private: bool,
    overwrite: bool,
    push: bool,
    dry_run: bool,
    append: bool = False,
) -> RunResult:
    """Build one dataset, and upload it unless ``push`` is False.

    A ``--fast`` run of a ``--full`` spec is redirected to its own directory and
    Hub repo (``<name>-fast<N>``). Otherwise ``--fast`` would write ~60 documents
    into ``artifacts/training_data/full`` - and, worse, push them over the
    canonical full dataset on the Hub, which is exactly what the committed
    ``full/`` build (mode ``sample20``) shows happened once already.
    """
    is_full_spec = "--full" in spec.flags
    name = f"{spec.name}-fast{fast_n}" if (fast and is_full_spec) else spec.name
    output_dir = output_root / name
    repo_id = repo_id_for(namespace, prefix, name) if namespace else None
    result = RunResult(name=spec.name, output_dir=str(output_dir), repo_id=repo_id)

    argv = resolve_flags(spec, fast, fast_n) + ["--output_dir", str(output_dir)]
    if overwrite:
        argv.append("--overwrite")
    if push:
        if not repo_id:
            raise SystemExit("--namespace is required to push (or pass --skip-push).")
        argv += ["--push-to-hub", "--repo-id", repo_id]
        if private:
            argv.append("--private")
        if append:
            argv.append("--append")

    if dry_run:
        logger.info("[dry-run] %s", " ".join(["python", "scripts/build_dataset.py", *argv]))
        return result

    try:
        args = bd.parse_args(argv)
        _records_by_split, stats, _builds = bd.run_build(args)
        result.built = True
        result.records = sum(len(recs) for recs in _records_by_split.values())
        bd.report(stats, Path(args.output_dir))

        if push:
            bd.push_to_hub(args, stats)
            result.pushed = True
    except SystemExit as exc:  # parse_args/quality gates raise SystemExit
        result.error = f"SystemExit: {exc}"
        logger.error("Dataset '%s' aborted: %s", spec.name, result.error)
    except Exception as exc:  # defensive: one bad dataset must not stop the batch
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error("Dataset '%s' failed: %s", spec.name, result.error)

    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ten section-aligned IN-ABS datasets and push each to the Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  --namespace <HF_USER> --dry-run\n"
            "  --namespace <HF_USER> --fast --fast-n 50\n"
            "  --namespace <HF_USER> --private\n"
            "  --namespace <HF_USER> --skip-push\n"
            "  --namespace <HF_USER> --only full --only evidence\n"
        ),
    )
    parser.add_argument("--namespace", default=None,
                        help="Hugging Face user/org that owns the datasets. Required to push.")
    parser.add_argument("--repo-prefix", default="legalLens-in-abs",
                        help="Repo name prefix; each repo becomes <prefix>-<dataset>.")
    parser.add_argument("--output-root", default="artifacts/training_data",
                        help="Directory that will hold one sub-directory per dataset.")
    parser.add_argument("--only", action="append", default=None,
                        help="Build only this dataset (repeatable). Defaults to all ten.")
    parser.add_argument("--fast", action="store_true",
                        help="Replace every --full with --sample <fast-n> (pipeline smoke test).")
    parser.add_argument("--fast-n", type=int, default=50,
                        help="Documents per split used by --fast (default 50).")
    parser.add_argument("--private", action="store_true",
                        help="Create Hub repos as private (recommended; default is off).")
    parser.add_argument("--append", action="store_true",
                        help=(
                            "Merge each build INTO the existing Hub dataset instead of "
                            "replacing it, so repeated runs accumulate into one repo per "
                            "system instead of creating a new one. Rows already on the "
                            "Hub are updated to the fresh values (upsert by doc_id + "
                            "chunk_id); remote-only rows are kept."
                        ))
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into output dirs that already hold a build.")
    parser.add_argument("--skip-push", action="store_true",
                        help="Build only; never upload.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the builder commands without running them.")
    parser.add_argument("--summary", default=None,
                        help="Path for the JSON run summary (default: <output-root>/build_and_push_summary.json).")

    args = parser.parse_args(argv)
    if not args.namespace and not args.skip_push and not args.dry_run:
        parser.error("--namespace is required unless you pass --skip-push or --dry-run.")
    if args.fast_n < 1:
        parser.error("--fast-n must be a positive number of documents.")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    specs = select_specs(DEFAULT_DATASETS, args.only)
    output_root = Path(args.output_root)
    push = not args.skip_push

    # Fail fast on auth, before spending any build time, unless we are not pushing.
    if push and not args.dry_run:
        bd.require_hf_auth(bd.resolve_hf_token())

    logger.info(
        "Building %d dataset(s)%s -> %s",
        len(specs), "" if args.dry_run else f" (fast={args.fast}, push={push})", output_root,
    )
    if args.dry_run:
        for spec in specs:
            run_one(spec, namespace=args.namespace, prefix=args.repo_prefix,
                    output_root=output_root, fast=args.fast, fast_n=args.fast_n,
                    private=args.private, overwrite=args.overwrite, push=push, dry_run=True,
                    append=args.append)
        return 0

    results: list[RunResult] = []
    for spec in specs:
        logger.info("=" * 72)
        logger.info("Dataset: %s  (%s)", spec.name, spec.purpose)
        logger.info("=" * 72)
        results.append(run_one(
            spec, namespace=args.namespace, prefix=args.repo_prefix, output_root=output_root,
            fast=args.fast, fast_n=args.fast_n, private=args.private,
            overwrite=args.overwrite, push=push, dry_run=False, append=args.append,
        ))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "namespace": args.namespace,
        "repo_prefix": args.repo_prefix,
        "fast": args.fast,
        "pushed": push,
        "datasets": [r.__dict__ for r in results],
        "built": sum(1 for r in results if r.built),
        "uploaded": sum(1 for r in results if r.pushed),
        "failed": sum(1 for r in results if r.error),
    }
    summary_path = Path(args.summary) if args.summary else output_root / "build_and_push_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["", "=" * 72, "BATCH SUMMARY", "=" * 72]
    for r in results:
        status = "uploaded" if r.pushed else ("built" if r.built else "FAILED")
        target = r.repo_id or "-"
        detail = r.error or (f"{r.records} records" if r.records is not None else "")
        lines.append(f"  {r.name:12s} {status:9s} {target:40s} {detail}")
    lines += [
        "=" * 72,
        f"  built {summary['built']} / uploaded {summary['uploaded']} / failed {summary['failed']}",
        f"  summary: {summary_path}",
        "=" * 72,
    ]
    print("\n".join(lines))

    # Non-zero exit if anything failed, so a CI/Colab cell notices.
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
