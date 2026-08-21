"""Compare a LUAD reproduction against frozen aggregate values and tolerances."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "publication/luad_baseline_os_v2/manifest.json"
DEFAULT_CANONICAL = REPO_ROOT / "publication/luad_baseline_os_v2/canonical_summary.json"


def strict_load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path}")
        ),
    )


def _by_protocol(rows: list[dict]) -> dict[str, dict]:
    return {row["protocol"]: row for row in rows}


def _check_numeric(
    checks: list[dict],
    name: str,
    observed,
    expected,
    tolerance: float,
) -> None:
    finite = isinstance(observed, (int, float)) and math.isfinite(observed)
    delta = abs(observed - expected) if finite else None
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "absolute_tolerance": tolerance,
            "absolute_delta": delta,
            "passed": bool(finite and delta <= tolerance),
        }
    )


def compare(run_summary: Path, canonical_path: Path, manifest_path: Path) -> dict:
    observed = strict_load(run_summary)
    canonical = strict_load(canonical_path)
    manifest = strict_load(manifest_path)
    tolerances = manifest["comparison_tolerances"]
    checks = []

    exact = {
        "parsed_cases": observed.get("n_cases"),
        "survival_cases": observed.get("n_cases_survival"),
        "survival_events": observed.get("n_events_survival"),
        "errors": observed.get("errors"),
    }
    expected_exact = {
        "parsed_cases": canonical["cohort"]["parsed_cases"],
        "survival_cases": canonical["cohort"]["survival_cases"],
        "survival_events": canonical["cohort"]["survival_events"],
        "errors": canonical["errors"],
    }
    for name, expected in expected_exact.items():
        checks.append(
            {
                "name": name,
                "observed": exact[name],
                "expected": expected,
                "passed": exact[name] == expected,
            }
        )

    holdout_observed = _by_protocol(observed["phases"]["phase_2_holdout"])
    for row in canonical["holdout"]:
        _check_numeric(
            checks,
            f"holdout.{row['protocol']}.cindex",
            holdout_observed[row["protocol"]]["mean"],
            row["cindex"],
            tolerances["cindex_absolute"],
        )

    cv_observed = _by_protocol(observed["phases"]["phase_2_repeated_cv"])
    for row in canonical["repeated_cross_validation"]:
        _check_numeric(
            checks,
            f"repeated_cv.{row['protocol']}.cindex_mean",
            cv_observed[row["protocol"]]["mean"],
            row["cindex_mean"],
            tolerances["cindex_absolute"],
        )
        checks.append(
            {
                "name": f"repeated_cv.{row['protocol']}.folds",
                "observed": cv_observed[row["protocol"]]["count"],
                "expected": row["folds"],
                "passed": cv_observed[row["protocol"]]["count"] == row["folds"],
            }
        )

    temporal_observed = _by_protocol(observed["phases"]["phase_2_temporal_validation"])
    for row in canonical["internal_temporal_transport"]:
        actual = temporal_observed[row["protocol"]]
        _check_numeric(
            checks,
            f"temporal.{row['protocol']}.cindex",
            actual["cindex"],
            row["cindex"],
            tolerances["cindex_absolute"],
        )
        _check_numeric(
            checks,
            f"temporal.{row['protocol']}.ipcw_cindex",
            actual["cindex_ipcw"],
            row["ipcw_cindex"],
            tolerances["cindex_absolute"],
        )
    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema": "clinical-core.reproduction-comparison/v1",
        "run_summary": str(run_summary),
        "checks": checks,
        "passed": not failed,
        "failed": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_summary", type=Path)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.run_summary, args.canonical, args.manifest)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
