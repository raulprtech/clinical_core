"""Reject individual-level artifacts newly introduced relative to a Git base.

Historical tracked results are not certified or silently grandfathered as safe:
the check covers added/modified result files only. It prints paths, never rows.
"""
import argparse
import csv
import io
import json
import re
import subprocess

PATIENT = re.compile(r"\b(?:TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|C3[NL]-[0-9]+)\b", re.I)
KEYS = {"case_id", "patient_id", "subject_id", "sample_id", "patient_ids", "case_ids"}


def individual_payload(path, content):
    if PATIENT.search(content):
        return True
    if path.endswith(".csv"):
        header = next(csv.reader(io.StringIO(content)), [])
        return bool(KEYS.intersection(column.strip().lower() for column in header))
    if path.endswith(".json"):
        def inspect(value):
            if isinstance(value, dict):
                return bool(KEYS.intersection(str(k).lower() for k in value)) or any(
                    inspect(v) for v in value.values()
                )
            return isinstance(value, list) and any(inspect(v) for v in value)
        return inspect(json.loads(content))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    base = args.base
    if not base or set(base) == {"0"}:
        base = "origin/master"
    files = subprocess.check_output([
        "git", "diff", "--name-only", "--diff-filter=ACMR", "-z", base, "HEAD"
    ]).decode().split("\0")
    failures = []
    for path in files:
        if not path.startswith(("results_", "publication/")):
            continue
        content = subprocess.check_output(["git", "show", f"HEAD:{path}"]).decode()
        if individual_payload(path, content):
            failures.append(path)
    for path in failures:
        print(f"Individual-level result artifact: {path}")
    print(f"Result artifact check: {len(failures)} violations")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
