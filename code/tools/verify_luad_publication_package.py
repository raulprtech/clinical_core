"""Verify the aggregate-only TCGA-LUAD publication package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "publication/luad_baseline_os_v2/manifest.json"
FORBIDDEN_KEYS = {
    "case_id",
    "patient_id",
    "submitter_id",
    "prediction",
    "risk_score",
    "raw_features",
    "raw_targets",
}
PATIENT_PATTERN = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")


def _strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path}")
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_aggregate_only(value, location: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"forbidden keys at {location}: {sorted(forbidden)}")
        for key, item in value.items():
            _audit_aggregate_only(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _audit_aggregate_only(item, f"{location}[{index}]")
    elif isinstance(value, str) and PATIENT_PATTERN.search(value):
        raise ValueError(f"patient identifier at {location}")


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    require_data: bool = False,
    require_run: bool = False,
) -> dict:
    manifest = _strict_load(manifest_path)
    if manifest.get("schema") != "clinical-core.publication-manifest/v1":
        raise ValueError("unsupported publication manifest schema")
    _audit_aggregate_only(manifest)

    checked = []
    for record in manifest["tracked_files"]:
        path = REPO_ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"hash mismatch for {record['path']}")
        if path.suffix == ".json":
            _audit_aggregate_only(_strict_load(path))
        checked.append(record["path"])

    config_text = (
        REPO_ROOT / "code/experiments/experiment_config_nigma_luad_baseline_os_v2.yaml"
    ).read_text(encoding="utf-8")
    if "/home/" in config_text or "/content/drive" in config_text:
        raise ValueError("active LUAD config contains a machine-specific path")

    source = manifest["source"]
    source_path = REPO_ROOT / source["local_manifest"]
    data_status = "not_required"
    if source_path.exists():
        if _sha256(source_path) != source["local_manifest_sha256"]:
            raise ValueError("source manifest hash mismatch")
        source_payload = _strict_load(source_path)
        if source_payload.get("patient_xml_count") != source["patient_xml_count"]:
            raise ValueError("source manifest XML count mismatch")
        data_status = "verified"
    else:
        restore_path = source_path.with_name("RESTORE.json")
        if restore_path.exists():
            restore = _strict_load(restore_path)
            expected_count = source["patient_xml_count"]
            expected_inventory = source["tracked_inventory_sha256"]
            if restore.get("schema") != "clinical-core.gdc-restore-receipt/v1":
                raise ValueError("unsupported GDC restore receipt schema")
            if restore.get("project_id") != source["project_id"]:
                raise ValueError("restore receipt project mismatch")
            if restore.get("inventory_sha256") != expected_inventory:
                raise ValueError("restore receipt inventory mismatch")
            if restore.get("requested_files") != expected_count:
                raise ValueError("restore receipt file count mismatch")
            if restore.get("complete") is not True:
                raise ValueError("restore receipt is incomplete")
            xml_dir = source_path.parent / "clinical_supplement"
            if len(list(xml_dir.glob("*.xml"))) != expected_count:
                raise ValueError("restored XML count mismatch")
            data_status = "verified_restoration"
        elif require_data:
            raise FileNotFoundError(source_path)

    run_status = "not_required"
    run = manifest["canonical_local_run"]
    run_path = REPO_ROOT / run["path"]
    if run_path.exists():
        if _sha256(run_path / "summary.json") != run["strict_summary_sha256"]:
            raise ValueError("canonical run summary hash mismatch")
        _strict_load(run_path / "summary.json")
        for filename, expected in run["aggregate_artifacts"].items():
            if _sha256(run_path / filename) != expected:
                raise ValueError(f"canonical run hash mismatch: {filename}")
        run_status = "verified"
    elif require_run:
        raise FileNotFoundError(run_path)

    return {"tracked_files": checked, "data": data_status, "run": run_status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-run", action="store_true")
    args = parser.parse_args()
    result = verify_manifest(args.manifest, require_data=args.require_data, require_run=args.require_run)
    print(json.dumps({"status": "passed", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
