"""Audit whether MMIST CPTAC-CCRCC supports external OS validation.

The MMIST release exposes a binary 12-month endpoint. This tool links the
CPTAC-only MMIST case identifiers to public GDC CPTAC-3 clinical metadata and
derives a time-to-event overall-survival endpoint without downloading images.

Patient-level inputs and outputs belong under ``data/`` (ignored by Git). Only
aggregate summaries should be copied to versioned result directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


MMIST_BASE = (
    "https://raw.githubusercontent.com/Multi-Modal-IST/"
    "multi-modal-ist.github.io/refs/heads/master/datasets/ccRCC/Code"
)
DEFAULT_URLS = {
    "clinical": f"{MMIST_BASE}/clinical%2Bgenomic_split.csv",
    "ct": f"{MMIST_BASE}/patients_with_labels_CT_final.csv",
    "mr": f"{MMIST_BASE}/patients_with_labels_MR_final.csv",
}
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
GDC_FIELDS = (
    "submitter_id,demographic.vital_status,demographic.days_to_death,"
    "diagnoses.days_to_last_follow_up,"
    "diagnoses.days_to_last_known_disease_status,follow_ups.days_to_follow_up"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def is_cptac_id(case_id: str) -> bool:
    return str(case_id).startswith(("C3L-", "C3N-"))


def fetch_gdc_cases(case_ids: Iterable[str], path: Path) -> None:
    case_ids = sorted(set(map(str, case_ids)))
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": ["CPTAC-3"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "submitter_id",
                    "value": case_ids,
                },
            },
        ],
    }
    params = {
        "filters": json.dumps(filters, separators=(",", ":")),
        "fields": GDC_FIELDS,
        "expand": "demographic,diagnoses,follow_ups",
        "format": "JSON",
        "size": str(max(len(case_ids), 1)),
    }
    request_url = f"{GDC_CASES_URL}?{urlencode(params)}"
    with urlopen(request_url, timeout=180) as response:
        payload = json.loads(response.read())
    hits = payload.get("data", {}).get("hits", [])
    returned = {hit.get("submitter_id") for hit in hits}
    missing = sorted(set(case_ids) - returned)
    if missing:
        raise RuntimeError(f"GDC did not return {len(missing)} MMIST cases")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def nonnegative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def derive_os(hit: dict[str, Any]) -> dict[str, Any]:
    demographic = hit.get("demographic") or {}
    vital_status = str(demographic.get("vital_status") or "").strip().lower()
    days_to_death = nonnegative_number(demographic.get("days_to_death"))

    followup_times: list[float] = []
    for diagnosis in hit.get("diagnoses") or []:
        for field in (
            "days_to_last_follow_up",
            "days_to_last_known_disease_status",
        ):
            value = nonnegative_number(diagnosis.get(field))
            if value is not None:
                followup_times.append(value)
    for followup in hit.get("follow_ups") or []:
        value = nonnegative_number(followup.get("days_to_follow_up"))
        if value is not None:
            followup_times.append(value)

    max_followup = max(followup_times) if followup_times else None
    if vital_status == "dead" and days_to_death is not None:
        return {
            "survival_days": days_to_death,
            "event": 1,
            "vital_status_gdc": vital_status,
            "max_followup_days": max_followup,
            "endpoint_status": "observed_death",
        }
    if vital_status == "alive" and max_followup is not None:
        return {
            "survival_days": max_followup,
            "event": 0,
            "vital_status_gdc": vital_status,
            "max_followup_days": max_followup,
            "endpoint_status": "censored_alive",
        }
    return {
        "survival_days": None,
        "event": None,
        "vital_status_gdc": vital_status,
        "max_followup_days": max_followup,
        "endpoint_status": "incomplete",
    }


def infer_12_month_status(survival_days: Any, event: Any) -> int | None:
    time = nonnegative_number(survival_days)
    if time is None or event not in (0, 1):
        return None
    if event == 1 and time <= 365:
        return 0
    if time >= 365:
        return 1
    return None


def summarize_cohort(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame.dropna(subset=["survival_days", "event"])
    events = int(valid["event"].sum())
    return {
        "mapped_cases": int(len(frame)),
        "gdc_matches": int(frame["gdc_match"].sum()),
        "valid_os": int(len(valid)),
        "events": events,
        "event_fraction": round(events / len(valid), 6) if len(valid) else None,
        "median_followup_or_death_days": (
            float(valid["survival_days"].median()) if len(valid) else None
        ),
    }


def build_audit(
    clinical: pd.DataFrame,
    ct: pd.DataFrame,
    mr: pd.DataFrame,
    gdc_payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"case_id", "vital_status_12"}
    if not required.issubset(clinical.columns):
        raise ValueError(f"MMIST clinical CSV must contain {sorted(required)}")
    if "case_id" not in ct.columns or "case_id" not in mr.columns:
        raise ValueError("MMIST CT/MR mappings must contain case_id")

    cptac = clinical[clinical["case_id"].map(is_cptac_id)].copy()
    cptac = cptac.drop_duplicates("case_id").set_index("case_id", drop=False)
    ct_ids = {str(value) for value in ct["case_id"] if is_cptac_id(value)}
    mr_ids = {str(value) for value in mr["case_id"] if is_cptac_id(value)}
    by_id = {
        hit["submitter_id"]: hit
        for hit in gdc_payload.get("data", {}).get("hits", [])
        if hit.get("submitter_id")
    }

    records = []
    for case_id, row in cptac.iterrows():
        hit = by_id.get(case_id)
        endpoint = derive_os(hit) if hit else {
            "survival_days": None,
            "event": None,
            "vital_status_gdc": None,
            "max_followup_days": None,
            "endpoint_status": "not_found",
        }
        inferred = infer_12_month_status(
            endpoint["survival_days"], endpoint["event"]
        )
        try:
            mmist_12 = int(row["vital_status_12"])
        except (TypeError, ValueError):
            mmist_12 = None
        records.append(
            {
                "case_id": case_id,
                "has_ct_mmist": case_id in ct_ids,
                "has_mr_mmist": case_id in mr_ids,
                "gdc_match": hit is not None,
                "mmist_vital_status_12": mmist_12,
                "gdc_inferred_status_12": inferred,
                "status_12_agrees": (
                    inferred == mmist_12
                    if inferred is not None and mmist_12 is not None
                    else None
                ),
                **endpoint,
            }
        )
    audit = pd.DataFrame(records).sort_values("case_id").reset_index(drop=True)

    masks = {
        "all_cptac": pd.Series(True, index=audit.index),
        "ct": audit["has_ct_mmist"],
        "mr": audit["has_mr_mmist"],
        "ct_or_mr": audit["has_ct_mmist"] | audit["has_mr_mmist"],
        "ct_and_mr": audit["has_ct_mmist"] & audit["has_mr_mmist"],
    }
    comparable = audit["status_12_agrees"].notna()
    agreements = int(audit.loc[comparable, "status_12_agrees"].sum())
    summary = {
        "cohorts": {
            name: summarize_cohort(audit[mask]) for name, mask in masks.items()
        },
        "endpoint_crosscheck": {
            "comparable_at_12_months": int(comparable.sum()),
            "agreements": agreements,
            "disagreements": int(comparable.sum()) - agreements,
        },
    }
    return audit, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("data/external/mmist_cptac_audit")
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    inputs = {name: args.work_dir / f"mmist_{name}.csv" for name in DEFAULT_URLS}
    for name, url in DEFAULT_URLS.items():
        download(url, inputs[name], force=args.force_download)

    clinical = pd.read_csv(inputs["clinical"])
    cptac_ids = sorted(
        set(clinical.loc[clinical["case_id"].map(is_cptac_id), "case_id"])
    )
    gdc_path = args.work_dir / "gdc_cptac3_cases.json"
    if args.force_download or not gdc_path.exists():
        fetch_gdc_cases(cptac_ids, gdc_path)

    gdc_payload = json.loads(gdc_path.read_text())
    audit, aggregate = build_audit(
        clinical,
        pd.read_csv(inputs["ct"]),
        pd.read_csv(inputs["mr"]),
        gdc_payload,
    )
    audit_path = args.work_dir / "patient_level_audit.csv"
    audit.to_csv(audit_path, index=False)

    aggregate.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "definition": {
                "event": "GDC vital_status=Dead with nonnegative days_to_death",
                "censoring": (
                    "GDC vital_status=Alive at maximum nonnegative diagnosis "
                    "or follow-up day"
                ),
                "time_origin": "GDC day fields are relative to initial diagnosis",
            },
            "sources": {**DEFAULT_URLS, "gdc_cases_api": GDC_CASES_URL},
            "input_sha256": {
                **{name: sha256(path) for name, path in inputs.items()},
                "gdc": sha256(gdc_path),
            },
            "patient_level_output": str(audit_path),
        }
    )
    summary_path = args.work_dir / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate["cohorts"], indent=2, sort_keys=True))
    print(f"Patient-level audit: {audit_path}")
    print(f"Aggregate summary: {summary_path}")


if __name__ == "__main__":
    main()
