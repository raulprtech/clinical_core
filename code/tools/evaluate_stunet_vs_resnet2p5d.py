"""Paired preliminary survival evaluation of frozen STU-Net and ResNet18 2.5D.

Both encoders are evaluated on the exact same patients, outcomes, outer
80/20 event-stratified holdouts, inner validation splits, linear Cox head,
hyperparameters, and initial head weights. The held-out partition is used
only once, after early stopping on a validation subset of the outer train
pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.processors.prognosis.models.linear_cox import (  # noqa: E402
    PrognosisProc_LinearCox,
)
from components.processors.prognosis.utils.statistical_tests import (  # noqa: E402
    bootstrap_cindex_ci,
    paired_cindex_test,
)


EMBEDDING_COLUMNS = [f"z{i:03d}" for i in range(768)]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resnet(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"case_id", *EMBEDDING_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ResNet cache is missing columns: {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("ResNet cache contains duplicate case_id values")
    result = frame.set_index("case_id")[EMBEDDING_COLUMNS].astype(np.float32)
    return validate_embeddings(result, "resnet18_2p5d")


def load_stunet(cases_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for complete_path in sorted(cases_dir.glob("*/complete.json")):
        payload = json.loads(complete_path.read_text())
        case_id = str(payload.get("case_id", complete_path.parent.name)).strip().upper()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != 768:
            raise ValueError(f"Invalid STU-Net embedding for {case_id}")
        rows.append({"case_id": case_id, **dict(zip(EMBEDDING_COLUMNS, embedding))})
    if not rows:
        raise ValueError(f"No complete STU-Net cases found under {cases_dir}")
    frame = pd.DataFrame(rows)
    if frame["case_id"].duplicated().any():
        raise ValueError("STU-Net cache contains duplicate case_id values")
    result = frame.set_index("case_id")[EMBEDDING_COLUMNS].astype(np.float32)
    return validate_embeddings(result, "stunet_s_fp32")


def validate_embeddings(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite embedding values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError(f"{label} contains zero-norm embeddings")
    # Re-normalization only removes serialization round-off and applies the
    # same deterministic contract to both encoders.
    values = values / norms[:, None]
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def load_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"case_id", "survival_days", "event"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("Target table contains duplicate case_id values")
    frame["survival_days"] = pd.to_numeric(frame["survival_days"], errors="coerce")
    frame["event"] = pd.to_numeric(frame["event"], errors="coerce")
    return frame.set_index("case_id")[["survival_days", "event"]]


def fit_one_head(
    embeddings: np.ndarray,
    survival: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    heldout_idx: np.ndarray,
    initialization_seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[np.ndarray, dict]:
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)

    x = torch.tensor(embeddings, dtype=torch.float32)
    head = PrognosisProc_LinearCox(
        fused_dim=x.shape[1], lr=lr, weight_decay=weight_decay
    )
    fit_result = head.fit(
        x[train_idx],
        torch.tensor(survival[train_idx], dtype=torch.float32),
        torch.tensor(events[train_idx], dtype=torch.float32),
        x[validation_idx],
        torch.tensor(survival[validation_idx], dtype=torch.float32),
        torch.tensor(events[validation_idx], dtype=torch.float32),
        epochs=epochs,
        patience=patience,
        verbose=False,
    )
    risk = head.predict_risk(x[heldout_idx]).astype(np.float64)
    metadata = {
        "best_validation_cindex": float(fit_result["best_val_cindex"]),
        "epochs_completed": int(len(fit_result["history"]["train_loss"])),
    }
    return risk, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stunet-cases-dir", type=Path, required=True)
    parser.add_argument("--resnet-embeddings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    if not 0 < args.holdout_fraction < 1:
        raise ValueError("holdout-fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    stunet = load_stunet(args.stunet_cases_dir)
    resnet = load_resnet(args.resnet_embeddings)
    targets = load_targets(args.targets)
    common_ids = sorted(set(stunet.index) & set(resnet.index) & set(targets.index))
    cohort = targets.loc[common_ids].copy()
    valid = (
        cohort["survival_days"].notna()
        & (cohort["survival_days"] > 0)
        & cohort["event"].isin([0, 1])
    )
    cohort = cohort.loc[valid].sort_index()
    common_ids = cohort.index.tolist()
    if len(common_ids) < 20:
        raise ValueError(f"Too few common cases with valid outcomes: {len(common_ids)}")

    survival = cohort["survival_days"].to_numpy(dtype=np.float64)
    events = cohort["event"].to_numpy(dtype=np.int64)
    if min(np.bincount(events, minlength=2)) < 2:
        raise ValueError("Both outcome strata need at least two patients")
    embedding_by_model = {
        "stunet_s_fp32_frozen": stunet.loc[common_ids].to_numpy(dtype=np.float32),
        "resnet18_2p5d_imagenet_frozen": resnet.loc[common_ids].to_numpy(dtype=np.float32),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_out = cohort.reset_index()
    cohort_out["in_stunet"] = True
    cohort_out["in_resnet18_2p5d"] = True
    cohort_out.to_csv(args.output_dir / "cohort_common.csv", index=False)

    all_indices = np.arange(len(common_ids))
    split_rows: list[dict] = []
    prediction_rows: list[dict] = []
    seed_rows: list[dict] = []

    for seed in args.seeds:
        outer_train_idx, heldout_idx = train_test_split(
            all_indices,
            test_size=args.holdout_fraction,
            stratify=events,
            random_state=int(seed),
        )
        train_idx, validation_idx = train_test_split(
            outer_train_idx,
            test_size=args.validation_fraction,
            stratify=events[outer_train_idx],
            random_state=int(seed) * 100,
        )
        membership = {
            **{int(idx): "train" for idx in train_idx},
            **{int(idx): "validation" for idx in validation_idx},
            **{int(idx): "heldout" for idx in heldout_idx},
        }
        split_rows.extend(
            {
                "seed": int(seed),
                "case_id": common_ids[idx],
                "partition": membership[idx],
                "survival_days": float(survival[idx]),
                "event": int(events[idx]),
            }
            for idx in all_indices
        )

        risks: dict[str, np.ndarray] = {}
        model_meta: dict[str, dict] = {}
        initialization_seed = int(seed) * 1000 + 17
        for model_name, embeddings in embedding_by_model.items():
            risk, metadata = fit_one_head(
                embeddings,
                survival,
                events,
                train_idx,
                validation_idx,
                heldout_idx,
                initialization_seed,
                args.epochs,
                args.patience,
                args.lr,
                args.weight_decay,
            )
            risks[model_name] = risk
            model_meta[model_name] = metadata

        stunet_name = "stunet_s_fp32_frozen"
        resnet_name = "resnet18_2p5d_imagenet_frozen"
        time_heldout = survival[heldout_idx]
        event_heldout = events[heldout_idx]
        ci_stunet = float(
            concordance_index(time_heldout, -risks[stunet_name], event_heldout)
        )
        ci_resnet = float(
            concordance_index(time_heldout, -risks[resnet_name], event_heldout)
        )
        ci_intervals = {}
        for offset, model_name in enumerate((stunet_name, resnet_name)):
            _, _, lo, hi = bootstrap_cindex_ci(
                risks[model_name],
                time_heldout,
                event_heldout,
                n_iter=args.bootstrap_iterations,
                seed=int(seed) + 10000 + offset,
            )
            ci_intervals[model_name] = (float(lo), float(hi))
        paired = paired_cindex_test(
            risks[resnet_name],
            risks[stunet_name],
            time_heldout,
            event_heldout,
            n_iter=args.bootstrap_iterations,
            seed=int(seed) + 20000,
        )

        seed_rows.append({
            "seed": int(seed),
            "n_train": int(len(train_idx)),
            "n_validation": int(len(validation_idx)),
            "n_heldout": int(len(heldout_idx)),
            "events_train": int(events[train_idx].sum()),
            "events_validation": int(events[validation_idx].sum()),
            "events_heldout": int(event_heldout.sum()),
            "cindex_stunet": ci_stunet,
            "cindex_stunet_ci95_lo": ci_intervals[stunet_name][0],
            "cindex_stunet_ci95_hi": ci_intervals[stunet_name][1],
            "cindex_resnet18_2p5d": ci_resnet,
            "cindex_resnet18_2p5d_ci95_lo": ci_intervals[resnet_name][0],
            "cindex_resnet18_2p5d_ci95_hi": ci_intervals[resnet_name][1],
            "delta_stunet_minus_resnet": ci_stunet - ci_resnet,
            "delta_ci95_lo": float(paired["delta_lo"]),
            "delta_ci95_hi": float(paired["delta_hi"]),
            "delta_bootstrap_p": float(paired["p_value"]),
            "stunet_best_validation_cindex": model_meta[stunet_name]["best_validation_cindex"],
            "resnet_best_validation_cindex": model_meta[resnet_name]["best_validation_cindex"],
            "stunet_epochs": model_meta[stunet_name]["epochs_completed"],
            "resnet_epochs": model_meta[resnet_name]["epochs_completed"],
        })
        for local_idx, cohort_idx in enumerate(heldout_idx):
            prediction_rows.append({
                "seed": int(seed),
                "case_id": common_ids[cohort_idx],
                "survival_days": float(survival[cohort_idx]),
                "event": int(events[cohort_idx]),
                "risk_stunet": float(risks[stunet_name][local_idx]),
                "risk_resnet18_2p5d": float(risks[resnet_name][local_idx]),
            })

    splits = pd.DataFrame(split_rows)
    predictions = pd.DataFrame(prediction_rows)
    per_seed = pd.DataFrame(seed_rows)
    splits.to_csv(args.output_dir / "splits.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)

    delta = per_seed["delta_stunet_minus_resnet"]
    summary = {
        "status": "preliminary_paired_holdout",
        "cohort": {
            "n_cases": int(len(common_ids)),
            "n_events": int(events.sum()),
            "case_ids": common_ids,
        },
        "protocol": {
            "outer_split": "80/20 stratified by event",
            "inner_validation_fraction_of_outer_train": args.validation_fraction,
            "seeds": [int(seed) for seed in args.seeds],
            "head": "PrognosisProc_LinearCox (768 -> 1)",
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "same_initial_head_weights_per_seed": True,
            "bootstrap_iterations_per_seed": args.bootstrap_iterations,
        },
        "results": {
            "stunet_s_fp32_frozen": {
                "cindex_mean": float(per_seed["cindex_stunet"].mean()),
                "cindex_std_across_seeds": float(per_seed["cindex_stunet"].std(ddof=1)),
                "cindex_per_seed": per_seed["cindex_stunet"].tolist(),
            },
            "resnet18_2p5d_imagenet_frozen": {
                "cindex_mean": float(per_seed["cindex_resnet18_2p5d"].mean()),
                "cindex_std_across_seeds": float(per_seed["cindex_resnet18_2p5d"].std(ddof=1)),
                "cindex_per_seed": per_seed["cindex_resnet18_2p5d"].tolist(),
            },
            "paired_delta_stunet_minus_resnet": {
                "mean": float(delta.mean()),
                "std_across_seeds": float(delta.std(ddof=1)),
                "per_seed": delta.tolist(),
                "stunet_wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "resnet_wins": int((delta < 0).sum()),
            },
        },
        "limitations": [
            "Only 50 common patients and 13 events are available.",
            "Each held-out set has 10 patients and only 2 or 3 events; uncertainty is large.",
            "Repeated holdouts overlap, so across-seed dispersion is not an independent-sample confidence interval.",
            "The local ResNet18 2.5D cache uses a frozen ImageNet backbone; this is not a split-specific fine-tuned ResNet.",
            "This preliminary result must not replace the final common-cohort comparison.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "stunet_cases_dir": str(args.stunet_cases_dir.resolve()),
            "resnet_embeddings": str(args.resnet_embeddings.resolve()),
            "resnet_embeddings_sha256": file_sha256(args.resnet_embeddings),
            "targets": str(args.targets.resolve()),
            "targets_sha256": file_sha256(args.targets),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(summary["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
