"""Paired survival benchmark: ResNet18 2.5D, attention and Mamba.

The historical patient-level ResNet18 embedding is kept as the baseline. The
attention and Mamba models train only on outcome-independent frozen axial token
caches. All three variants use identical outer/inner splits, and held-out data
is evaluated only after validation-based early stopping.
"""

from __future__ import annotations

import argparse
import copy
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

from components.adapters.ingestion.vision.models.sequence_pooling import (  # noqa: E402
    build_sequence_model,
    cox_ph_loss,
)
from components.processors.prognosis.models.linear_cox import (  # noqa: E402
    PrognosisProc_LinearCox,
)


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
EMBEDDING_COLUMNS = [f"z{i:03d}" for i in range(768)]
MODEL_NAMES = (
    "resnet18_2p5d_baseline",
    "resnet18_2p5d_attention",
    "resnet18_2p5d_mamba",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"case_id", *EMBEDDING_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"Baseline cache is missing columns: {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("Baseline cache contains duplicate case_id values")
    values = frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Baseline cache contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Baseline cache contains zero-norm embeddings")
    values /= norms[:, None]
    return pd.DataFrame(values, index=frame["case_id"], columns=EMBEDDING_COLUMNS)


def load_sequences(
    sequence_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    manifest_path = sequence_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sequence manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"case_id", "sequence_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Sequence manifest is missing columns: {sorted(missing)}")
    manifest["case_id"] = manifest["case_id"].astype(str).str.strip().str.upper()
    if manifest["case_id"].duplicated().any():
        raise ValueError("Sequence manifest contains duplicate case_id values")
    result = {}
    for row in manifest.itertuples(index=False):
        path = Path(str(row.sequence_path))
        if not path.is_absolute():
            path = sequence_dir / path
        with np.load(path, allow_pickle=False) as payload:
            features = payload["features"].astype(np.float32)
            positions = payload["positions"].astype(np.float32)
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError(f"Invalid sequence feature shape for {row.case_id}: {features.shape}")
        if positions.shape != (features.shape[0],):
            raise ValueError(f"Invalid sequence positions for {row.case_id}")
        if not np.isfinite(features).all() or not np.isfinite(positions).all():
            raise ValueError(f"Non-finite sequence values for {row.case_id}")
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms <= 0):
            raise ValueError(f"Zero-norm sequence token for {row.case_id}")
        result[row.case_id] = (features / norms[:, None], positions)
    if not result:
        raise ValueError(f"No valid sequence cases found under {sequence_dir}")
    return result


def load_targets(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"case_id", "survival_days", "event"} - set(frame.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    frame["case_id"] = frame["case_id"].astype(str).str.strip().str.upper()
    if frame["case_id"].duplicated().any():
        raise ValueError("Target table contains duplicate case_id values")
    frame["survival_days"] = pd.to_numeric(frame["survival_days"], errors="coerce")
    frame["event"] = pd.to_numeric(frame["event"], errors="coerce")
    return frame.set_index("case_id")[["survival_days", "event"]]


def pad_sequences(
    sequences: dict[str, tuple[np.ndarray, np.ndarray]], case_ids: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(sequences[case_id][0].shape[0] for case_id in case_ids)
    features = np.zeros((len(case_ids), max_length, 512), dtype=np.float32)
    positions = np.zeros((len(case_ids), max_length), dtype=np.float32)
    mask = np.zeros((len(case_ids), max_length), dtype=bool)
    for row, case_id in enumerate(case_ids):
        case_features, case_positions = sequences[case_id]
        length = len(case_features)
        features[row, :length] = case_features
        positions[row, :length] = case_positions
        mask[row, :length] = True
    return (
        torch.from_numpy(features),
        torch.from_numpy(positions),
        torch.from_numpy(mask),
    )


def safe_cindex(
    survival: np.ndarray, risk: np.ndarray, events: np.ndarray
) -> float:
    try:
        return float(concordance_index(survival, -risk, events))
    except ZeroDivisionError:
        return float("nan")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_baseline(
    embeddings: np.ndarray,
    survival: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    heldout_idx: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[np.ndarray, dict]:
    seed_everything(seed)
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
    return head.predict_risk(x[heldout_idx]).astype(np.float64), {
        "best_validation_cindex": float(fit_result["best_val_cindex"]),
        "epochs_completed": len(fit_result["history"]["train_loss"]),
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
    }


def fit_sequence_model(
    name: str,
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    heldout_idx: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    model_dim: int,
    attention_dim: int,
    state_dim: int,
    mamba_blocks: int,
    dropout: float,
) -> tuple[np.ndarray, dict]:
    seed_everything(seed)
    common = {
        "input_dim": 512,
        "model_dim": model_dim,
        "attention_dim": attention_dim,
        "dropout": dropout,
    }
    if name == "mamba":
        common.update({"state_dim": state_dim, "n_blocks": mamba_blocks})
    model = build_sequence_model(name, **common).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    train_tensors = (
        features[train_idx].to(device),
        positions[train_idx].to(device),
        mask[train_idx].to(device),
        torch.tensor(survival[train_idx], dtype=torch.float32, device=device),
        torch.tensor(events[train_idx], dtype=torch.float32, device=device),
    )
    validation_tensors = (
        features[validation_idx].to(device),
        positions[validation_idx].to(device),
        mask[validation_idx].to(device),
    )
    best_state = None
    best_validation = -float("inf")
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(*train_tensors[:3])
        loss = cox_ph_loss(risk, train_tensors[3], train_tensors[4])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epochs_completed = epoch + 1

        model.eval()
        with torch.inference_mode():
            validation_risk = model(*validation_tensors).detach().cpu().numpy()
        validation_cindex = safe_cindex(
            survival[validation_idx], validation_risk, events[validation_idx]
        )
        score = validation_cindex if np.isfinite(validation_cindex) else -float("inf")
        if score > best_validation + 1e-8:
            best_validation = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        heldout_risk = model(
            features[heldout_idx].to(device),
            positions[heldout_idx].to(device),
            mask[heldout_idx].to(device),
        ).cpu().numpy().astype(np.float64)
    return heldout_risk, {
        "best_validation_cindex": float(best_validation),
        "epochs_completed": epochs_completed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-embeddings", required=True, type=Path)
    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--mamba-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    baseline = load_baseline(args.baseline_embeddings)
    sequences = load_sequences(args.sequence_dir)
    targets = load_targets(args.targets)
    common_ids = sorted(set(baseline.index) & set(sequences) & set(targets.index))
    cohort = targets.loc[common_ids]
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
    baseline_values = baseline.loc[common_ids].to_numpy(dtype=np.float32)
    features, positions, mask = pad_sequences(sequences, common_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    all_indices = np.arange(len(common_ids))
    split_rows = []
    prediction_rows = []
    metric_rows = []
    for seed in args.seeds:
        outer_train, heldout = train_test_split(
            all_indices,
            test_size=args.holdout_fraction,
            stratify=events,
            random_state=int(seed),
        )
        train, validation = train_test_split(
            outer_train,
            test_size=args.validation_fraction,
            stratify=events[outer_train],
            random_state=int(seed) * 100,
        )
        membership = {
            **{int(index): "train" for index in train},
            **{int(index): "validation" for index in validation},
            **{int(index): "heldout" for index in heldout},
        }
        split_rows.extend({
            "seed": int(seed),
            "case_id": common_ids[index],
            "partition": membership[index],
            "survival_days": float(survival[index]),
            "event": int(events[index]),
        } for index in all_indices)

        initialization_seed = int(seed) * 1000 + 17
        risks = {}
        metadata = {}
        risks[MODEL_NAMES[0]], metadata[MODEL_NAMES[0]] = fit_baseline(
            baseline_values, survival, events, train, validation, heldout,
            initialization_seed, args.epochs, args.patience, args.lr,
            args.weight_decay,
        )
        for short_name, model_name in (
            ("attention", MODEL_NAMES[1]),
            ("mamba", MODEL_NAMES[2]),
        ):
            risks[model_name], metadata[model_name] = fit_sequence_model(
                short_name, features, positions, mask, survival, events,
                train, validation, heldout, initialization_seed, args.epochs,
                args.patience, args.lr, args.weight_decay, device,
                args.model_dim, args.attention_dim, args.state_dim,
                args.mamba_blocks, args.dropout,
            )
        cindices = {
            name: safe_cindex(survival[heldout], risk, events[heldout])
            for name, risk in risks.items()
        }
        metric_rows.append({
            "seed": int(seed),
            "n_train": len(train),
            "n_validation": len(validation),
            "n_heldout": len(heldout),
            "events_heldout": int(events[heldout].sum()),
            **{f"cindex_{name}": cindices[name] for name in MODEL_NAMES},
            "delta_attention_minus_baseline": cindices[MODEL_NAMES[1]] - cindices[MODEL_NAMES[0]],
            "delta_mamba_minus_baseline": cindices[MODEL_NAMES[2]] - cindices[MODEL_NAMES[0]],
            "delta_mamba_minus_attention": cindices[MODEL_NAMES[2]] - cindices[MODEL_NAMES[1]],
            **{
                f"{name}_{field}": metadata[name][field]
                for name in MODEL_NAMES
                for field in ("best_validation_cindex", "epochs_completed", "parameters")
            },
        })
        for local, index in enumerate(heldout):
            prediction_rows.append({
                "seed": int(seed),
                "case_id": common_ids[index],
                "survival_days": float(survival[index]),
                "event": int(events[index]),
                **{f"risk_{name}": float(risks[name][local]) for name in MODEL_NAMES},
            })
        print(
            f"seed={seed} " + " ".join(
                f"{name}={cindices[name]:.4f}" for name in MODEL_NAMES
            ),
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "heldout_predictions.csv", index=False
    )
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    results = {}
    for name in MODEL_NAMES:
        column = f"cindex_{name}"
        results[name] = {
            "cindex_mean": float(metrics[column].mean()),
            "cindex_std_across_seeds": float(metrics[column].std(ddof=1)),
            "cindex_per_seed": metrics[column].tolist(),
        }
    summary = {
        "status": "paired_repeated_holdout",
        "cohort": {"n_cases": len(common_ids), "n_events": int(events.sum())},
        "protocol": {
            "outer_split": "event-stratified repeated holdout",
            "holdout_fraction": args.holdout_fraction,
            "inner_validation_fraction": args.validation_fraction,
            "seeds": [int(seed) for seed in args.seeds],
            "frozen_encoder": "ResNet18 ImageNet1K V1",
            "attention_and_mamba_same_axial_tokens": True,
            "heldout_used_for_selection": False,
            "device": str(device),
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "results": results,
        "mean_deltas": {
            "attention_minus_baseline": float(metrics["delta_attention_minus_baseline"].mean()),
            "mamba_minus_baseline": float(metrics["delta_mamba_minus_baseline"].mean()),
            "mamba_minus_attention": float(metrics["delta_mamba_minus_attention"].mean()),
        },
        "limitations": [
            "Repeated holdouts overlap and are not independent confidence intervals.",
            "The baseline uses three central anatomical views; sequence variants use axial tokens.",
            "The PyTorch selective scan is mathematically Mamba-style but not the fused mamba-ssm kernel.",
            "External validation is required before making performance claims.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "baseline_embeddings": str(args.baseline_embeddings.resolve()),
        "baseline_sha256": file_sha256(args.baseline_embeddings),
        "sequence_manifest": str((args.sequence_dir / "manifest.csv").resolve()),
        "sequence_manifest_sha256": file_sha256(args.sequence_dir / "manifest.csv"),
        "targets": str(args.targets.resolve()),
        "targets_sha256": file_sha256(args.targets),
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
    }
    provenance["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in provenance["arguments"].items()
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
