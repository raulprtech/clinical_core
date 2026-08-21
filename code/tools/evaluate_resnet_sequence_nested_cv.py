"""Confirmatory nested repeated CV for ResNet18 sequence survival models.

The official train-only PCA+Cox baseline, attention pooling and a Mamba-style
selective SSM use identical repeated outer folds. PCA/Cox hyperparameters and
neural epoch counts are selected strictly within each outer-train partition.
Every candidate is then reinitialized/refitted on the complete outer-train
before the outer fold is evaluated.

Patient-level splits and predictions are written for local audit. They must not
be published; only aggregate fold/repeat metrics, bootstrap summaries and
non-identifying provenance are suitable for version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from components.adapters.ingestion.vision.models.sequence_pooling import (  # noqa: E402
    build_sequence_model,
    cox_ph_loss,
)
from evaluate_resnet_official_comparison import (  # noqa: E402
    DEFAULT_PCA_DIMS,
    DEFAULT_PENALIZERS,
)
from evaluate_resnet_sequence_models import (  # noqa: E402
    load_baseline,
    load_sequences,
    load_targets,
    pad_sequences,
    safe_cindex,
    seed_everything,
)
from evaluate_trimodal_fusion import fit_predict, select_spec_and_oof  # noqa: E402


MODEL_NAMES = ("official_pca_cox", "attention", "mamba")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(name: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    kwargs = {
        "input_dim": 512,
        "model_dim": args.model_dim,
        "attention_dim": args.attention_dim,
        "dropout": args.dropout,
        "use_position": getattr(args, "use_position", True),
    }
    if name == "mamba":
        kwargs.update({"state_dim": args.state_dim, "n_blocks": args.mamba_blocks})
    return build_sequence_model(name, **kwargs).to(device)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: torch.Tensor,
    events: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    risk = model(features, positions, mask)
    loss = cox_ph_loss(risk, survival, events)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu())


def select_epoch(
    name: str,
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float]:
    seed_everything(seed)
    model = make_model(name, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_features = features[train_idx].to(device)
    train_positions = positions[train_idx].to(device)
    train_mask = mask[train_idx].to(device)
    train_survival = torch.tensor(
        survival[train_idx], dtype=torch.float32, device=device
    )
    train_events = torch.tensor(events[train_idx], dtype=torch.float32, device=device)
    validation_features = features[validation_idx].to(device)
    validation_positions = positions[validation_idx].to(device)
    validation_mask = mask[validation_idx].to(device)

    best_epoch = 1
    best_cindex = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(
            model,
            optimizer,
            train_features,
            train_positions,
            train_mask,
            train_survival,
            train_events,
        )
        model.eval()
        with torch.inference_mode():
            validation_risk = model(
                validation_features, validation_positions, validation_mask
            ).cpu().numpy()
        score = safe_cindex(
            survival[validation_idx], validation_risk, events[validation_idx]
        )
        score = score if np.isfinite(score) else -float("inf")
        if score > best_cindex + 1e-8:
            best_cindex = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    return int(best_epoch), float(best_cindex)


def select_epochs_inner(
    name: str,
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    outer_train: np.ndarray,
    split_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, list[int], list[float]]:
    splitter = StratifiedKFold(
        n_splits=args.inner_folds, shuffle=True, random_state=split_seed
    )
    epochs = []
    scores = []
    model_offset = 100 if name == "attention" else 200
    for inner_fold, (inner_train_local, validation_local) in enumerate(
        splitter.split(outer_train, events[outer_train])
    ):
        inner_train = outer_train[inner_train_local]
        validation = outer_train[validation_local]
        selected, score = select_epoch(
            name,
            features,
            positions,
            mask,
            survival,
            events,
            inner_train,
            validation,
            split_seed * 1000 + model_offset + inner_fold,
            args,
            device,
        )
        epochs.append(selected)
        scores.append(score)
    chosen = max(1, int(np.rint(np.median(epochs))))
    return chosen, epochs, scores


def refit_and_predict(
    name: str,
    features: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    survival: np.ndarray,
    events: np.ndarray,
    outer_train: np.ndarray,
    heldout: np.ndarray,
    epochs: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    seed_everything(seed)
    model = make_model(name, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_features = features[outer_train].to(device)
    train_positions = positions[outer_train].to(device)
    train_mask = mask[outer_train].to(device)
    train_survival = torch.tensor(
        survival[outer_train], dtype=torch.float32, device=device
    )
    train_events = torch.tensor(
        events[outer_train], dtype=torch.float32, device=device
    )
    for _ in range(epochs):
        train_epoch(
            model,
            optimizer,
            train_features,
            train_positions,
            train_mask,
            train_survival,
            train_events,
        )
    model.eval()
    with torch.inference_mode():
        risk = model(
            features[heldout].to(device),
            positions[heldout].to(device),
            mask[heldout].to(device),
        ).cpu().numpy().astype(np.float64)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return risk, int(parameters)


def summarize_repeats(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Repeat {repeat} contains duplicate held-out patients")
        row = {
            "repeat": int(repeat),
            "n_cases": len(frame),
            "n_events": int(frame["event"].sum()),
        }
        for name in MODEL_NAMES:
            row[f"cindex_{name}"] = safe_cindex(
                frame["survival_days"].to_numpy(dtype=np.float64),
                frame[f"risk_{name}"].to_numpy(dtype=np.float64),
                frame["event"].to_numpy(dtype=np.int64),
            )
        row["delta_mamba_minus_official"] = (
            row["cindex_mamba"] - row["cindex_official_pca_cox"]
        )
        row["delta_mamba_minus_attention"] = (
            row["cindex_mamba"] - row["cindex_attention"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_patient_bootstrap(
    predictions: pd.DataFrame,
    comparisons: tuple[tuple[str, str], ...],
    n_iter: int,
    seed: int,
) -> pd.DataFrame:
    repeats = []
    expected_ids = None
    for _, frame in predictions.groupby("repeat", sort=True):
        ordered = frame.sort_values("case_id").reset_index(drop=True)
        ids = ordered["case_id"].tolist()
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("Each repeat must contain the same ordered patient set")
        repeats.append(ordered)
    if not repeats:
        raise ValueError("No repeated predictions were provided")
    n_cases = len(repeats[0])
    rng = np.random.default_rng(seed)
    rows = []
    for candidate, reference in comparisons:
        observed_by_repeat = [
            safe_cindex(
                frame["survival_days"].to_numpy(),
                frame[f"risk_{candidate}"].to_numpy(),
                frame["event"].to_numpy(),
            )
            - safe_cindex(
                frame["survival_days"].to_numpy(),
                frame[f"risk_{reference}"].to_numpy(),
                frame["event"].to_numpy(),
            )
            for frame in repeats
        ]
        samples = []
        for _ in range(n_iter):
            indices = rng.integers(0, n_cases, size=n_cases)
            deltas = []
            for frame in repeats:
                survival = frame["survival_days"].to_numpy()[indices]
                events = frame["event"].to_numpy()[indices]
                candidate_risk = frame[f"risk_{candidate}"].to_numpy()[indices]
                reference_risk = frame[f"risk_{reference}"].to_numpy()[indices]
                candidate_ci = safe_cindex(survival, candidate_risk, events)
                reference_ci = safe_cindex(survival, reference_risk, events)
                if np.isfinite(candidate_ci) and np.isfinite(reference_ci):
                    deltas.append(candidate_ci - reference_ci)
            if deltas:
                samples.append(float(np.mean(deltas)))
        if not samples:
            raise RuntimeError("Clustered bootstrap produced no valid samples")
        samples_array = np.asarray(samples, dtype=np.float64)
        p_value = min(
            1.0,
            2.0
            * min(
                float(np.mean(samples_array <= 0)),
                float(np.mean(samples_array >= 0)),
            ),
        )
        rows.append({
            "candidate": candidate,
            "reference": reference,
            "mean_delta_across_repeats": float(np.mean(observed_by_repeat)),
            "ci95_lo": float(np.quantile(samples_array, 0.025)),
            "ci95_hi": float(np.quantile(samples_array, 0.975)),
            "bootstrap_p": p_value,
            "n_bootstrap": int(len(samples_array)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-embeddings", required=True, type=Path)
    parser.add_argument("--sequence-dir", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--mamba-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=DEFAULT_PCA_DIMS)
    parser.add_argument("--penalizers", nargs="+", type=float, default=DEFAULT_PENALIZERS)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.outer_folds < 2 or args.outer_repeats < 1 or args.inner_folds < 2:
        raise ValueError("CV fold counts and repeats must be positive and valid")
    if args.epochs < 1 or args.patience < 1 or args.bootstrap_iterations < 1:
        raise ValueError("Training and bootstrap counts must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)

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
    survival = cohort["survival_days"].to_numpy(dtype=np.float64)
    events = cohort["event"].to_numpy(dtype=np.int64)
    if len(common_ids) < args.outer_folds * 2:
        raise ValueError("Too few valid common patients for requested outer CV")
    if np.min(np.bincount(events, minlength=2)) < args.outer_folds:
        raise ValueError("Each outcome stratum must support every outer fold")
    baseline_values = baseline.loc[common_ids].to_numpy(dtype=np.float64)
    features, positions, mask = pad_sequences(sequences, common_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.outer_folds,
        n_repeats=args.outer_repeats,
        random_state=args.random_state,
    )
    fold_rows = []
    prediction_rows = []
    split_rows = []
    started = time.monotonic()
    for split_number, (outer_train, heldout) in enumerate(
        splitter.split(np.zeros(len(events)), events)
    ):
        repeat = split_number // args.outer_folds
        fold = split_number % args.outer_folds
        split_seed = args.random_state + repeat * 100 + fold

        spec, _, official_inner_cindex = select_spec_and_oof(
            baseline_values[outer_train],
            survival[outer_train],
            events[outer_train],
            "vision",
            split_seed,
            args.pca_dims,
            args.penalizers,
            args.inner_folds,
        )
        _, official_risk = fit_predict(
            baseline_values,
            survival,
            events,
            outer_train,
            heldout,
            "vision",
            spec,
            split_seed * 1000 + 7,
        )
        risks = {"official_pca_cox": official_risk}
        metadata = {}
        for name, offset in (("attention", 17), ("mamba", 29)):
            selected, inner_epochs, inner_scores = select_epochs_inner(
                name,
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                split_seed,
                args,
                device,
            )
            risk, parameters = refit_and_predict(
                name,
                features,
                positions,
                mask,
                survival,
                events,
                outer_train,
                heldout,
                selected,
                split_seed * 1000 + offset,
                args,
                device,
            )
            risks[name] = risk
            metadata[name] = {
                "selected_epochs": selected,
                "inner_epochs": inner_epochs,
                "inner_cindex_mean": float(np.mean(inner_scores)),
                "parameters": parameters,
            }

        cindices = {
            name: safe_cindex(survival[heldout], risks[name], events[heldout])
            for name in MODEL_NAMES
        }
        fold_rows.append({
            "repeat": repeat,
            "fold": fold,
            "split_seed": split_seed,
            "n_outer_train": len(outer_train),
            "n_heldout": len(heldout),
            "events_heldout": int(events[heldout].sum()),
            "official_pca_dim": int(spec.pca_dim),
            "official_penalizer": float(spec.penalizer),
            "official_inner_cindex": float(official_inner_cindex),
            **{f"cindex_{name}": cindices[name] for name in MODEL_NAMES},
            "delta_mamba_minus_official": (
                cindices["mamba"] - cindices["official_pca_cox"]
            ),
            "delta_mamba_minus_attention": (
                cindices["mamba"] - cindices["attention"]
            ),
            **{
                f"{name}_{field}": metadata[name][field]
                for name in ("attention", "mamba")
                for field in (
                    "selected_epochs",
                    "inner_cindex_mean",
                    "parameters",
                )
            },
            "attention_inner_epochs": json.dumps(metadata["attention"]["inner_epochs"]),
            "mamba_inner_epochs": json.dumps(metadata["mamba"]["inner_epochs"]),
        })
        train_set = set(outer_train.tolist())
        split_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": case_id,
            "partition": "outer_train" if index in train_set else "heldout",
            "survival_days": float(survival[index]),
            "event": int(events[index]),
        } for index, case_id in enumerate(common_ids))
        prediction_rows.extend({
            "repeat": repeat,
            "fold": fold,
            "case_id": common_ids[index],
            "survival_days": float(survival[index]),
            "event": int(events[index]),
            **{f"risk_{name}": float(risks[name][local]) for name in MODEL_NAMES},
        } for local, index in enumerate(heldout))
        print(
            f"repeat={repeat} fold={fold} official={cindices['official_pca_cox']:.4f} "
            f"attention={cindices['attention']:.4f} mamba={cindices['mamba']:.4f} "
            f"epochs(attn/mamba)={metadata['attention']['selected_epochs']}/"
            f"{metadata['mamba']['selected_epochs']}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeats = summarize_repeats(predictions)
    bootstrap = clustered_patient_bootstrap(
        predictions,
        (("mamba", "official_pca_cox"), ("mamba", "attention")),
        args.bootstrap_iterations,
        args.random_state + 50000,
    )
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_cluster_bootstrap.csv", index=False)

    results = {}
    for name in MODEL_NAMES:
        column = f"cindex_{name}"
        results[name] = {
            "mean_pooled_repeat_cindex": float(repeats[column].mean()),
            "std_across_repeats": float(repeats[column].std(ddof=1))
            if len(repeats) > 1
            else None,
            "per_repeat": repeats[column].tolist(),
            "mean_outer_fold_cindex": float(folds[column].mean()),
        }
    summary = {
        "status": "confirmatory_nested_repeated_cv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
        "cohort": {"n_cases": len(common_ids), "n_events": int(events.sum())},
        "protocol": {
            "outer_cv": "event-stratified repeated K-fold",
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "epoch_selection": "median best epoch across inner folds",
            "outer_refit": "all models reinitialized and fit on complete outer-train",
            "official_baseline": "train-only scaler + PCA + Cox ridge",
            "patient_clustered_bootstrap_iterations": args.bootstrap_iterations,
            "same_outer_splits": True,
            "same_axial_tokens_for_attention_and_mamba": True,
            "device": str(device),
        },
        "results": results,
        "paired_comparisons": bootstrap.to_dict(orient="records"),
        "claim_boundary": (
            "Confirmatory internal resampling only; external generalization remains untested."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "baseline_embeddings": str(args.baseline_embeddings.resolve()),
        "baseline_sha256": file_sha256(args.baseline_embeddings),
        "sequence_manifest": str((args.sequence_dir / "manifest.csv").resolve()),
        "sequence_manifest_sha256": file_sha256(args.sequence_dir / "manifest.csv"),
        "targets": str(args.targets.resolve()),
        "targets_sha256": file_sha256(args.targets),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(summary["results"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
