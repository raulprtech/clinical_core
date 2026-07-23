"""Compare compact concatenation and cross-attention with convex late fusion.

All methods use the same common cohort and outer 80/20 event-stratified
holdouts. Neural models use fixed, train-only dimensionality reduction and an
inner validation split for epoch selection; they are then reinitialized and
refitted on the complete outer-train partition for the selected epoch count.
"""

from __future__ import annotations

import argparse
import copy
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
from sklearn.preprocessing import StandardScaler


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.processors.fusion.models.cross_attention import (  # noqa: E402
    CrossAttentionSurvivalFusion,
    HierarchicalResidualSurvivalFusion,
    ProjectedConcatSurvivalFusion,
    count_trainable_parameters,
)
from components.processors.prognosis.models.linear_cox import (  # noqa: E402
    cox_partial_likelihood_loss,
)
from tools.evaluate_trimodal_fusion import (  # noqa: E402
    DEFAULT_SEEDS,
    EmbeddingTransform,
    TabularTransform,
    load_indexed_csv,
    load_text_npz,
    load_vision_csv,
)


MODALITIES = ("tabular", "text", "vision")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index(time, -risk, event))


class BlockPreprocessor:
    """Fit one leakage-safe transform and output scaler per modality."""

    def __init__(self, pca_dim: int, seed: int):
        self.pca_dim = int(pca_dim)
        self.seed = int(seed)

    def fit(self, values: dict[str, np.ndarray], indices: np.ndarray):
        self.transforms = {}
        self.output_scalers = {}
        self.output_dims = []
        for offset, modality in enumerate(MODALITIES):
            if modality == "tabular":
                transform = TabularTransform().fit(values[modality][indices])
            else:
                transform = EmbeddingTransform(
                    self.pca_dim, self.seed + offset
                ).fit(values[modality][indices])
            transformed = transform.transform(values[modality][indices])
            keep = np.isfinite(transformed).all(axis=0) & (
                np.std(transformed, axis=0) > 1e-8
            )
            if not keep.any():
                raise ValueError(f"No nonconstant {modality} features remain")
            scaler = StandardScaler().fit(transformed[:, keep])
            self.transforms[modality] = (transform, keep)
            self.output_scalers[modality] = scaler
            self.output_dims.append(int(keep.sum()))
        return self

    def transform(
        self, values: dict[str, np.ndarray], indices: np.ndarray
    ) -> list[torch.Tensor]:
        blocks = []
        for modality in MODALITIES:
            transform, keep = self.transforms[modality]
            transformed = transform.transform(values[modality][indices])[:, keep]
            transformed = self.output_scalers[modality].transform(transformed)
            blocks.append(torch.tensor(transformed, dtype=torch.float32))
        return blocks


def make_model(
    architecture: str,
    modality_dims: list[int],
    d_model: int,
    num_heads: int,
    dropout: float,
):
    if architecture == "concat":
        return ProjectedConcatSurvivalFusion(
            modality_dims, d_model=d_model, dropout=dropout
        )
    if architecture == "cross_attention":
        return CrossAttentionSurvivalFusion(
            modality_dims,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
    if architecture == "hierarchical":
        return HierarchicalResidualSurvivalFusion(
            modality_dims, d_model=d_model, dropout=dropout
        )
    raise ValueError(architecture)


def train_epoch(
    model: torch.nn.Module,
    blocks: list[torch.Tensor],
    time: torch.Tensor,
    event: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    risk = model(blocks)
    loss = cox_partial_likelihood_loss(risk, time, event)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach())


def select_epoch(
    architecture: str,
    modality_dims: list[int],
    train_blocks: list[torch.Tensor],
    train_time: np.ndarray,
    train_event: np.ndarray,
    validation_blocks: list[torch.Tensor],
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    seed: int,
    d_model: int,
    num_heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> tuple[int, float, int]:
    set_seed(seed)
    model = make_model(architecture, modality_dims, d_model, num_heads, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    time_tensor = torch.tensor(train_time, dtype=torch.float32)
    event_tensor = torch.tensor(train_event, dtype=torch.float32)
    best_ci = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(max_epochs):
        train_epoch(model, train_blocks, time_tensor, event_tensor, optimizer)
        model.eval()
        with torch.no_grad():
            validation_risk = model(validation_blocks).cpu().numpy()
        score = cindex(validation_time, validation_event, validation_risk)
        if score > best_ci + 1e-12:
            best_ci = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{architecture} did not produce a validation checkpoint")
    return best_epoch + 1, float(best_ci), count_trainable_parameters(model)


def refit_and_predict(
    architecture: str,
    modality_dims: list[int],
    train_blocks: list[torch.Tensor],
    train_time: np.ndarray,
    train_event: np.ndarray,
    heldout_blocks: list[torch.Tensor],
    epochs: int,
    seed: int,
    d_model: int,
    num_heads: int,
    dropout: float,
    lr: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    set_seed(seed)
    model = make_model(architecture, modality_dims, d_model, num_heads, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    time_tensor = torch.tensor(train_time, dtype=torch.float32)
    event_tensor = torch.tensor(train_event, dtype=torch.float32)
    for _ in range(epochs):
        train_epoch(model, train_blocks, time_tensor, event_tensor, optimizer)
    model.eval()
    with torch.no_grad():
        risk = model(heldout_blocks).cpu().numpy().astype(np.float64)
        weights = model.modality_weights(heldout_blocks).cpu().numpy()
        pairwise = None
        if architecture == "cross_attention":
            pairwise = model.pairwise_attention(heldout_blocks).cpu().numpy()
    return risk, weights, pairwise, count_trainable_parameters(model)


def load_convex_reference(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(path / "per_seed_metrics.csv")
    predictions = pd.read_csv(path / "heldout_predictions.csv")
    required_metrics = {"seed", "cindex_late_tuned_simplex"}
    required_predictions = {"seed", "case_id", "risk_late_tuned_simplex"}
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("Convex metrics are incomplete")
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("Convex heldout predictions are incomplete")
    return metrics, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--vision-label", default="vision")
    parser.add_argument("--convex-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    features = load_indexed_csv(args.features, {"case_id"}).astype(np.float64)
    targets = load_indexed_csv(
        args.targets, {"case_id", "survival_days", "event"}
    )[["survival_days", "event"]]
    text = load_text_npz(args.text_embeddings)
    vision = load_vision_csv(args.vision_embeddings)
    common = sorted(set(features.index) & set(targets.index) & set(text.index) & set(vision.index))
    cohort = targets.loc[common].copy()
    cohort["survival_days"] = pd.to_numeric(cohort["survival_days"], errors="coerce")
    cohort["event"] = pd.to_numeric(cohort["event"], errors="coerce")
    cohort = cohort.loc[
        cohort["survival_days"].notna()
        & (cohort["survival_days"] > 0)
        & cohort["event"].isin([0, 1])
    ].sort_index()
    ids = cohort.index.tolist()
    if len(ids) < 50:
        raise ValueError(f"Only {len(ids)} common patients have valid outcomes")
    time = cohort["survival_days"].to_numpy(dtype=np.float64)
    event = cohort["event"].to_numpy(dtype=np.int64)
    values = {
        "tabular": features.loc[ids].to_numpy(dtype=np.float64),
        "text": text.loc[ids].to_numpy(dtype=np.float64),
        "vision": vision.loc[ids].to_numpy(dtype=np.float64),
    }
    convex_metrics, convex_predictions = load_convex_reference(args.convex_results_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    rows = []
    prediction_rows = []
    attention_rows = []
    all_idx = np.arange(len(ids))
    architectures = ("concat", "cross_attention")
    for seed in args.seeds:
        outer_train, heldout = train_test_split(
            all_idx, test_size=0.20, stratify=event, random_state=int(seed)
        )
        expected_heldout = set(
            convex_predictions.loc[
                convex_predictions["seed"] == int(seed), "case_id"
            ].astype(str)
        )
        actual_heldout = {ids[index] for index in heldout}
        if expected_heldout != actual_heldout:
            raise ValueError(f"Convex and neural heldout IDs differ for seed {seed}")
        inner_train_local, validation_local = train_test_split(
            np.arange(len(outer_train)),
            test_size=0.20,
            stratify=event[outer_train],
            random_state=int(seed) * 100,
        )
        inner_train = outer_train[inner_train_local]
        validation = outer_train[validation_local]

        inner_preprocessor = BlockPreprocessor(args.pca_dim, int(seed)).fit(
            values, inner_train
        )
        inner_train_blocks = inner_preprocessor.transform(values, inner_train)
        validation_blocks = inner_preprocessor.transform(values, validation)
        full_preprocessor = BlockPreprocessor(args.pca_dim, int(seed) + 50000).fit(
            values, outer_train
        )
        outer_train_blocks = full_preprocessor.transform(values, outer_train)
        heldout_blocks = full_preprocessor.transform(values, heldout)
        seed_result = {
            "seed": int(seed),
            "n_train": int(len(outer_train)),
            "n_heldout": int(len(heldout)),
            "events_train": int(event[outer_train].sum()),
            "events_heldout": int(event[heldout].sum()),
            "cindex_convex": float(
                convex_metrics.loc[
                    convex_metrics["seed"] == int(seed),
                    "cindex_late_tuned_simplex",
                ].iloc[0]
            ),
        }
        heldout_risks = {}
        for architecture in architectures:
            # Reuse the same random stream for shared modality projections;
            # architecture-specific layers naturally consume only the extra
            # draws they require.
            model_seed = int(seed) * 1000 + 101
            epochs, validation_ci, _ = select_epoch(
                architecture,
                inner_preprocessor.output_dims,
                inner_train_blocks,
                time[inner_train],
                event[inner_train],
                validation_blocks,
                time[validation],
                event[validation],
                model_seed,
                args.d_model,
                args.num_heads,
                args.dropout,
                args.lr,
                args.weight_decay,
                args.max_epochs,
                args.patience,
            )
            risk, modality_weights, pairwise, n_parameters = refit_and_predict(
                architecture,
                full_preprocessor.output_dims,
                outer_train_blocks,
                time[outer_train],
                event[outer_train],
                heldout_blocks,
                epochs,
                model_seed,
                args.d_model,
                args.num_heads,
                args.dropout,
                args.lr,
                args.weight_decay,
            )
            heldout_risks[architecture] = risk
            score = cindex(time[heldout], event[heldout], risk)
            seed_result[f"cindex_{architecture}"] = score
            seed_result[f"{architecture}_validation_cindex"] = validation_ci
            seed_result[f"{architecture}_epochs"] = epochs
            seed_result[f"{architecture}_parameters"] = n_parameters
            for modality_index, modality in enumerate(MODALITIES):
                seed_result[f"{architecture}_mean_weight_{modality}"] = float(
                    modality_weights[:, modality_index].mean()
                )
            if pairwise is not None:
                # [patient, head, query modality, key modality]
                mean_pairwise = pairwise.mean(axis=(0, 1))
                for query_index, query in enumerate(MODALITIES):
                    for key_index, key in enumerate(MODALITIES):
                        attention_rows.append({
                            "seed": int(seed),
                            "query_modality": query,
                            "key_modality": key,
                            "mean_attention": float(mean_pairwise[query_index, key_index]),
                        })
            for local, cohort_index in enumerate(heldout):
                for modality_index, modality in enumerate(MODALITIES):
                    prediction_rows.append({
                        "seed": int(seed),
                        "case_id": ids[cohort_index],
                        "survival_days": float(time[cohort_index]),
                        "event": int(event[cohort_index]),
                        "architecture": architecture,
                        "risk": float(risk[local]),
                        "modality": modality,
                        "modality_weight": float(modality_weights[local, modality_index]),
                    })
        rows.append(seed_result)
        print(
            f"seed={seed} convex={seed_result['cindex_convex']:.3f} "
            f"concat={seed_result['cindex_concat']:.3f} "
            f"cross_attention={seed_result['cindex_cross_attention']:.3f}",
            flush=True,
        )

    per_seed = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    attention = pd.DataFrame(attention_rows)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions_and_weights.csv", index=False)
    attention.to_csv(args.output_dir / "pairwise_attention.csv", index=False)
    results = {}
    for name in ("convex", "concat", "cross_attention"):
        values_ci = per_seed[f"cindex_{name}"]
        results[name] = {
            "cindex_mean": float(values_ci.mean()),
            "cindex_std_across_seeds": float(values_ci.std(ddof=1)),
            "cindex_per_seed": values_ci.tolist(),
        }
    for candidate in ("concat", "cross_attention"):
        delta = per_seed[f"cindex_{candidate}"] - per_seed["cindex_convex"]
        results[f"{candidate}_minus_convex"] = {
            "mean": float(delta.mean()),
            "std_across_seeds": float(delta.std(ddof=1)),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "per_seed": delta.tolist(),
        }
    summary = {
        "status": "paired_three_fusion_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "vision_label": args.vision_label,
        "cohort": {"n_cases": len(ids), "n_events": int(event.sum())},
        "protocol": {
            "outer_split": "80/20 event-stratified holdout",
            "seeds": [int(seed) for seed in args.seeds],
            "pca_dim_fixed_train_only": args.pca_dim,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "epoch_selection": "inner validation, then refit on full outer train",
        },
        "results": results,
        "limitations": [
            "The current run is diagnostic until repeated with the 160-patient STU-Net export.",
            "Attention weights describe model allocation and are not causal feature importance.",
            "Repeated holdouts overlap and are not independent replicates.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
