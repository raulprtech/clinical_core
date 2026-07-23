"""Evaluate orthogonal alignment and hierarchical residual trimodal fusion.

The experiment is paired to an existing convex benchmark.  Every PCA,
standardizer, Procrustes rotation, model checkpoint decision and refit is
estimated without access to the outer held-out patients.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.evaluate_cross_attention_fusion import (  # noqa: E402
    BlockPreprocessor,
    cindex,
    load_convex_reference,
    refit_and_predict,
    select_epoch,
)
from tools.evaluate_trimodal_fusion import (  # noqa: E402
    DEFAULT_SEEDS,
    load_indexed_csv,
    load_text_npz,
    load_vision_csv,
)


MODALITIES = ("tabular", "text", "vision")
VARIANTS = {
    "orthogonal_concat": ("concat", True),
    "hierarchical": ("hierarchical", False),
    "orthogonal_hierarchical": ("hierarchical", True),
}


class OrthogonalVisionAlignment:
    """Train-only Procrustes rotation of vision toward clinical-text context."""

    def __init__(self, seed: int):
        self.seed = int(seed)

    def fit(self, blocks: list[torch.Tensor]) -> "OrthogonalVisionAlignment":
        tabular, text, vision = [block.numpy().astype(np.float64) for block in blocks]
        if text.shape[1] != vision.shape[1]:
            raise ValueError(
                "Text and vision must have equal PCA dimensions for orthogonal alignment"
            )
        dimension = vision.shape[1]
        if min(tabular.shape[0] - 1, tabular.shape[1]) < dimension:
            raise ValueError("Tabular block is too small for the requested context dimension")
        self.tabular_pca = PCA(
            n_components=dimension, svd_solver="full", random_state=self.seed
        ).fit(tabular)
        tabular_context = self.tabular_pca.transform(tabular)
        self.tabular_scaler = StandardScaler().fit(tabular_context)
        tabular_context = self.tabular_scaler.transform(tabular_context)
        context = 0.5 * (tabular_context + text)
        left, _, right_t = np.linalg.svd(vision.T @ context, full_matrices=False)
        self.rotation = left @ right_t
        identity = np.eye(dimension)
        self.orthogonality_error = float(
            np.linalg.norm(self.rotation.T @ self.rotation - identity, ord="fro")
        )
        return self

    def transform(self, blocks: list[torch.Tensor]) -> list[torch.Tensor]:
        aligned = [block.clone() for block in blocks]
        vision = aligned[2].numpy().astype(np.float64) @ self.rotation
        aligned[2] = torch.tensor(vision, dtype=torch.float32)
        return aligned

    def diagnostics(self, blocks: list[torch.Tensor]) -> dict[str, float]:
        tabular, text, vision = [block.numpy().astype(np.float64) for block in blocks]
        tabular_context = self.tabular_scaler.transform(
            self.tabular_pca.transform(tabular)
        )
        context = 0.5 * (tabular_context + text)
        aligned = vision @ self.rotation

        def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            return np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)

        return {
            "cosine_before": float(cosine(vision, context).mean()),
            "cosine_after": float(cosine(aligned, context).mean()),
            "orthogonality_error": self.orthogonality_error,
        }


def validate_optional_baselines(path: Path | None, seeds: list[int]) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path / "per_seed_metrics.csv")
    required = {"seed", "cindex_concat", "cindex_cross_attention"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks the expected three-fusion metrics")
    if set(frame["seed"].astype(int)) != set(map(int, seeds)):
        raise ValueError("Optional baseline seeds do not match this run")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--vision-embeddings", type=Path, required=True)
    parser.add_argument("--convex-results-dir", type=Path, required=True)
    parser.add_argument("--three-fusion-results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
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
    cohort = targets.loc[common].apply(pd.to_numeric, errors="coerce")
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
    old_baselines = validate_optional_baselines(args.three_fusion_results_dir, args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(args.output_dir / "cohort_common.csv", index=False)
    rows = []
    prediction_rows = []
    alignment_rows = []
    all_indices = np.arange(len(ids))

    for seed in args.seeds:
        outer_train, heldout = train_test_split(
            all_indices, test_size=0.20, stratify=event, random_state=int(seed)
        )
        expected_heldout = set(
            convex_predictions.loc[
                convex_predictions["seed"] == int(seed), "case_id"
            ].astype(str)
        )
        if expected_heldout != {ids[index] for index in heldout}:
            raise ValueError(f"Convex and candidate heldout IDs differ for seed {seed}")
        inner_local, validation_local = train_test_split(
            np.arange(len(outer_train)), test_size=0.20,
            stratify=event[outer_train], random_state=int(seed) * 100,
        )
        inner_train = outer_train[inner_local]
        validation = outer_train[validation_local]

        inner_preprocessor = BlockPreprocessor(args.pca_dim, int(seed)).fit(
            values, inner_train
        )
        inner_blocks = inner_preprocessor.transform(values, inner_train)
        validation_blocks = inner_preprocessor.transform(values, validation)
        inner_alignment = OrthogonalVisionAlignment(int(seed)).fit(inner_blocks)
        aligned_inner = inner_alignment.transform(inner_blocks)
        aligned_validation = inner_alignment.transform(validation_blocks)

        full_preprocessor = BlockPreprocessor(args.pca_dim, int(seed) + 50000).fit(
            values, outer_train
        )
        outer_blocks = full_preprocessor.transform(values, outer_train)
        heldout_blocks = full_preprocessor.transform(values, heldout)
        full_alignment = OrthogonalVisionAlignment(int(seed) + 50000).fit(outer_blocks)
        aligned_outer = full_alignment.transform(outer_blocks)
        aligned_heldout = full_alignment.transform(heldout_blocks)
        for partition, blocks in (("outer_train", outer_blocks), ("heldout", heldout_blocks)):
            alignment_rows.append({
                "seed": int(seed), "partition": partition,
                **full_alignment.diagnostics(blocks),
            })

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
        if old_baselines is not None:
            baseline = old_baselines.loc[old_baselines["seed"] == int(seed)].iloc[0]
            seed_result["cindex_concat"] = float(baseline["cindex_concat"])
            seed_result["cindex_cross_attention"] = float(
                baseline["cindex_cross_attention"]
            )

        for variant, (architecture, use_alignment) in VARIANTS.items():
            selection_train = aligned_inner if use_alignment else inner_blocks
            selection_validation = aligned_validation if use_alignment else validation_blocks
            refit_train = aligned_outer if use_alignment else outer_blocks
            refit_heldout = aligned_heldout if use_alignment else heldout_blocks
            model_seed = int(seed) * 1000 + 101
            epochs, validation_ci, _ = select_epoch(
                architecture,
                inner_preprocessor.output_dims,
                selection_train,
                time[inner_train],
                event[inner_train],
                selection_validation,
                time[validation],
                event[validation],
                model_seed,
                args.d_model,
                4,
                args.dropout,
                args.lr,
                args.weight_decay,
                args.max_epochs,
                args.patience,
            )
            risk, weights, _, parameters = refit_and_predict(
                architecture,
                full_preprocessor.output_dims,
                refit_train,
                time[outer_train],
                event[outer_train],
                refit_heldout,
                epochs,
                model_seed,
                args.d_model,
                4,
                args.dropout,
                args.lr,
                args.weight_decay,
            )
            seed_result[f"cindex_{variant}"] = cindex(time[heldout], event[heldout], risk)
            seed_result[f"{variant}_validation_cindex"] = validation_ci
            seed_result[f"{variant}_epochs"] = epochs
            seed_result[f"{variant}_parameters"] = parameters
            for index, modality in enumerate(MODALITIES):
                seed_result[f"{variant}_mean_weight_{modality}"] = float(
                    weights[:, index].mean()
                )
            prediction_rows.extend(
                {
                    "seed": int(seed),
                    "case_id": ids[cohort_index],
                    "survival_days": float(time[cohort_index]),
                    "event": int(event[cohort_index]),
                    "variant": variant,
                    "risk": float(risk[local]),
                    **{
                        f"weight_{modality}": float(weights[local, index])
                        for index, modality in enumerate(MODALITIES)
                    },
                }
                for local, cohort_index in enumerate(heldout)
            )

        rows.append(seed_result)
        print(
            f"seed={seed} convex={seed_result['cindex_convex']:.3f} "
            f"ortho_concat={seed_result['cindex_orthogonal_concat']:.3f} "
            f"hier={seed_result['cindex_hierarchical']:.3f} "
            f"ortho_hier={seed_result['cindex_orthogonal_hierarchical']:.3f}",
            flush=True,
        )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "heldout_predictions_and_weights.csv", index=False
    )
    pd.DataFrame(alignment_rows).to_csv(
        args.output_dir / "orthogonal_alignment_diagnostics.csv", index=False
    )
    names = ["convex", *(["concat", "cross_attention"] if old_baselines is not None else []), *VARIANTS]
    results = {}
    for name in names:
        values_ci = per_seed[f"cindex_{name}"]
        result = {
            "cindex_mean": float(values_ci.mean()),
            "cindex_std_across_seeds": float(values_ci.std(ddof=1)),
            "cindex_per_seed": values_ci.tolist(),
        }
        if name != "convex":
            delta = values_ci - per_seed["cindex_convex"]
            result["minus_convex_mean"] = float(delta.mean())
            result["wins_ties_losses_vs_convex"] = [
                int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())
            ]
        results[name] = result
    summary = {
        "status": "paired_hierarchical_orthogonal_fusion_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"n_cases": len(ids), "n_events": int(event.sum())},
        "protocol": {
            "outer_split": "80/20 event-stratified holdout",
            "seeds": [int(seed) for seed in args.seeds],
            "pca_dim_train_only": args.pca_dim,
            "orthogonal_fit": "Procrustes on outer-train only toward mean(tabular-PCA, text)",
            "hierarchy": "tabular anchor + gated text residual + gated vision residual",
            "epoch_selection": "inner validation, refit on complete outer train",
        },
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
