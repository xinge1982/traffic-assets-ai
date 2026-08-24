#!/usr/bin/env python3
"""Train a camera-forward-depth regressor from sign_true_depth training data.

The target is ``true_camera_z`` in metres (+Z is camera forward).  Validation
is grouped by ``feature_id`` so observations of the same physical sign never
appear in both the training and validation portions of one fold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "raw_depth_median",
    "raw_depth_p10",
    "raw_depth_p90",
    "raw_depth_std",
    "depth_valid_ratio",
    "mask_area_ratio",
    "bbox_area_ratio",
    "center_u_normalized",
    "center_v_normalized",
    "confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and group-validate a model that predicts true camera Z "
            "from DA3, SAM2, bbox, and image-position features."
        )
    )
    parser.add_argument("training_csv", help="Path to training_data.csv")
    parser.add_argument(
        "--output-dir",
        default="sign_true_depth/model",
        help="Directory for model.joblib, metrics.json and OOF predictions",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "ridge", "extra_trees", "random_forest"],
        default="auto",
        help="Model to fit; auto selects by group-balanced OOF MAE",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-z", type=float, default=0.0)
    parser.add_argument("--max-z", type=float, default=math.inf)
    parser.add_argument(
        "--exclude-feature-id",
        action="append",
        default=[],
        help="Feature ID to exclude; may be repeated",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated model feature columns",
    )
    return parser.parse_args()


def make_models(seed: int) -> dict[str, object]:
    # Log-transforming the positive target prevents tree/linear predictions
    # from becoming negative and makes relative errors less distance-dependent.
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )
    models = {
        "ridge": TransformedTargetRegressor(
            regressor=ridge, func=np.log1p, inverse_func=np.expm1
        ),
        "extra_trees": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        ExtraTreesRegressor(
                            n_estimators=500,
                            min_samples_leaf=2,
                            max_features=0.8,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "random_forest": TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=500,
                            min_samples_leaf=2,
                            max_features=0.8,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }
    return models


def metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    error = np.abs(y_true - y_pred)
    per_sign = pd.DataFrame({"feature_id": groups, "abs_error": error}).groupby(
        "feature_id", sort=False
    )["abs_error"].mean()
    return {
        "observation_mae_m": float(mean_absolute_error(y_true, y_pred)),
        "observation_rmse_m": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "observation_median_abs_error_m": float(np.median(error)),
        "feature_balanced_mae_m": float(per_sign.mean()),
        "p95_abs_error_m": float(np.percentile(error, 95)),
    }


def load_and_filter(args: argparse.Namespace, features: list[str]) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(args.training_csv, dtype={"feature_id": "string"})
    required = ["feature_id", "true_camera_z", *features]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    data = source.copy()
    for column in ["true_camera_z", *features]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    keep = data["feature_id"].notna() & np.isfinite(data["true_camera_z"])
    keep &= data["true_camera_z"].between(args.min_z, args.max_z, inclusive="both")
    if "status" in data.columns:
        keep &= data["status"].astype(str).str.lower().isin(["ok", "success"])
    if args.exclude_feature_id:
        excluded = {str(value) for value in args.exclude_feature_id}
        keep &= ~data["feature_id"].astype(str).isin(excluded)

    data = data.loc[keep].reset_index(drop=False).rename(columns={"index": "source_row"})
    data = data.replace([np.inf, -np.inf], np.nan)
    # Median imputation can handle missing feature values, but a row with no
    # usable input at all carries no information and is rejected.
    data = data.loc[data[features].notna().any(axis=1)].reset_index(drop=True)

    summary = {
        "source_rows": int(len(source)),
        "retained_rows": int(len(data)),
        "rejected_rows": int(len(source) - len(data)),
        "unique_feature_ids": int(data["feature_id"].nunique()),
    }
    if len(data) < 10:
        raise ValueError(f"Only {len(data)} usable rows remain; need at least 10")
    if summary["unique_feature_ids"] < 3:
        raise ValueError("Need at least 3 distinct feature_id groups")
    return data, summary


def cross_validate(
    name: str,
    estimator: object,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splitter: GroupKFold,
) -> tuple[dict, np.ndarray, np.ndarray]:
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups)):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        predictions[valid_idx] = np.maximum(model.predict(X[valid_idx]), 0.001)
        fold_ids[valid_idx] = fold
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{name} did not produce a prediction for every row")
    return metrics(y, predictions, groups), predictions, fold_ids


def scale_baseline_oof(
    raw_depth: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splitter: GroupKFold,
) -> tuple[dict, np.ndarray]:
    predictions = np.full(len(y), np.nan, dtype=float)
    dummy_X = raw_depth.reshape(-1, 1)
    for train_idx, valid_idx in splitter.split(dummy_X, y, groups):
        valid_scale = y[train_idx] / raw_depth[train_idx]
        scale = float(np.median(valid_scale[np.isfinite(valid_scale)]))
        predictions[valid_idx] = np.maximum(raw_depth[valid_idx] * scale, 0.001)
    return metrics(y, predictions, groups), predictions


def main() -> None:
    args = parse_args()
    features = [item.strip() for item in args.features.split(",") if item.strip()]
    if not features:
        raise ValueError("--features must contain at least one column")

    data, data_summary = load_and_filter(args, features)
    X = data[features].to_numpy(dtype=float)
    y = data["true_camera_z"].to_numpy(dtype=float)
    groups = data["feature_id"].astype(str).to_numpy()

    n_splits = min(args.n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Group cross-validation needs at least 2 folds")
    splitter = GroupKFold(n_splits=n_splits)
    models = make_models(args.seed)
    selected_names = list(models) if args.model == "auto" else [args.model]

    results: dict[str, dict] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    fold_ids = None
    for name in selected_names:
        result, predictions, model_fold_ids = cross_validate(
            name, models[name], X, y, groups, splitter
        )
        results[name] = result
        predictions_by_model[name] = predictions
        if fold_ids is None:
            fold_ids = model_fold_ids

    raw_depth = data["raw_depth_median"].to_numpy(dtype=float)
    baseline_metrics, baseline_predictions = scale_baseline_oof(
        raw_depth, y, groups, splitter
    )
    results["median_scale_baseline"] = baseline_metrics

    best_name = min(
        selected_names, key=lambda name: results[name]["feature_balanced_mae_m"]
    )
    final_model = clone(models[best_name])
    final_model.fit(X, y)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "true_camera_z_model.joblib"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "oof_predictions.csv"

    artifact = {
        "model": final_model,
        "model_name": best_name,
        "features": features,
        "target": "true_camera_z",
        "target_unit": "m",
        "coordinate_contract": "+Z is camera forward",
        "training_rows": int(len(data)),
        "training_feature_ids": int(len(np.unique(groups))),
        "minimum_prediction_m": 0.001,
    }
    joblib.dump(artifact, model_path)

    report = {
        "selected_model": best_name,
        "selection_metric": "feature_balanced_mae_m",
        "data": data_summary,
        "features": features,
        "target": {"column": "true_camera_z", "unit": "m", "axis": "camera +Z forward"},
        "cross_validation": {
            "type": "GroupKFold",
            "group_column": "feature_id",
            "n_splits": n_splits,
            "note": "OOF metrics are model-selection estimates, not a final independent test.",
        },
        "models": results,
        "warnings": [],
    }
    if len(data) < 100 or len(np.unique(groups)) < 30:
        report["warnings"].append(
            "Small dataset: use this run to verify the pipeline, not as final deployment evidence."
        )
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    output = data[["source_row", "feature_id", "img", "true_camera_z"]].copy()
    output["fold"] = fold_ids
    for name, predictions in predictions_by_model.items():
        output[f"pred_{name}_z_m"] = predictions
        output[f"abs_error_{name}_m"] = np.abs(y - predictions)
    output["pred_median_scale_baseline_z_m"] = baseline_predictions
    output["abs_error_median_scale_baseline_m"] = np.abs(y - baseline_predictions)
    output.to_csv(predictions_path, index=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"OOF predictions: {predictions_path}")


if __name__ == "__main__":
    main()
