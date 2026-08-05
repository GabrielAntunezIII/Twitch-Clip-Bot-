"""
Trains an XGBoost clip-scoring model on features from `training/feature-extraction/` and
engagement-derived labels from `training/data-collection/`.

Joins training/feature-extraction/dataset/features.json (audio spikes, scene changes,
transcript stats, sentiment) with training/data-collection/dataset/metadata.json
(views, likes, subscriber_count, published_at) on video_id. Since there's no
ground-truth "clip-worthy" label, one is derived from engagement ranked
WITHIN each channel:

    views_per_day = view_count / days_since_published (floor 1 day)
    engagement = log1p(views_per_day) + 0.5 * log1p(like_count)
    percentile = engagement.rank(pct=True) computed per channel_id,
                 using ALL of that channel's videos in metadata.json
                 (not just the ones sampled for feature-extraction)

Two confounds got fixed here, in order of discovery:

1. Cross-channel engagement (e.g. normalizing by subscriber_count) mostly
   encoded channel identity rather than clip quality — a repost/compilation
   channel with 17M subs can have a structurally lower views-per-sub ceiling
   than the original creator's own channel, which has nothing to do with any
   individual clip being better or worse. Fixed by ranking each video only
   against other videos from the *same* channel.
2. Within a channel, publish dates in this dataset span 2021-2026, so raw
   view_count also conflates "good clip" with "old clip that had years to
   accumulate views." Fixed by dividing by days_since_published before
   ranking.

Videos are then labeled 1/0 by a percentile split (median by default)
within their own channel.

With few labeled rows (the common case until feature-extraction has been run
at scale), a single train/test split is too noisy to trust, so below
`--cv-min-rows` the script instead reports stratified k-fold cross-validated
metrics and fits the final model on all available data.

Usage:
  python training/model-training/train.py
  python training/model-training/train.py --label-percentile 0.67 --n-estimators 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

TRAINING_ROOT = Path(__file__).resolve().parent.parent  # training/
REPO_ROOT = TRAINING_ROOT.parent                          # true repo root
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

METADATA_PATH = TRAINING_ROOT / "data-collection" / "dataset" / "metadata.json"
FEATURES_PATH = TRAINING_ROOT / "feature-extraction" / "dataset" / "features.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"

FEATURE_COLUMNS = [
    "duration_seconds",
    "audio_spike_count",
    "scene_change_count",
    "transcript_word_count",
    "transcript_keyword_score",
    "sentiment_compound",
    "sentiment_pos",
    "sentiment_neg",
]


def _channel_percentiles(metadata: pd.DataFrame) -> pd.DataFrame:
    """Engagement percentile of each video within its own channel, computed
    against ALL of that channel's videos in metadata (not just the ones
    sampled for feature-extraction) so the ranking isn't skewed by a small
    sample of one channel's output.

    Normalizes views by days_since_published first — without this, videos
    published years apart within the same channel aren't comparable, since
    older ones have had far longer to accumulate views regardless of
    quality.
    """
    published_at = pd.to_datetime(metadata["published_at"], utc=True)
    days_since_published = (pd.Timestamp.now(tz="UTC") - published_at).dt.total_seconds() / 86400
    views_per_day = metadata["view_count"] / days_since_published.clip(lower=1)

    engagement = np.log1p(views_per_day) + 0.5 * np.log1p(metadata["like_count"])
    percentile = engagement.groupby(metadata["channel_id"]).rank(pct=True)
    return metadata[["video_id"]].assign(engagement_percentile=percentile)


def _load_dataset(metadata_path: Path, features_path: Path) -> pd.DataFrame:
    metadata = pd.DataFrame(json.loads(metadata_path.read_text(encoding="utf-8")))
    features = pd.DataFrame(json.loads(features_path.read_text(encoding="utf-8")))
    percentiles = _channel_percentiles(metadata)
    merged = features.merge(percentiles, on="video_id", how="inner")
    dropped = len(features) - len(merged)
    if dropped:
        logger.warning("%d feature rows had no matching metadata row — dropped", dropped)
    return merged


def _label_engagement(df: pd.DataFrame, percentile: float) -> pd.Series:
    return (df["engagement_percentile"] >= percentile).astype(int)


def _cross_validated_metrics(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict, folds: int
) -> dict:
    n_splits = min(folds, y.value_counts().min(), groups.nunique())
    if n_splits < 2:
        logger.warning("Not enough class/group balance for cross-validation — skipping metrics")
        return {}

    grouped = groups.nunique() >= n_splits * 2
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_args = (X, y, groups)
        method = f"{n_splits}-fold grouped cross-validation (grouped by source video)"
    else:
        logger.warning(
            "Only %d distinct source videos — too few to hold whole videos out per fold; "
            "falling back to ungrouped cross-validation (metrics may be optimistic)",
            groups.nunique(),
        )
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_args = (X, y)
        method = f"{n_splits}-fold cross-validation (ungrouped)"

    accuracies, aucs = [], []
    for train_idx, test_idx in splitter.split(*split_args):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        accuracies.append(accuracy_score(y.iloc[test_idx], preds))
        if y.iloc[test_idx].nunique() > 1:
            probs = model.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[test_idx], probs))

    return {
        "method": method,
        "accuracy_mean": float(np.mean(accuracies)),
        "auc_mean": float(np.mean(aucs)) if aucs else None,
    }


def _holdout_metrics(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict, test_size: float, seed: int
) -> dict:
    if groups.nunique() >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        method = "holdout (grouped by source video)"
    else:
        logger.warning(
            "Only %d distinct source videos — too few to hold whole videos out; "
            "falling back to ungrouped holdout (metrics may be optimistic)",
            groups.nunique(),
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        method = "holdout (ungrouped)"

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if y_test.nunique() > 1 else None
    return {
        "method": method,
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(auc) if auc is not None else None,
    }


def _load_explicit_dataset(features_path: Path) -> pd.DataFrame:
    """Loads a features file that already carries a `label` column (e.g. from
    training/feature-extraction/extract_window_features.py, where the label comes from
    training/alignment/build_windows.py's clip<->source-video cross-referencing rather
    than an engagement proxy). No metadata join needed."""
    df = pd.DataFrame(json.loads(features_path.read_text(encoding="utf-8")))
    if "label" not in df.columns:
        raise ValueError(f"{features_path} has no 'label' column — expected explicit per-row labels")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-path", default=str(METADATA_PATH))
    parser.add_argument("--features-path", default=str(FEATURES_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--label-source", choices=["engagement", "explicit"], default="engagement",
        help="'engagement' derives a label from views/likes percentile (needs --metadata-path). "
             "'explicit' reads a 'label' column already present in --features-path (e.g. windows "
             "built by alignment/build_windows.py) and skips the metadata join entirely.",
    )
    parser.add_argument("--label-percentile", type=float, default=0.5, help="Engagement quantile above which a video is labeled clip-worthy (--label-source engagement only)")
    parser.add_argument("--cv-min-rows", type=int, default=100, help="Below this many rows, use cross-validation instead of a holdout split")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features_path = Path(args.features_path)

    if args.label_source == "explicit":
        if not features_path.exists():
            logger.error("Missing features file %s — run training/feature-extraction/extract_window_features.py first", features_path)
            sys.exit(1)
        df = _load_explicit_dataset(features_path)
    else:
        metadata_path = Path(args.metadata_path)
        if not metadata_path.exists() or not features_path.exists():
            logger.error(
                "Missing input data (metadata=%s exists=%s, features=%s exists=%s) — "
                "run training/data-collection/youtube_scraper.py and training/feature-extraction/extract_features.py first",
                metadata_path, metadata_path.exists(), features_path, features_path.exists(),
            )
            sys.exit(1)
        df = _load_dataset(metadata_path, features_path)
        df["label"] = _label_engagement(df, args.label_percentile)

    if len(df) < 10:
        logger.error("Only %d rows available — need more before training means anything.", len(df))
        sys.exit(1)

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["channel_id"]
    logger.info(
        "Dataset: %d rows across %d source videos, label distribution: %s",
        len(df), groups.nunique(), y.value_counts().to_dict(),
    )

    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        eval_metric="logloss",
        random_state=args.seed,
    )

    if len(df) < args.cv_min_rows:
        logger.info("%d rows < --cv-min-rows=%d, using cross-validation instead of a holdout split", len(df), args.cv_min_rows)
        metrics = _cross_validated_metrics(X, y, groups, params, args.cv_folds)
    else:
        metrics = _holdout_metrics(X, y, groups, params, args.test_size, args.seed)

    if metrics:
        logger.info("Eval metrics (%s): %s", metrics["method"], metrics)
    else:
        logger.warning("No eval metrics computed — dataset too small or too imbalanced")

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "clip_scorer.json"
    final_model.save_model(str(model_path))

    metadata_out = {
        "feature_columns": FEATURE_COLUMNS,
        "label_source": args.label_source,
        "label_percentile": args.label_percentile if args.label_source == "engagement" else None,
        "n_training_rows": len(df),
        "eval_metrics": metrics,
        "xgboost_params": params,
    }
    (output_dir / "clip_scorer.meta.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    logger.info("Saved model to %s and metadata to %s", model_path, output_dir / "clip_scorer.meta.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
