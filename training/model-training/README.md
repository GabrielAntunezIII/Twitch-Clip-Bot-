# model-training

Trains the clip-scoring model on features produced by `../feature-extraction/` and labels
derived from `../data-collection/` engagement metadata.

- Candidate models: XGBoost / LightGBM (tabular features, small-to-mid dataset size,
  fast inference for near-real-time scoring during live streams).
- Output: a serialized model artifact saved to the repo's top-level `models/` directory
  (outside `training/` — it's the interface the live bot eventually reads from).

Nothing implemented yet — placeholder pending feature-extraction pipeline.
