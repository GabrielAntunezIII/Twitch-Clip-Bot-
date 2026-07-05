# models

Saved trained model artifacts (e.g. `.json`/`.txt` XGBoost/LightGBM dumps) produced by
`model-training/`.

Intended to eventually be loaded from `bot/ai/validator.py` to replace or augment the
current LLM-only chat-based judgment with a trained score.

Large binary artifacts here should be gitignored or tracked via Git LFS once real models
land — placeholder for now, nothing checked in yet.
