"""Request-scoped keys for fitted numerical transformation reuse."""

from typing import Any

import joblib

FEATURE_ENGINE_VERSION = "0.3.0"


def preprocessing_signature(preprocessor: Any) -> str:
    """Hash ordered columns, imputation/scaling state, indicators and training bounds."""
    return joblib.hash(preprocessor)


def transformation_key(bundle: Any, signature: str) -> tuple[str, str]:
    return (bundle.metadata["feature_engine_version"], signature)
