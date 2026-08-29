"""Amharic Sentiment Classification Engine package."""

from .preprocessor import AmharicPreprocessor
from .threshold import classify_sentiment, ThresholdConfig, SentimentResult

try:
    from .engine import SentimentInferenceEngine
    __all__ = [
        "AmharicPreprocessor",
        "classify_sentiment",
        "ThresholdConfig",
        "SentimentResult",
        "SentimentInferenceEngine",
    ]
except ImportError:
    __all__ = [
        "AmharicPreprocessor",
        "classify_sentiment",
        "ThresholdConfig",
        "SentimentResult",
    ]
