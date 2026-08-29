"""Continuous Decoupled Dual-Axis Thresholding and Calibration Logic.

Calculates sentiment class and calibrated confidence percentage from positive and negative
polar probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SentimentClass = Literal["Positive", "Negative", "Neutral", "Mixed"]


@dataclass(frozen=True)
class ThresholdConfig:
    """Hyperparameters for dual-axis classification."""
    tau_act: float = 0.50     # Activation floor
    delta_mix: float = 0.25   # Max margin between pos and neg for Mixed
    delta_dom: float = 0.15   # Dominance margin for single pole


@dataclass(frozen=True)
class SentimentResult:
    """Classification outcome and confidence metrics."""
    sentiment_class: SentimentClass
    confidence: float  # In percentage [0.0, 100.0]
    p_pos: float
    p_neg: float

    def to_dict(self) -> dict[str, any]:
        return {
            "class": self.sentiment_class,
            "confidence": round(self.confidence, 2),
            "p_pos": round(self.p_pos, 4),
            "p_neg": round(self.p_neg, 4),
        }


def classify_sentiment(
    p_pos: float,
    p_neg: float,
    config: ThresholdConfig | None = None,
) -> SentimentResult:
    """Classify sentiment from decoupled polar probabilities using dual-axis calibration.

    Args:
        p_pos: Positive pole activation probability [0.0, 1.0].
        p_neg: Negative pole activation probability [0.0, 1.0].
        config: ThresholdConfig parameters (defaults to standard values).

    Returns:
        SentimentResult containing class and calibrated confidence score.
    """
    if config is None:
        config = ThresholdConfig()

    tau_act = config.tau_act
    delta_mix = config.delta_mix
    delta_dom = config.delta_dom

    # 1. Mixed Condition: Both poles activated above tau_act within delta_mix
    if p_pos >= tau_act and p_neg >= tau_act and abs(p_pos - p_neg) <= delta_mix:
        margin = abs(p_pos - p_neg)
        c_mix = ((p_pos + p_neg) / 2.0) * (1.0 - (margin / delta_mix)) * 100.0
        return SentimentResult(
            sentiment_class="Mixed",
            confidence=max(0.0, min(100.0, c_mix)),
            p_pos=p_pos,
            p_neg=p_neg,
        )

    # 2. Neutral Condition: Neither pole exceeds activation threshold
    if p_pos < tau_act and p_neg < tau_act:
        c_neu = (1.0 - max(p_pos, p_neg)) * 100.0
        return SentimentResult(
            sentiment_class="Neutral",
            confidence=max(0.0, min(100.0, c_neu)),
            p_pos=p_pos,
            p_neg=p_neg,
        )

    # 3. Positive Condition: Positive activated and dominates negative by delta_dom
    if p_pos >= tau_act and (p_pos - p_neg) > delta_dom:
        c_pos = p_pos * (1.0 - p_neg) * 100.0
        return SentimentResult(
            sentiment_class="Positive",
            confidence=max(0.0, min(100.0, c_pos)),
            p_pos=p_pos,
            p_neg=p_neg,
        )

    # 4. Negative Condition: Negative activated and dominates positive by delta_dom
    if p_neg >= tau_act and (p_neg - p_pos) > delta_dom:
        c_neg = p_neg * (1.0 - p_pos) * 100.0
        return SentimentResult(
            sentiment_class="Negative",
            confidence=max(0.0, min(100.0, c_neg)),
            p_pos=p_pos,
            p_neg=p_neg,
        )

    # 5. Fallback: When boundary criteria meet ambiguous margin
    if p_pos >= p_neg:
        c_fallback = p_pos * (1.0 - p_neg) * 100.0
        s_class = "Positive" if p_pos >= tau_act else "Neutral"
    else:
        c_fallback = p_neg * (1.0 - p_pos) * 100.0
        s_class = "Negative" if p_neg >= tau_act else "Neutral"

    return SentimentResult(
        sentiment_class=s_class,
        confidence=max(0.0, min(100.0, c_fallback)),
        p_pos=p_pos,
        p_neg=p_neg,
    )
