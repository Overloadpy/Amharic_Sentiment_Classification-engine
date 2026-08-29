"""Unit tests for thresholding and dual-axis calibration."""

import pytest
from src.threshold import classify_sentiment, ThresholdConfig


def test_positive_dominance():
    # p_pos = 0.90, p_neg = 0.10 -> Positive
    # c_pos = 0.90 * (1 - 0.10) * 100 = 81.0%
    res = classify_sentiment(p_pos=0.90, p_neg=0.10)
    assert res.sentiment_class == "Positive"
    assert round(res.confidence, 2) == 81.0
    assert res.p_pos == 0.90
    assert res.p_neg == 0.10


def test_negative_dominance():
    # p_pos = 0.05, p_neg = 0.85 -> Negative
    # c_neg = 0.85 * (1 - 0.05) * 100 = 80.75%
    res = classify_sentiment(p_pos=0.05, p_neg=0.85)
    assert res.sentiment_class == "Negative"
    assert round(res.confidence, 2) == 80.75


def test_neutral_condition():
    # Both below tau_act = 0.50
    # p_pos = 0.30, p_neg = 0.20 -> Neutral
    # c_neu = (1.0 - max(0.30, 0.20)) * 100 = 70.0%
    res = classify_sentiment(p_pos=0.30, p_neg=0.20)
    assert res.sentiment_class == "Neutral"
    assert round(res.confidence, 2) == 70.0


def test_mixed_condition():
    # Both >= 0.50 and |0.80 - 0.70| = 0.10 <= delta_mix (0.25)
    # c_mix = ((0.80 + 0.70) / 2) * (1 - (0.10 / 0.25)) * 100
    #       = 0.75 * (1 - 0.40) * 100 = 0.75 * 0.60 * 100 = 45.0%
    res = classify_sentiment(p_pos=0.80, p_neg=0.70)
    assert res.sentiment_class == "Mixed"
    assert round(res.confidence, 2) == 45.0


def test_custom_config():
    custom = ThresholdConfig(tau_act=0.60, delta_mix=0.20, delta_dom=0.20)
    # p_pos = 0.55, p_neg = 0.10 -> under tau_act (0.60), so Neutral
    res = classify_sentiment(p_pos=0.55, p_neg=0.10, config=custom)
    assert res.sentiment_class == "Neutral"
