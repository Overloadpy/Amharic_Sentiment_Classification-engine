"""Unit tests for AmharicPreprocessor."""

import pytest
from src.preprocessor import AmharicPreprocessor


def test_empty_and_none():
    assert AmharicPreprocessor.normalize("") == ""
    assert AmharicPreprocessor.normalize(None) == ""


def test_ha_series_unification():
    # ሐ/ኀ series -> ሀ series
    # ሐ, ሑ, ሒ, ሓ, ሔ, ሕ, ሖ -> ሀ, ሁ, ሂ, ሃ, ሄ, ህ, ሆ
    ha1 = "ሐሑሒሓሔሕሖ"
    expected1 = "ሀሁሂሃሄህሆ"
    assert AmharicPreprocessor.normalize(ha1) == expected1

    # ኀ, ኁ, ኂ, ኃ, ኄ, ኅ, ኆ -> ሀ, ሁ, ሂ, ሃ, ሄ, ህ, ሆ
    ha2 = "ኀኁኂኃኄኅኆ"
    assert AmharicPreprocessor.normalize(ha2) == expected1


def test_sa_series_unification():
    # ሠ, ሡ, ሢ, ሣ, ሤ, ሥ, ሦ -> ሰ, ሱ, ሲ, ሳ, ሴ, ስ, ሶ
    sa = "ሠሡሢሣሤሥሦ"
    expected = "ሰሱሲሳሴስሶ"
    assert AmharicPreprocessor.normalize(sa) == expected


def test_glottal_series_unification():
    # ዐ, ዑ, ዒ, ዓ, ዔ, ዕ, ዖ -> አ, ኡ, ኢ, ኣ, ኤ, እ, ኦ
    glottal = "ዐዑዒዓዔዕዖ"
    expected = "አኡኢኣኤእኦ"
    assert AmharicPreprocessor.normalize(glottal) == expected


def test_tsa_series_unification():
    # ፀ, ፁ, ፂ, ፃ, ፄ, ፅ, ፆ -> ጸ, ጹ, ጺ, ጻ, ጼ, ጽ, ጾ
    tsa = "ፀፁፂፃፄፅፆ"
    expected = "ጸጹጺጻጼጽጾ"
    assert AmharicPreprocessor.normalize(tsa) == expected


def test_labiovelar_reduction():
    # ቈ, ቊ, ቍ, ቌ -> ቁ, ቂ, ቅ, ቄ
    assert AmharicPreprocessor.normalize("ቈቊቍቌ") == "ቁቂቅቄ"
    # ኰ, ኲ, ኵ, ኴ -> ኩ, ኪ, ክ, ኬ
    assert AmharicPreprocessor.normalize("ኰኲኵኴ") == "ኩኪክኬ"
    # ጐ, ጒ, ጕ, ጔ -> ጉ, ጊ, ግ, ጌ
    assert AmharicPreprocessor.normalize("ጐጒጕጔ") == "ጉጊግጌ"
    # ዀ, ዂ, ዅ, ዄ -> ሁ, ሂ, ህ, ሄ
    assert AmharicPreprocessor.normalize("ዀዂዅዄ") == "ሁሂህሄ"


def test_punctuation_normalization():
    raw = "አገልግሎቱ፡በጣም፡ፈጣንና፡አስተማማኝ፡ነው።"
    normalized = AmharicPreprocessor.normalize(raw)
    assert normalized == "አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው."

    raw_punct = "ምርቱ፣ጥራቱ፤የላቀ፦ነው፧"
    assert AmharicPreprocessor.normalize(raw_punct) == "ምርቱ,ጥራቱ;የላቀ:ነው?"


def test_elongation_and_hygiene():
    elongated = "በጣምምምም አሪፍፍፍፍፍፍፍፍ!!!"
    normalized = AmharicPreprocessor.normalize(elongated)
    assert normalized == "በጣም አሪፍ!"

    spaces = "  ሰላም    ዓለም   "
    assert AmharicPreprocessor.normalize(spaces) == "ሰላም ኣለም"
