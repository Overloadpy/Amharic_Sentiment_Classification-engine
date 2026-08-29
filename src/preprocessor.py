"""Deterministic Amharic Orthographic Preprocessor and Normalizer.

Performs O(N) homophone unification, labiovelar reduction, punctuation normalization,
and elongation hygiene on Ge'ez / Amharic text.
"""

from __future__ import annotations

import re
import unicodedata


class AmharicPreprocessor:
    """O(N) Amharic text normalizer using C-level str.translate and regex hygiene."""

    # Homophone and orthographic character mapping table
    _MAPPING: dict[str, str] = {
        # 1. Ha series: ሐ/ኀ -> ሀ
        "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ", "ሗ": "ኋ",
        "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ", "ኇ": "ኋ",
        "ኋ": "ኋ", "ኌ": "ሄ", "ኍ": "ህ", "኎": "ህ",

        # 2. Sa series: ሠ -> ሰ
        "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ", "ሧ": "ሷ",

        # 3. Glottal series: ዐ -> አ
        "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",

        # 4. Tsa series: ፀ -> ጸ
        "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",

        # 5. Labiovelars
        "ቈ": "ቁ", "ቊ": "ቂ", "ቍ": "ቅ", "ቌ": "ቄ",
        "ኰ": "ኩ", "ኲ": "ኪ", "ኵ": "ክ", "ኴ": "ኬ",
        "ጐ": "ጉ", "ጒ": "ጊ", "ጕ": "ግ", "ጔ": "ጌ",
        "ዀ": "ሁ", "ዂ": "ሂ", "ዅ": "ህ", "ዄ": "ሄ",

        # 6. Ge'ez Punctuation to ASCII
        "፡": " ",  # Ethiopic wordspace -> space
        "።": ".",  # Ethiopic full stop -> period
        "፣": ",",  # Ethiopic comma -> comma
        "፤": ";",  # Ethiopic semicolon -> semicolon
        "፥": ":",  # Ethiopic colon -> colon
        "፦": ":",  # Ethiopic preface colon -> colon
        "፧": "?",  # Ethiopic question mark -> question mark
        "፨": "\n", # Ethiopic paragraph separator -> newline
        "፠": " ",  # Ethiopic section mark -> space
    }

    # Precompute translation table for O(N) C-level substitution
    _TRANSLATION_TABLE = str.maketrans(_MAPPING)

    # Regex patterns for elongation and whitespace hygiene
    _ELONGATION_PATTERN = re.compile(r"([^\d\s])\1{2,}")  # Collapse 3+ repeated characters to 1
    _PUNCTUATION_REPEAT_PATTERN = re.compile(r"([.!?,;:~])\1+")
    _WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")

    @classmethod
    def normalize(cls, text: str | None) -> str:
        """Normalize Amharic text with Unicode NFKC, homophone unification, and regex hygiene.

        Args:
            text: Input raw text string.

        Returns:
            Normalized and cleaned text.
        """
        if not text:
            return ""

        # 1. Unicode NFKC Normalization
        text = unicodedata.normalize("NFKC", text)

        # 2. O(N) Character Translation (Homophones, Labiovelars, Punctuation)
        text = text.translate(cls._TRANSLATION_TABLE)

        # 3. Collapse repeated character elongations (e.g., በጣምምምም -> በጣም)
        text = cls._ELONGATION_PATTERN.sub(r"\1", text)

        # 4. Normalize repeated punctuation (e.g., !!! -> !)
        text = cls._PUNCTUATION_REPEAT_PATTERN.sub(r"\1", text)

        # 5. Collapse inline whitespace and strip leading/trailing spaces
        text = cls._WHITESPACE_PATTERN.sub(" ", text).strip()

        return text
