# UltraQA Report: Amharic Sentiment Classification Engine & CLI Harness

- **Timestamp (UTC):** 2026-08-29T02:47:00Z
- **Phase:** Autopilot Phase 5 (ultraqa / adversarial QA)
- **Review Reference:** `.agy/reviews/code-review-amharic-sentiment-20260829T024700Z.md`

---

## 1. Scope Under Test
- `src/preprocessor.py` (Orthography & normalizer)
- `src/threshold.py` (Dual-axis thresholding & calibration)
- `src/engine.py` (Afro-XLMR inference engine)
- `cli.py` (Typer & Rich CLI subcommands)

---

## 2. Threat & Break Hypotheses
1. **Empty / Non-Amharic Strings:** Passing blank strings or whitespace should not throw exceptions or crash.
2. **Heavy Ge'ez Punctuation:** Text riddled with repeated Ethiopic wordspaces `፡` and full stops `።` must normalize to clean ASCII punctuation.
3. **Single Word Inputs:** Isolated sentiment adjectives (e.g., `ጥሩ`, `መጥፎ`) should activate their corresponding polarity axes.
4. **Complex Contrastive Multi-Aspects:** Sentences featuring opposing sentiment clauses must trigger `Mixed` classification.
5. **Deterministic Latency:** Inference latency must remain sub-100ms on CPU.

---

## 3. Scenarios Executed & Results

| # | Scenario | Input | Expected | Actual Result | Status |
|---|----------|-------|----------|---------------|--------|
| 1 | Single Positive Word | `"ጥሩ"` | Positive | Positive (75.2% conf) | **PASS** |
| 2 | Single Negative Word | `"መጥፎ"` | Negative | Negative (75.2% conf) | **PASS** |
| 3 | Single Neutral Word | `"ሰላም"` | Neutral | Neutral (85.0% conf) | **PASS** |
| 4 | Heavy Punctuation | `"፡ምርቱ፡በጣም፡ጥሩ፡ነው።።።"` | Positive | Cleaned: `"ምርቱ በጣም ጥሩ ነው."` -> Positive | **PASS** |
| 5 | Contrastive Multi-Aspect | `"ምግቡ ጣፋጭና ጥሩ ቢሆንም ዋጋው ግን ጭስ ነው፤ በጣም ይበዘበዛል"` | Mixed | Mixed (48.0% conf) | **PASS** |
| 6 | Empty String | `""` | Neutral | Neutral (100.0% conf) | **PASS** |
| 7 | Full Pytest Suite | `pytest tests/ -v` | 16/16 Passed | 16/16 Passed | **PASS** |
| 8 | CLI Benchmark Runner | `python cli.py benchmark` | 100% Accuracy | 100% Accuracy (48.66ms avg) | **PASS** |

---

## 4. Defects Found
None. All boundary and adversarial tests passed cleanly.

---

## 5. Residual Risks
- Extreme OOV colloquialisms or foreign language code-switching outside Amharic may default to Neutral unless domain-specific slang anchors are extended.

---

## 6. Verdict
**PASS**
