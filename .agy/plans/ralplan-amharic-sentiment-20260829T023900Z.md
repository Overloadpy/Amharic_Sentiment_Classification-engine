# Ralplan: Amharic Sentiment Classification Engine & CLI Test Harness

- **Timestamp (UTC):** 2026-08-29T02:39:00Z
- **Phase:** Autopilot Phase 2 (ralplan / consensus planning)
- **Spec:** `.agy/specs/deep-interview-amharic-sentiment-20260829T023900Z.md`

---

## 1. Summary
Implement `src/preprocessor.py`, `src/threshold.py`, `src/engine.py`, `cli.py`, and `tests/` (`test_preprocessor.py`, `test_engine.py`, `benchmark_cases.json`) to deliver an Afro-XLMR-powered Amharic Sentiment Classification engine with decoupled dual-axis calibration and a Rich CLI.

---

## 2. Ordered Tasks & Dependencies

1. **Task 1: Project Environment & Structure**
   - Create `src/` and `tests/` packages.
   - Verify `requirements.txt` dependencies.

2. **Task 2: Deterministic Normalizer (`src/preprocessor.py`)**
   - Implement `AmharicPreprocessor` with `str.translate` for $O(N)$ speed.
   - Unify Ha series, Sa series, Glottal series, Tsa series, and Labiovelar reductions.
   - Map Ge'ez punctuation to ASCII equivalents.
   - Apply character elongation reduction (`(.)\1{2,} -> \1`) and whitespace normalization.

3. **Task 3: Dual-Axis Thresholding & Calibration (`src/threshold.py`)**
   - Implement `classify_sentiment(p_pos, p_neg, tau_act=0.50, delta_mix=0.25, delta_dom=0.15)`.
   - Calculate exact confidence scores $C_{\text{mix}}$, $C_{\text{neu}}$, $C_{\text{pos}}$, $C_{\text{neg}}$, and fallback.

4. **Task 4: Transformer Inference Engine (`src/engine.py`)**
   - Implement `SentimentInferenceEngine` wrapping `Davlan/afro-xlmr-base`.
   - Device autodetection (CUDA -> MPS -> CPU).
   - Lazy loading with `load()`.
   - Compute polar probabilities using contextual representation projections and anchor embeddings/cloze scoring.
   - Return clean dictionary structure with timing.

5. **Task 5: Benchmark Data & Test Suites (`tests/`)**
   - Create `tests/benchmark_cases.json` with the 10 Golden Benchmark Cases.
   - Create `tests/test_preprocessor.py` testing homophones, labiovelars, punctuation, and elongation.
   - Create `tests/test_engine.py` testing engine loading, thresholding edge cases, and golden test cases.

6. **Task 6: Interactive & Single-Shot CLI (`cli.py`)**
   - Implement `typer` CLI with `analyze`, `repl`, and `benchmark` commands.
   - Format outputs using `rich.table.Table` and `rich.console.Console`.

7. **Task 7: Verification & Acceptance**
   - Run `pytest tests/`.
   - Run CLI benchmark and analyze commands.

---

## 3. Risks & Mitigations
- **Risk:** Large model download failure or slow initial inference.
  - **Mitigation:** Background pre-download weights, lazy loading, and caching.
- **Risk:** Amharic character encoding mismatches.
  - **Mitigation:** Explicit Unicode NFKC normalization prior to character translation.

---

## 4. Steelman Alternative
- *Alternative:* Use basic dictionary lookup / rule-based sentiment lexicon without transformer representation.
- *Rejection Reason:* Lexicons fail on complex syntax, negation, sarcasm, and subtle context (e.g. mixed reviews where both positive and negative attributes appear). Afro-XLMR provides contextual semantic representations.

---

## 5. Critic Pass
- **Review:** Are all mathematical formulas in `threshold.py` matching the specification in `AUTOPILOT_INSTRUCTIONS.md`? Yes.
- **Review:** Does the preprocessor handle all 4 homophone families and 4 labiovelar families? Yes.
- **Review:** Are all 10 Golden benchmark sentences accounted for? Yes.
- **Verdict:** **APPROVE** (Proceed to `ultragoal`).
