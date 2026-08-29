# Specification: Amharic Sentiment Classification Engine & CLI Test Harness

- **Timestamp (UTC):** 2026-08-29T02:39:00Z
- **Phase:** Autopilot Phase 1 (deep-interview / requirements crystallization)
- **Directive Source:** `AUTOPILOT_INSTRUCTIONS.md`

---

## 1. Goal & Non-Goals

### Goal:
Build an end-to-end, native Amharic Sentiment Classification Engine and CLI test harness leveraging `Davlan/afro-xlmr-base` transformer representations and decoupled dual-axis thresholding math.

### Non-Goals:
- Mocking or stubbing inference outputs (real deterministic Afro-XLMR embeddings and classification).
- Web frontends (this directive focuses strictly on the Python inference engine and rich CLI harness).
- Cloud-only APIs (must execute locally on CPU/CUDA/MPS).

---

## 2. Constraints
- **Runtime:** Python >= 3.10
- **Libraries:** `torch`, `transformers`, `sentencepiece`, `accelerate`, `typer`, `rich`, `pytest`
- **Normalization:** Fast $O(N)$ C-level `str.translate` based orthographic normalization for Ge'ez homophones, labiovelars, punctuation, and elongation regex.
- **Decision Engine:** Continuous decoupled dual-axis thresholds ($\tau_{\text{act}} = 0.50$, $\delta_{\text{mix}} = 0.25$, $\delta_{\text{dom}} = 0.15$).

---

## 3. Acceptance Criteria
1. **Normalization Accuracy:** Orthographic normalization passes 100% of Ge'ez character mappings (Ha, Sa, Glottal, Tsa, Labiovelars, Punctuation, Elongation reduction).
2. **Thresholding Math:** Correctly produces `Positive`, `Negative`, `Neutral`, `Mixed` classes along with calculated confidence metrics and polar probabilities.
3. **Inference Latency & Output:** `SentimentInferenceEngine` returns dictionary with `class`, `confidence`, `p_pos`, `p_neg`, `cleaned_text`, `latency_ms`.
4. **CLI Suite:** `cli.py` provides `analyze <text>`, `repl`, and `benchmark` subcommands with colored Rich tables.
5. **Test Suite:** `pytest tests/` passes 100% on the 10 Golden Benchmark Cases and unit tests.

---

## 4. Open Questions Remaining
- None (requirements are fully specified in `AUTOPILOT_INSTRUCTIONS.md`).
- Ready for plan phase (`ralplan`).
