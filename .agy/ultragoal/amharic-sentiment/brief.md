# Ultragoal Brief: Amharic Sentiment Classification & CLI Harness

## Objective
Implement and verify all components specified in `AUTOPILOT_INSTRUCTIONS.md`:
1. `src/preprocessor.py`: $O(N)$ orthographic normalization for Amharic.
2. `src/threshold.py`: Decoupled dual-axis calibration mathematics.
3. `src/engine.py`: `Davlan/afro-xlmr-base` inference engine with latency instrumentation.
4. `cli.py`: Interactive and single-shot CLI with Typer and Rich.
5. `tests/`: 10 Golden benchmark test cases and unit tests for normalizer & engine.

## Subgoals
- **SG-1:** Create package structure and normalizer (`src/preprocessor.py`) + unit tests (`tests/test_preprocessor.py`).
- **SG-2:** Implement thresholding math (`src/threshold.py`) + unit tests.
- **SG-3:** Create benchmark dataset (`tests/benchmark_cases.json`) and inference engine (`src/engine.py`).
- **SG-4:** Implement CLI (`cli.py`) with `analyze`, `repl`, and `benchmark`.
- **SG-5:** Run complete test suite and CLI verification commands.
