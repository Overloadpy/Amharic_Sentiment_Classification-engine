# AUTOPILOT DIRECTIVE: AMHARIC SENTIMENT CLASSIFICATION & CLI HARNESS

## EXECUTION MODE: AUTONOMOUS & DETERMINISTIC (NO MOCKS / NO HARDCODED LOGIC)

You are an expert Machine Learning Systems Engineer and Python CLI Developer. Your mission is to build, calibrate, test, and verify the **Amharic Sentiment Classification Engine & CLI Test Harness** end-to-end based on native Afro-centric transformer representations (`Davlan/afro-xlmr-base`) and decoupled dual-axis thresholding.

DO NOT stop until all modules are implemented, the environment is configured, dependencies are resolved, and the test suite/CLI runs successfully with clean verification output.

---

## 1. ARCHITECTURAL BLUEPRINT & STACK SPECIFICATION

### Core Tech Stack:
- **Language / Runtime:** Python >= 3.10
- **Deep Learning / NLP:** `torch`, `transformers`, `sentencepiece`, `accelerate`
- **CLI & Terminal UI:** `typer`, `rich`
- **Testing & Verification:** `pytest`

### Directory Layout to Create:
```
amharic_sentiment/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessor.py        # O(N) Unicode & Homophone Normalizer
│   ├── engine.py              # Afro-XLMR Dual-Axis Inference Engine
│   └── threshold.py           # Multi-Polarity Thresholding & Calibration
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py   # Unit tests for orthographic normalization
│   ├── test_engine.py         # End-to-end benchmark test suite
│   └── benchmark_cases.json   # 10 Curated Golden Benchmark Cases
└── cli.py                     # Typer + Rich Interactive & Single-Shot CLI
```

---

## 2. STEP-BY-STEP IMPLEMENTATION INSTRUCTIONS

### Step 1: Dependencies (`requirements.txt`)
Declare the required packages:
```text
torch>=2.0.0
transformers>=4.35.0
sentencepiece>=0.1.99
accelerate>=0.24.0
typer[all]>=0.9.0
rich>=13.0.0
pytest>=7.0.0
```
*Note on Downloads: Check if `torch` and `transformers` are installed in the active Python environment. If not, prompt the user or run `pip install -r requirements.txt`. Weights for `Davlan/afro-xlmr-base` will download automatically on first run via Hugging Face Hub (approx. 1.1 GB cache).*

---

### Step 2: Deterministic Preprocessor (`src/preprocessor.py`)
Implement `AmharicPreprocessor` using Python C-level `str.translate` table for $O(N)$ speed:
1. **Unicode NFKC normalization**.
2. **Homophone Unification:**
   - Ha series: `ሐ/ኀ` $\to$ `ሀ` (`ሐ, ሑ, ሒ, ሓ, ሔ, ሕ, ሖ` and `ኀ, ኁ, ኂ, ኃ, ኄ, ኅ, ኆ` $\to$ `ሀ, ሁ, ሂ, ሃ, ሄ, ህ, Host`)
   - Sa series: `ሠ` $\to$ `ሰ` (`ሠ, ሡ, ሢ, ሣ, ሤ, ሥ, ሦ` $\to$ `ሰ, ሱ, ሲ, ሳ, ሴ, ስ, ሶ`)
   - Glottal series: `ዐ` $\to$ `አ` (`ዐ, ዑ, ዒ, ዓ, ዔ, ዕ, ዖ` $\to$ `አ, ኡ, ኢ, ኣ, ኤ, እ, ኦ`)
   - Tsa series: `ፀ` $\to$ `ጸ` (`ፀ, ፁ, ፂ, ፃ, ፄ, ፅ, ፆ` $\to$ `ጸ, ጹ, ጺ, ጻ, ጼ, ጽ, ጾ`)
3. **Labiovelars:** `ቈ, ቊ, ቍ, ቌ` $\to$ `ቁ, ቂ, ቅ, ቄ`; `ኰ, ኲ, ኵ, ኴ` $\to$ `ኩ, ኪ, ክ, ኬ`; `ጐ, ጒ, ጕ, ጔ` $\to$ `ጉ, ጊ, ግ, ጌ`; `ዀ, ዂ, ዅ, ዄ` $\to$ `ሁ, ሂ, ህ, ሄ`.
4. **Punctuation:** `፡` $\to$ `' '`, `።` $\to$ `.`, `፣` $\to$ `,`, `፤` $\to$ `;`, `፥/፦` $\to$ `:`, `፧` $\to$ `?`, `፨` $\to$ `\n`.
5. **Hygiene Regex:**
   - Collapse repeated character elongations (e.g., `በጣምምምም` $\to$ `በጣም`).
   - Normalize repeated punctuation and collapse whitespace.

---

### Step 3: Thresholding & Calibration Math (`src/threshold.py`)
Implement the continuous decoupled dual-axis decision logic:
- Parameters:
  - $\tau_{\text{act}} = 0.50$ (Activation floor)
  - $\delta_{\text{mix}} = 0.25$ (Max margin for Mixed classification)
  - $\delta_{\text{dom}} = 0.15$ (Dominance margin for single pole)

- **Rules:**
  1. **Mixed:** $p_{\text{pos}} \ge \tau_{\text{act}} \land p_{\text{neg}} \ge \tau_{\text{act}} \land |p_{\text{pos}} - p_{\text{neg}}| \le \delta_{\text{mix}}$
     $$C_{\text{mix}} = \left(\frac{p_{\text{pos}} + p_{\text{neg}}}{2}\right) \cdot \left(1.0 - \frac{|p_{\text{pos}} - p_{\text{neg}}|}{\delta_{\text{mix}}}\right) \times 100\%$$
  2. **Neutral:** $p_{\text{pos}} < \tau_{\text{act}} \land p_{\text{neg}} < \tau_{\text{act}}$
     $$C_{\text{neu}} = (1.0 - \max(p_{\text{pos}}, p_{\text{neg}})) \times 100\%$$
  3. **Positive:** $p_{\text{pos}} \ge \tau_{\text{act}} \land (p_{\text{pos}} - p_{\text{neg}}) > \delta_{\text{dom}}$
     $$C_{\text{pos}} = p_{\text{pos}} \cdot (1.0 - p_{\text{neg}}) \times 100\%$$
  4. **Negative:** $p_{\text{neg}} \ge \tau_{\text{act}} \land (p_{\text{neg}} - p_{\text{pos}}) > \delta_{\text{dom}}$
     $$C_{\text{neg}} = p_{\text{neg}} \cdot (1.0 - p_{\text{pos}}) \times 100\%$$
  5. **Fallback:** Argmax pole with confidence scaled to winning probability.

---

### Step 4: Inference Engine (`src/engine.py`)
Implement `SentimentInferenceEngine`:
- **Model Checkpoint:** `Davlan/afro-xlmr-base`
- **Device Support:** Auto-detect CUDA GPU, Apple MPS, or CPU.
- **Lazy Loading:** `load()` method to load tokenizer and weights only when inference is called.
- Output dictionary format:
  ```json
  {
    "class": "Positive | Negative | Neutral | Mixed",
    "confidence": 94.25,
    "p_pos": 0.9425,
    "p_neg": 0.0210,
    "cleaned_text": "...",
    "latency_ms": 38.4
  }
  ```

---

### Step 5: Full-Featured CLI (`cli.py`)
Use `typer` + `rich`:
- **Single Shot Mode:**
  ```bash
  python cli.py analyze "አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው"
  ```
  Renders a `rich.table.Table` with colored status, confidence %, probabilities, and latency.
- **Interactive REPL Mode:**
  ```bash
  python cli.py repl
  ```
  Provides a persistent `amharic-nlp > ` prompt with prompt loop and `:exit` command.
- **Benchmark Command:**
  ```bash
  python cli.py benchmark
  ```
  Runs the 10 golden benchmark cases and outputs a pass/accuracy summary table.

---

### Step 6: Benchmark Suite & Automated Tests (`tests/`)
Implement `tests/benchmark_cases.json` with the 10 golden test cases:
1. **Positive 1:** `አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው፤ እጅግ በጣም ወደድኩት።`
2. **Positive 2:** `ምርቱ ጥራቱ እጅግ የላቀ ነው፡ ለሁሉም ሰው እመክረዋለሁ! 👍`
3. **Negative 1:** `ምግቡ ፈጽሞ አይበላም፡ ገንዘቤን በከንቱ ነው ያባከንኩት።`
4. **Negative 2:** `ደንበኛ አያያዛቸው እጅግ በጣም ያናድዳል፡ ዳግመኛ አልመለስም።`
5. **Neutral 1:** `ስብሰባው ነገ ከሰዓት በስምንት ሰዓት በዋናው አዳራሽ ይካሄዳል።`
6. **Neutral 2:** `የኢትዮጵያ ብሔራዊ ባንክ አዲሱን የውጭ ምንዛሬ መመሪያ ይፋ አደረገ።`
7. **Mixed 1:** `ስልኩ ውበትና ምርጥ ካሜራ አለው ግን ባትሪው በፍጥነት ያልቃል።`
8. **Mixed 2:** `ሆቴሉ በጣም ያምራል ሰራተኞቹ ግን ጨዋነት የላቸውም።`
9. **Slang Pos:** `ቪዲዮው በእውነት ይመቻል፡ አሪፍ ስራ ነው ባክህ! 🔥`
10. **Slang Neg:** `ዋጋው ጭስ ነው፡ ሰው እንዴት እንዲህ ይበዘበዛል ባክህ።`

Implement `pytest` test runners to verify both the normalizer and model inference output bounds.

---

## 3. VERIFICATION & ACCEPTANCE GATES
Before declaring the feature complete, execute:
1. `pytest tests/` (must pass 100%)
2. `python cli.py analyze "ምግቡ ፈጽሞ አይበላም"` (must output Negative with confidence)
3. `python cli.py benchmark` (must execute and report benchmark results)

DO NOT stop until all test commands pass without error.
