# Amharic Sentiment Classification Engine & CLI Test Harness

A high-performance, deterministic Amharic Sentiment Classification Engine leveraging Afro-centric transformer representations (`Davlan/afro-xlmr-base`), $O(N)$ orthographic normalization, and continuous decoupled dual-axis thresholding.

---

## 🌟 Key Features

1. **Afro-XLMR Foundation**: Utilizes `Davlan/afro-xlmr-base` contextual representations tailored for African languages.
2. **$O(N)$ Ge'ez Normalization**: High-speed C-level `str.translate` normalizer handling Ha/Sa/Glottal/Tsa homophone unifications, labiovelars, Ge'ez punctuation, and character elongation hygiene.
3. **Decoupled Dual-Axis Calibration**: Independent positive ($p_{\text{pos}}$) and negative ($p_{\text{neg}}$) activation axes allowing natural multi-aspect and mixed sentiment recognition (`Positive`, `Negative`, `Neutral`, `Mixed`).
4. **Rich Terminal Experience**: Interactive REPL mode, single-shot analyzer, and automated benchmark test runner built with `Typer` and `Rich`.
5. **100% Test Coverage**: Full pytest suite covering orthography, calibration mathematics, and the 10 Golden Benchmark Cases.

---

## 📐 Mathematical Formulation

The engine decouples sentiment polarities across activation floor $\tau_{\text{act}} = 0.50$, mixed margin $\delta_{\text{mix}} = 0.25$, and dominance margin $\delta_{\text{dom}} = 0.15$:

- **Mixed Condition:**
  $$p_{\text{pos}} \ge \tau_{\text{act}} \land p_{\text{neg}} \ge \tau_{\text{act}} \land |p_{\text{pos}} - p_{\text{neg}}| \le \delta_{\text{mix}}$$
  $$C_{\text{mix}} = \left(\frac{p_{\text{pos}} + p_{\text{neg}}}{2}\right) \cdot \left(1.0 - \frac{|p_{\text{pos}} - p_{\text{neg}}|}{\delta_{\text{mix}}}\right) \times 100\%$$

- **Neutral Condition:**
  $$p_{\text{pos}} < \tau_{\text{act}} \land p_{\text{neg}} < \tau_{\text{act}}$$
  $$C_{\text{neu}} = (1.0 - \max(p_{\text{pos}}, p_{\text{neg}})) \times 100\%$$

- **Positive Dominance:**
  $$p_{\text{pos}} \ge \tau_{\text{act}} \land (p_{\text{pos}} - p_{\text{neg}}) > \delta_{\text{dom}}$$
  $$C_{\text{pos}} = p_{\text{pos}} \cdot (1.0 - p_{\text{neg}}) \times 100\%$$

- **Negative Dominance:**
  $$p_{\text{neg}} \ge \tau_{\text{act}} \land (p_{\text{neg}} - p_{\text{pos}}) > \delta_{\text{dom}}$$
  $$C_{\text{neg}} = p_{\text{neg}} \cdot (1.0 - p_{\text{pos}}) \times 100\%$$

---

## 🚀 Installation

```bash
# Clone and enter directory
git clone https://github.com/Overloadpy/Amharic_Sentiment_Classification-engine.git
cd Amharic_Sentiment_Classification-engine

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 CLI Usage

### 1. Single-Shot Analysis
```bash
python cli.py analyze "አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው፤ እጅግ በጣም ወደድኩት።"
```

### 2. Interactive REPL Mode
```bash
python cli.py repl
```
```text
amharic-nlp > አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው
 -> Class: Positive | Conf: 86.40% | P(pos): 0.900 | P(neg): 0.040 | Latency: 55.2ms
amharic-nlp > :exit
```

### 3. Golden Benchmark Evaluation
```bash
python cli.py benchmark
```

---

## 🧪 Testing

Execute automated unit and integration tests:

```bash
pytest tests/ -v
```

---

## 📊 Benchmark Performance

| # | Category | Expected | Accuracy | Avg Latency |
|---|----------|----------|----------|-------------|
| 1-2 | Standard Positives | Positive | 100% | ~48 ms |
| 3-4 | Standard Negatives | Negative | 100% | ~48 ms |
| 5-6 | Factual Neutrals | Neutral | 100% | ~48 ms |
| 7-8 | Contrastive / Multi-Aspect | Mixed | 100% | ~48 ms |
| 9-10 | Modern Slang (Pos/Neg) | Pos / Neg | 100% | ~48 ms |
| **Overall** | **10 Golden Cases** | — | **100.0%** | **~48.66 ms** |
