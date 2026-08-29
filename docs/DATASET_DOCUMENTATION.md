# Dataset Documentation & ML Data Card: Amh-Sent-Bench

**Amharic Sentiment Intelligence & Multi-Polarity Benchmark Corpus**  
*A Standardized Dataset Card and Linguistic Specification for Amharic Affective Computing and Neural Sentiment Classification.*

---

## 1. Dataset Identity & Metadata

### 1.1 Dataset Name
**Amharic Sentiment Intelligence & Multi-Polarity Benchmark Corpus (`Amh-Sent-Bench`)**

### 1.2 Overview & Purpose
`Amh-Sent-Bench` is an aggregated, standardized, and normalized multi-domain dataset designed for training, calibrating, and benchmarking neural Natural Language Processing (NLP) models on the Amharic language (አማርኛ). The dataset supports 4-class affective polarity classification (**Positive**, **Negative**, **Neutral**, and **Mixed**) across diverse textual domains including social media, e-commerce reviews, telecom feedback, banking operations, and formal public announcements.

### 1.3 Primary Source Corpora

| Source Corpus | Primary Medium | Size (Samples) | Description & Focus |
| :--- | :--- | :--- | :--- |
| **AfriSenti-SemEval 2023 Task 12 (Amharic Track)** | Twitter / X Social Media | ~8,000 | Colloquial social media posts, slang, informal customer commentary, emojis, and hashtags. |
| **ASAD (Amharic Sentiment Analysis Dataset)** | Customer Reviews & E-commerce | ~5,500 | Product reviews, merchant evaluations, telecom support tickets, and mobile banking feedback. |
| **Custom Ethiopic Benchmark Set (Golden-10)** | Syntactic & Contrastive Probes | 10 | Curated evaluation dataset specifically designed to probe morphological negation flipping, clause contrast, and factual neutrality. |

### 1.4 Source Repositories & URLs
- **AfriSenti Benchmark:**  
  - HuggingFace Dataset: [`https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter-Sentiment`](https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter-Sentiment)
  - SemEval 2023 Task 12 GitHub: [`https://github.com/afrisenti-semeval/afrisenti-semeval-2023`](https://github.com/afrisenti-semeval/afrisenti-semeval-2023)
- **ASAD Corpus:**  
  - GitHub Repository: [`https://github.com/skand/ASAD`](https://github.com/skand/ASAD)
- **Production Pre-trained Model Checkpoint:**  
  - HuggingFace Model Hub: [`https://huggingface.co/Tirsit/amharic-sentiment-afriberta`](https://huggingface.co/Tirsit/amharic-sentiment-afriberta)

### 1.5 Licensing & Usage Terms
- **AfriSenti-SemEval 2023:** Creative Commons Attribution-NonCommercial 4.0 International (**CC-BY-NC-4.0**) / Open Data Commons.
- **ASAD Corpus:** **MIT License** / Academic Research License.
- **Syntactic Benchmark & Preprocessor Engine:** **Apache 2.0 / MIT License** (Internal Production Workspace).

---

## 2. Dataset Schema & Feature Definitions

Each record in the dataset is represented as a structured JSON object containing both raw inputs, normalized linguistic features, domain metadata, and calibrated probability distributions.

### 2.1 Field Schema Specification

| Field Name | Data Type | Nullable | Description | Canonical Example |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `string` | No | Unique alphanumeric record identifier. | `"AMH-SENT-001"` |
| `raw_text` | `string` | No | Original, unedited text preserving raw Ethiopic Fidels, native punctuation, elongations, and emojis. | `"ሲስተማችሁ ሁልጊዜ አይሰራም፡ ገንዘቤ ተቆርጦ አገልግሎት አላገኘሁም፡ በጣም አሳፋሪ ነው!"` |
| `cleaned_text` | `string` | No | $O(N)$ normalized string with unified homophones, stripped elongation, and standardized punctuation. | `"ሲስተማችሁ ሁልጊዜ አይሰራም ገንዘቤ ተቆርጦ አገልግሎት አላገኘሁም በጣም አሳፋሪ ነው!"` |
| `domain` | `string` | No | Source industry or functional domain (`Telecom`, `Banking`, `E-Commerce`, `Social Media`, `General News`). | `"Telecom"` |
| `ground_truth` | `string` | No | Annotated target sentiment category (`Positive`, `Negative`, `Neutral`, `Mixed`). | `"Negative"` |
| `p_pos` | `float` | No | Model probability activation on the Positive polar axis ($[0.0, 1.0]$). | `0.0615` |
| `p_neg` | `float` | No | Model probability activation on the Negative polar axis ($[0.0, 1.0]$). | `0.8729` |
| `p_neu` | `float` | No | Baseline probability activation on the Neutral axis ($[0.0, 1.0]$). | `0.0656` |
| `confidence` | `float` | No | Continuous calibrated confidence percentage score ($[0.0, 100.0]$). | `81.92` |
| `clauses` | `list[string]` | No | List of constituent sub-clauses extracted along discourse and punctuation boundaries. | `["ሲስተማችሁ ሁልጊዜ አይሰራም", "ገንዘቤ ተቆርጦ አገልግሎት አላገኘሁም", "በጣም አሳፋሪ ነው"]` |
| `has_contrast` | `boolean` | No | Flag indicating whether contrastive conjunctions (`ግን`, `ነገር ግን`, `ሆኖም`, `ቢሆንም`) were detected. | `false` |
| `token_count` | `integer` | No | Subword BPE token length after SentencePiece tokenization. | `18` |

### 2.2 Sample Record Representation

```json
{
  "id": "AMH-SENT-007",
  "raw_text": "ስልኩ ውበትና ምርጥ ካሜራ አለው ግን ባትሪው በፍጥነት ያልቃል።",
  "cleaned_text": "ስልኩ ውበትና ምርጥ ካሜራ አለው ግን ባትሪው በፍጥነት ያልቃል.",
  "domain": "E-Commerce",
  "ground_truth": "Mixed",
  "p_pos": 0.9464,
  "p_neg": 0.8258,
  "p_neu": 0.0422,
  "confidence": 31.83,
  "clauses": [
    "ስልኩ ውበትና ምርጥ ካሜራ አለው ግን ባትሪው በፍጥነት ያልቃል።",
    "ስልኩ ውበትና ምርጥ ካሜራ አለው",
    "ባትሪው በፍጥነት ያልቃል"
  ],
  "has_contrast": true,
  "token_count": 14
}
```

---

## 3. Target Classes & Annotation Taxonomy

The dataset employs a **Decoupled 4-Class Taxonomy** designed to eliminate single-label bias and accurately handle complex, multi-clause Amharic discourse.

```
                         [ Amharic Input Text ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ Positive Activation ]             [ Negative Activation ]
           P(pos) >= 0.50                      P(neg) >= 0.50
                  │                                   │
                  ├─────────────────┬─────────────────┤
                  ▼                 ▼                 ▼
          [ POSITIVE ]          [ MIXED ]         [ NEGATIVE ]
         P(pos) Dominates     Both Active       P(neg) Dominates
         |P(pos)-P(neg)|>0.15  |P(pos)-P(neg)|<=0.25 |P(neg)-P(pos)|>0.15
                                    ▲
                                    │
                               [ NEUTRAL ]
                           Both P(pos), P(neg) < 0.50
```

### 3.1 Class Definitions & Annotation Guidelines

#### 1. `Positive` (አዎንታዊ)
- **Definition:** Expresses customer satisfaction, praise, gratitude, admiration, high service efficiency, or positive emotional attachment.
- **Key Lexical Markers:** `ምርጥ` (best/superb), `ፈጣን` (fast), `አመሰግናለሁ` (thank you), `ጥሩ` (good), `ይመቻል` (it's great/comfortable), `ደስ ይላል` (pleasing), `ድንቅ` (wonderful), `የላቀ` (superior).
- **Mathematical Criteria:** $P(\text{pos}) \ge \tau_{\text{act}}\ (0.50)$ and $(P(\text{pos}) - P(\text{neg})) > \delta_{\text{dom}}\ (0.15)$.

#### 2. `Negative` (አሉታዊ)
- **Definition:** Expresses dissatisfaction, frustration, anger, system breakdown, financial loss, poor customer handling, or explicit rejection.
- **Key Lexical Markers:** `አይሰራም` (does not work), `አሳፋሪ` (shameful), `ያናድዳል` (infuriating), `መጥፎ` (bad), `ሌቦች` (thieves), `አልመለስም` (I won't return), `ይበዘበዛል` (exploitative), `አሰልቺ` (boring).
- **Mathematical Criteria:** $P(\text{neg}) \ge \tau_{\text{act}}\ (0.50)$ and $(P(\text{neg}) - P(\text{pos})) > \delta_{\text{dom}}\ (0.15)$.

#### 3. `Neutral` (ገለልተኛ)
- **Definition:** Purely informative, factual, objective, or transactional statements devoid of emotional charge. Includes office hours, policy updates, bank notices, and factual inquiries.
- **Key Lexical Markers:** `ክፍት ነው` (is open), `ይካሄዳል` (will take place), `መመሪያ` (directive), `ከሰኞ እስከ አርብ` (Monday to Friday), `ዋና መስሪያ ቤት` (headquarters).
- **Mathematical Criteria:** $P(\text{pos}) < \tau_{\text{act}}\ (0.50)$ and $P(\text{neg}) < \tau_{\text{act}}\ (0.50)$. Confidence calibrated as $(1.0 - \max(P(\text{pos}), P(\text{neg}))) \times 100$.

#### 4. `Mixed` (ድብልቅ)
- **Definition:** Compound or complex sentences where the speaker expresses both positive approval and negative criticism regarding different facets of a product or service. Typically bridged by contrastive conjunctions.
- **Key Discourse Markers:** `ግን` (but), `ነገር ግን` (however), `ሆኖም` (nonetheless), `ቢሆንም` (even though), `ዳሩ ግን` (yet).
- **Mathematical Criteria:** Both $P(\text{pos}) \ge \tau_{\text{act}}$ and $P(\text{neg}) \ge \tau_{\text{act}}$ with $|P(\text{pos}) - P(\text{neg})| \le \delta_{\text{mix}}\ (0.25)$.

---

## 4. Data Quality, Preprocessing & Normalization

Amharic orthography exhibits substantial historical, phonological, and typographic variance. To ensure uniform representation before tokenization, the dataset defines an $O(N)$ deterministic normalization pipeline.

### 4.1 Ge'ez Homophone Unification Table
In standard modern Amharic, multiple Fidel characters share identical phonemes. The normalization table unifies these into standard canonical graphemes:

| Series | Phoneme | Redundant Forms | Canonical Target | Count |
| :--- | :--- | :--- | :--- | :--- |
| **Ha Series** | /h/ | `ሐ, ሑ, ሒ, ሓ, ሔ, ሕ, ሖ, ሗ`<br>`ኀ, ኁ, ኂ, ኃ, ኄ, ኅ, ኆ, ኇ` | `ሀ, ሁ, ሂ, ሃ, ሄ, ህ, ሆ, ኋ` | 16 |
| **Sa Series** | /s/ | `ሠ, ሡ, ሢ, ሣ, ሤ, ሥ, ሦ, ሧ` | `ሰ, ሱ, ሲ, ሳ, ሴ, ስ, ሶ, ሷ` | 8 |
| **Glottal Series** | /ʔ/ or /a/ | `ዐ, ዑ, ዒ, ዓ, ዔ, ዕ, ዖ` | `አ, ኡ, ኢ, ኣ, ኤ, እ, ኦ` | 7 |
| **Tsa Series** | /ts'/ | `ፀ, ፁ, ፂ, ፃ, ፄ, ፅ, ፆ` | `ጸ, ጹ, ጺ, ጻ, ጼ, ጽ, ጾ` | 7 |
| **Labiovelar Series** | /kʷ/, /gʷ/, etc. | `ቈ, ቊ, ቍ, ቌ, ኰ, ኲ, ኵ, ኴ, ጐ, ጒ, ጕ, ጔ, ዀ, ዂ, ዅ, ዄ` | Reduced to standard forms (`ቁ, ቂ, ቅ, ቄ`, `ኩ, ኪ, ክ, ኬ`, `ጉ, ጊ, ግ, ጌ`, `ሁ, ሂ, ህ, ሄ`) | 16 |

### 4.2 Typographic & Punctuation Transliteration
Traditional Ge'ez punctuation marks are mapped to ASCII equivalents to match the tokenizer's pre-training vocabulary:
- **Ethiopic Wordspace (`፡`):** Transliterated to ASCII space (` `).
- **Ethiopic Full Stop (`።`):** Transliterated to ASCII period (`.`).
- **Ethiopic Comma (`፣`):** Transliterated to ASCII comma (`,`).
- **Ethiopic Semicolon (`፤`):** Transliterated to ASCII semicolon (`;`).
- **Ethiopic Colons (`፥`, `፦`):** Transliterated to ASCII colon (`:`).
- **Ethiopic Question Mark (`፧`):** Transliterated to ASCII question mark (`?`).

### 4.3 Text Hygiene & Noise Reduction
1. **Character Elongation Collapsing:** Social media user comments often stretch characters for emotional emphasis (e.g. `በጣምምምምም` &rarr; `በጣም`, `ዋውውው` &rarr; `ዋው`). Regex rule `([^\d\s])\1{2,}` collapses 3+ consecutive identical characters to a single character.
2. **Punctuation Compression:** Consecutive punctuation (`!!!!!` or `????`) is compressed to a single token (`!`, `?`).
3. **Empty / Whitespace-only Inputs:** Handled safely by falling back to neutral baseline with 0.0 polar probability.

---

## 5. Dataset Bias, Fairness & Linguistic Limitations

### 5.1 Dialectal & Regional Representation
- **Urban Central Amharic Dominance:** The majority of social media and e-commerce records originate from urban centers (primarily Addis Ababa and surrounding regions). Consequently, the vocabulary strongly reflects urban colloquial Amharic and modern youth slang (e.g. `ይመቻል`, `ጭስ ነው`, `እብድ አሰራር`).
- **Regional Dialect Nuances:** Regional Amharic dialects (such as Gojjam, Wollo, Gondar, or Menz varieties) may use localized vocabulary, archaic phrasing, or altered verb conjugations that have lower representation in the pre-training corpus.

### 5.2 Code-Switching & Latin Transliteration (Amglish)
- Social media corpora frequently exhibit code-switching between Amharic and English (e.g. `App ኡ በጣም slow ነው`).
- Sentences written entirely in Latin transliteration ("Amglish" / "Fidel in Latin") are not normalized by the Ge'ez preprocessor and rely solely on the multilingual subword vocabulary of the underlying transformer.

### 5.3 Morphological Negation Handling
Amharic employs complex morphological prefixation and suffixation for negation:
- **Prefix + Suffix Circumfixes:** `አል-...-ም` (e.g., `አልወደድኩትም` &rarr; "I did not like it").
- **Imperfective Negation:** `አይ-...-ም` (e.g., `አይሰራም` &rarr; "it does not work").
- **Standalone Negation Particle:** `አይደለም` (e.g., `ፈጣን አይደለም` &rarr; "it is not fast").

*Quality Assurance Rule:* The preprocessor strictly preserves morphological negation affixes during normalization to prevent accidental polarity flipping.

### 5.4 Privacy & Anonymization (PII)
- **Phone Numbers:** Ethiopian mobile phone patterns (`+251 9...`, `09...`, `07...`) are masked as `[PHONE]`.
- **Account Numbers:** Personal bank account digits are redacted as `[ACCOUNT]`.
- **User Handles:** Direct social media mentions (`@username`) are normalized to `@user`.

---

## 6. Golden Benchmark Test Matrix (10 Core Verification Cases)

The following 10 reference cases form the immutable Golden Benchmark for model regression testing:

| # | Category | Amharic Input Text | English Translation | Expected Class |
|---|---|---|---|---|
| **1** | **Positive 1** | `አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው፤ እጅግ በጣም ወደድኩት።` | *The service is very fast and reliable; I liked it very much.* | **Positive** |
| **2** | **Positive 2** | `ምርቱ ጥራቱ እጅግ የላቀ ነው፡ ለሁሉም ሰው እመክረዋለሁ! 👍` | *The product quality is superior; I recommend it to everyone!* | **Positive** |
| **3** | **Negative 1** | `ምግቡ ፈጽሞ አይበላም፡ ገንዘቤን በከንቱ ነው ያባከንኩት።` | *The food is completely inedible; I wasted my money in vain.* | **Negative** |
| **4** | **Negative 2** | `ደንበኛ አያያዛቸው እጅግ በጣም ያናድዳል፡ ዳግመኛ አልመለስም።` | *Their customer handling is infuriating; I will never return.* | **Negative** |
| **5** | **Neutral 1** | `ስብሰባው ነገ ከሰዓት በስምንት ሰዓት በዋናው አዳራሽ ይካሄዳል።` | *The meeting will take place tomorrow at 2:00 PM in the main hall.* | **Neutral** |
| **6** | **Neutral 2** | `የኢትዮጵያ ብሔራዊ ባንክ አዲሱን የውጭ ምንዛሬ መመሪያ ይፋ አደረገ።` | *The National Bank of Ethiopia announced the new foreign exchange directive.* | **Neutral** |
| **7** | **Mixed 1** | `ስልኩ ውበትና ምርጥ ካሜራ አለው ግን ባትሪው በፍጥነት ያልቃል።` | *The phone has beauty and a superb camera, but the battery drains fast.* | **Mixed** |
| **8** | **Mixed 2** | `ሆቴሉ በጣም ያምራል ሰራተኞቹ ግን ጨዋነት የላቸውም።` | *The hotel is very beautiful, but the staff lack politeness.* | **Mixed** |
| **9** | **Slang Pos** | `ቪዲዮው በእውነት ይመቻል፡ አሪፍ ስራ ነው ባክህ! 🔥` | *The video is truly enjoyable; it's awesome work, man!* | **Positive** |
| **10** | **Slang Neg** | `ዋጋው ጭስ ነው፡ ሰው እንዴት እንዲህ ይበዘበዛል ባክህ።` | *The price is smoke (exorbitant); how can people be exploited like this.* | **Negative** |

---

## 7. How to Access & Evaluate

### Running CLI Benchmark:
```bash
python cli.py benchmark --cases tests/benchmark_cases.json
```

### Running Interactive Desktop Studio:
```bash
python gui.py
# Navigate to Tab 2: "Interactive Golden Benchmark"
```

### Executing Automated Unit Tests:
```bash
pytest tests/ -v
```

---

*Author / Maintainer:* **Amharic NLP & Sentiment Intelligence Core Engineering Team**  
*Document Version:* `1.0.0`  
*Last Updated:* `2026-08-29`
