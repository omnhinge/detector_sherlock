# Detector Sherlock — Comprehensive NLP Deep Dive

> **Document Type:** Academic & Technical Deep Dive  
> **Target Audience:** Seminar Committees, Project Examiners, Viva Voce Boards, and Engineering Collaborators  
> **Source Repository:** [`Detector Sherlock`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock)

---

## 1. Introduction: Why NLP is the Core of Detector Sherlock

Most AI tools today are **Generative**: they predict the next likely word based on statistics. However, when an AI model answers questions about medicine, finance, or law, it often produces **hallucinations** — statements that sound grammatically perfect and confident, but are factually fabricated.

Asking another generative model *"Is this true?"* creates a **recursive hallucination trap**: you are asking a system that can lie to verify another system that can lie.

**Detector Sherlock** is built on an entirely different philosophy: **Deterministic, Evidence-Grounded Verification**. Instead of generating new opinions, it uses **Natural Language Processing (NLP)** as an analytical pipeline to:
1. Deconstruct human language into testable factual claims.
2. Route those claims to authoritative databases (**PubMed, Crossref, OpenAlex, SEC EDGAR, World Bank**).
3. Search and extract the most relevant proof sentences.
4. Compare the claim against the proof using both fast lexical heuristics and transformer-based neural reasoning.

Below is an exhaustive, beginner-friendly breakdown of every single NLP technique implemented in the project.

---

## 2. The 10 NLP Techniques Used in Detector Sherlock

```
                              AI-Generated Input Text
                                         │
    [ 1. Dependency Parsing ] ───────────┼──> Decouples coordinated clauses
    [ 2. SPO Triple Extraction ] ────────┼──> Extracts (Subject, Predicate, Object)
                                         ▼
                               Independent Claims
                                         │
    [ 3. Lexical Tokenization ] ─────────┼──> Chops sentences into clean tokens
    [ 4. Stopword Removal ] ─────────────┼──> Strips filler words (the, is, at)
    [ 5. Entity-Gated Filtering ] ───────┼──> Checks for core subject entities
                                         ▼
                           Authoritative Evidence Search
                                         │
    [ 6. Sliding Window Search ] ────────┼──> Scans abstracts sentence-by-sentence
    [ 7. Keyword Overlap Scoring ] ──────┼──> Computes lexical topical similarity
    [ 8. Polarity Matrix Analysis ] ─────┼──> Flags directional contradictions
    [ 9. Cross-Encoder NLI (DeBERTa) ] ──┼──> Neural entailment / contradiction
    [ 10. Inverted Index Recovery ] ─────┼──> Rebuilds text from OpenAlex coordinates
                                         ▼
                                   Final Verdict
```

---

### Technique 1: Dependency Parsing & Syntactic Tree Analysis
- **Implemented in:** [`app/claims.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/claims.py) (Function: `_split_with_spacy`)
- **Library Used:** spaCy (`en_core_web_sm`)

#### What is it in plain English?
In grade school, you might have diagrammed sentences by drawing arrows from words to the verbs they belong to. **Dependency parsing** is the computer version of that: it builds a grammatical tree showing how words connect and depend on one another.

#### Why do we need it?
AI models frequently write compound sentences like:
> *"The Tesla Roadster began customer deliveries in Berlin in March 2023, and Tesla delivered over 1,200 units during its first fiscal quarter."*

If you search an academic or public database for that entire 25-word sentence, **zero documents will match**. No single filing has both clauses in the exact same sentence. We must split compound sentences into **atomic claims**.

Naive code splits sentences on commas (`,`). But splitting on every comma destroys meaning (e.g. splitting *"Berlin, Germany"* separates a city from its country!). Dependency parsing checks the grammatical tree to split **only when two distinct actions (verbs) are present**.

#### How it works in our code:
1. spaCy parses the sentence into tokens with dependency tags (`dep_`) and parts-of-speech (`pos_`).
2. We identify the `ROOT` verb (e.g., `began`).
3. We look for a coordinated verb (`dep_ == "conj"`, `pos_ == "VERB"`, e.g., `delivered`).
4. If the second clause shares the first clause's subject (e.g., *"Aspirin reduces fever and prevents heart attacks"*), our code **inherits the subject** so the second claim becomes *"Aspirin prevents heart attacks"*, rather than an incomplete fragment (*"prevents heart attacks"*).

---

### Technique 2: Subject-Predicate-Object (SPO) Fact Decomposition
- **Implemented in:** [`app/claims.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/claims.py) (Class: `StructuredClaim`, Function: `_extract_structured_claims`)

#### What is it in plain English?
Every complete factual assertion contains three core elements:
1. **Subject:** Who or what is the sentence about?
2. **Predicate:** What action or state is being asserted?
3. **Object:** What is receiving the action or being affected?

#### Why do we need it?
Converting human sentences into structured triples allows the system to evaluate factual completeness and track negation.
- *"Aspirin reduces fever"* $\rightarrow$ `Subject: Aspirin`, `Predicate: reduces`, `Object: fever`.
- *"The treatment helped"* $\rightarrow$ `Subject: The treatment`, `Predicate: helped`, `Object: None` $\rightarrow$ Flagged as **incomplete**.
- *"Aspirin does not reduce fever"* $\rightarrow$ `negated = True`.

---

### Technique 3: Lexical Tokenization
- **Implemented in:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) (Function: `extract_keywords`)

#### What is it in plain English?
Computers cannot read full paragraphs at once; they must break raw text into individual building blocks called **tokens** (usually words or alphanumeric units).

#### Why do we need it?
We need to strip away punctuation, line breaks, and formatting noise while keeping important numbers and scientific terms (like `COVID-19`, `10-K`, `Q1`, `type-2`).

#### How it works in our code:
We use a targeted regular expression:
```python
words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", clean_text.lower())
```

---

### Technique 4: Stopword Filtering & Noise Reduction
- **Implemented in:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) (Constant: `STOPWORDS`)

#### What is it in plain English?
In English, words like *"the"*, *"is"*, *"at"*, *"which"*, and *"were"* appear in almost every sentence. They provide grammatical glue, but carry **zero factual information**.

#### Why do we need it?
Comparing sentences without stopword removal causes false matches because sentences share common filler words like `is`, `a`, `for`. Removing stopwords ensures only the core factual keywords (`metformin`, `diabetes`, `cardiac`) are compared.

---

### Technique 5: Term Overlap Scoring (Jaccard-Style Lexical Similarity)
- **Implemented in:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) (Function: `find_best_passage`)

#### The Formula:
$$\text{Overlap Ratio} = \frac{|\text{Claim Keywords} \cap \text{Sentence Keywords}|}{|\text{Claim Keywords}|}$$

This metric is computationally lightweight ($\approx 0.05\text{ ms}$), allowing us to screen hundreds of candidate sentences in real time.

---

### Technique 6: Sliding Window Passage Retrieval
- **Implemented in:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) (Function: `find_best_passage`)

A scientific paper abstract is usually 200–300 words long. A **sliding window** moves sentence-by-sentence across the abstract to pinpoint the single exact sentence that serves as the proof quote.

---

### Technique 7: Semantic Polarity & Antagonistic Sentiment Analysis
- **Implemented in:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) (Constant: `POLARITY_PAIRS`, Function: `check_for_contradiction`)

We maintain an **antagonistic polarity matrix**:
```python
POLARITY_PAIRS = [
    ({"reduces", "decreases", "lowers", "prevents", "cures", "treats", "protects"},
     {"causes", "increases", "induces", "aggravates", "triggers", "elevates"}),
    ({"safe", "beneficial", "effective", "approved", "recommended"},
     {"dangerous", "harmful", "toxic", "banned", "ineffective", "hazardous"}),
]
```
If the claim contains a word from Group A (`reduces`) but the evidence contains an opposing word from Group B (`increases`), the system detects a polarity flip and immediately assigns a verdict of **`CONTRADICTED`**.

---

### Technique 8: Entity Gating & Noise Filtering
- **Implemented in:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) (Function: `evaluate_claim_against_evidence`)

1. We extract all capitalized entity tokens from the claim (e.g. `Tesla`, `Metformin`, `India`).
2. If the paper's title and abstract **do not contain any of these entities**, the paper is discarded as search noise.
3. If no entity-matching paper has at least 20% relevance, the system cleanly returns **`INCONCLUSIVE`** rather than showing random search noise.

---

### Technique 9: Cross-Encoder Natural Language Inference (NLI)
- **Implemented in:** [`app/nli_engine.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/nli_engine.py) (Model: `cross-encoder/nli-deberta-v3-small`)

1. Uses a **Cross-Encoder DeBERTa Transformer** where Premise and Hypothesis are fed simultaneously.
2. The model outputs probabilities over `[contradiction, entailment, neutral]`.
3. If $\text{Entailment} \ge 65\%$, the claim is confirmed as **`SUPPORTED`**.
4. If $\text{Contradiction} \ge 65\%$, the claim is flagged as **`CONTRADICTED`**.

---

### Technique 10: Inverted Index Positional Text Recovery
- **Implemented in:** [`app/sources/openalex.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/openalex.py) (Function: `reconstruct_abstract`)

OpenAlex returns abstracts as inverted word-position coordinate dictionaries. Detector Sherlock inverts the mapping, sorts by coordinate index, and reconstructs coherent, human-readable text.

---

## 3. Summary Formulas for Presentation Reference

### 1. Overall Claim Reliability Score:
$$\text{Reliability Score} = \left(\frac{\text{Count of Supported Claims}}{\text{Total Claims Detected}}\right) \times 100$$

### 2. Evidence Coverage Rate:
$$\text{Evidence Coverage} = \left(\frac{\text{Claims with Literature Found}}{\text{Total Claims Detected}}\right) \times 100$$

### 3. Dual-Tier Blended Confidence:
$$\text{Confidence} = 0.40 \times (\text{Lexical Overlap Score}) + 0.60 \times (\text{Neural Entailment Probability})$$

### 4. Verdict Definitions:
- **`SUPPORTED`**: Authoritative literature directly corroborates the factual claim.
- **`CONTRADICTED`**: Authoritative literature asserts opposing factual findings.
- **`INCONCLUSIVE`**: No authoritative public record or peer-reviewed study in the queried repositories possessed sufficient evidence overlap to corroborate the claim.
