# Detector Sherlock — Strategic Roadmap & Robustness Architecture

**Document Version:** 2.0  
**Focus Areas:** Real-Time Authoritative Public Records, NLP Robustness, Semantic Inference, and Enterprise Production Readiness  
**Target Audience:** Academic Seminar Committee, Research Supervisors, and Open-Source Collaborators  

---

## 1. Executive Summary

In its foundational phase, **Detector Sherlock** established a deterministic, evidence-grounded verification pipeline against scholarly repositories (**PubMed**, **Crossref**, **OpenAlex**) combined with a **Manifest V3 Chrome Extension**.

While peer-reviewed literature serves as the scientific gold standard, LLM hallucination in real-world deployments frequently spans **contemporary commercial claims** (e.g., Tesla deliveries, stock numbers), **legal precedents**, **macroeconomic figures**, and **breaking news**.

This document outlines the **4-Phase Strategic Roadmap** to scale Detector Sherlock into a multi-tier, real-time authoritative verification engine.

---

## 2. Multi-Domain Public Records Integration (Data Expansion)

```
                            ┌── Academic Tier (PubMed, Crossref, OpenAlex)
                            ├── Corporate & Regulatory (SEC EDGAR, MCA21)
Detector Sherlock Router ───┼── Macroeconomic & Gov (RBI, MoSPI, Data.gov.in)
                            ├── Legal & Judicial (India Code, Supreme Court)
                            └── Real-Time News Wire (GDELT, AP/Reuters Feeds)
```

### Phase 1: Corporate Disclosures & Regulatory Records
*Target Claims: Financial figures, production numbers, product launches, corporate filings (e.g., "Tesla delivered 1,200 Roadsters in Q1").*

- **SEC EDGAR API (U.S.)**:
  - Direct integration with Form 10-K (Annual reports), 10-Q (Quarterly financials), and 8-K (Major current events).
  - Eliminates reliance on unverified blog rumors for corporate facts.
- **Ministry of Corporate Affairs (MCA21 / India)**:
  - Regulatory filings, company incorporation records, and authorized director datasets.
- **USPTO & Google Patents API**:
  - Direct patent number and patent owner lookup for intellectual property claims.

### Phase 2: Macroeconomic & Government Statistics
*Target Claims: GDP growth rates, inflation numbers, population demographics, policy announcements.*

- **Open Government Data Platform India (`data.gov.in`)**:
  - Official ministry datasets across health, agriculture, education, and commerce.
- **Reserve Bank of India (RBI) Database on Indian Economy (DBIE)**:
  - Real-time repo rates, foreign exchange reserves, monetary policy metrics.
- **World Bank Open Data & FRED (Federal Reserve Economic Data)**:
  - Global economic indicators, historical inflation, and trade balances.

### Phase 3: Legal & Judicial Records
*Target Claims: Supreme Court rulings, constitutional articles, enacted parliamentary bills.*

- **India Code & Legislative Departments**:
  - Digital gazettes, enacted central acts, and statutory rules.
- **eCourts Services & Supreme Court Case Status**:
  - Judgments, citations, and constitutional bench orders.

### Phase 4: Real-Time Fact & News Registry
*Target Claims: Breaking natural events, geopolitical updates, international sports.*

- **GDELT Project (Global Database of Events, Language, and Tone)**:
  - Monitored real-time stream of global news across 100 languages, updated every 15 minutes.
- **Associated Press (AP) / Reuters Verified Wire Feeds**:
  - Primary source journalistic reports with cryptographic or verified publisher signatures.

---

## 3. Engineering Robustness & Pipeline Hardening

To transform the prototype into a production-grade, fault-tolerant system, the following architectural upgrades are specified:

### 3.1 Advanced Semantic Claim Decomposition (Atomic Facts)
- **Current Limitation:** Naive regex punctuation splitting can cluster multiple independent claims into one sentence.
- **Roadmap Upgrade:**
  - Implement an **Atomic Proposition Generator** (inspired by *Min et al., FActScore*).
  - Decompose compound sentences into independent Subject-Predicate-Object (SPO) triples:
    - *Compound Sentence:* "The Tesla Roadster began customer deliveries in Berlin in March 2023, delivering over 1,200 units."
    - *Triple 1:* `(Tesla Roadster, began deliveries in, Berlin)`
    - *Triple 2:* `(Tesla Roadster delivery date, is, March 2023)`
    - *Triple 3:* `(First fiscal quarter units, count, >1,200)`
  - Each atomic triple is corroborated or refuted independently.

### 3.2 Dual-Tier Inference: Lexical Heuristics + Transformer NLI
- **Current Limitation:** Lexical keyword overlap cannot detect subtle semantic nuances or passive-voice variations.
- **Roadmap Upgrade (Hybrid NLI Engine):**
  - **Tier 1 (Fast Filter):** Existing lexical keyword matching prunes irrelevant records in $< 10\text{ms}$.
  - **Tier 2 (Neural Verification):** Deploy a quantized cross-encoder or Natural Language Inference (NLI) model (e.g., `DeBERTa-v3-small` / `roberta-large-snli_mnli` via **ONNX Runtime** on CPU):
    - **Premise:** Extracted evidence passage.
    - **Hypothesis:** Candidate claim.
    - **Softmax Output:** Probability distributions over $[\text{Entailment}, \text{Contradiction}, \text{Neutral}]$.
  - Preserves sub-second response times while delivering state-of-the-art semantic accuracy.

### 3.3 Dynamic Caching & Rate-Limit Resilience
- **Redis Multi-Level Caching**:
  - Cache normalized source responses by query hash with a domain-specific TTL (e.g., 30 days for PubMed papers, 1 hour for SEC filings).
  - Slashes repeated query latency from $\sim 1.5\text{s}$ down to $< 15\text{ms}$.
- **Adaptive Exponential Backoff & Circuit Breakers**:
  - Implement resilience patterns via `tenacity` or `resilience4j`-style interceptors to gracefully handle external API rate limits (e.g., NCBI 429 warnings).

---

## 4. Evaluation Benchmark & Calibration Plan

To validate Detector Sherlock's reliability scores before academic publication:

1. **Benchmark Datasets**:
   - Evaluate against **TruthfulQA** (Lin et al., 2022) for resistance to common misconceptions.
   - Evaluate against **FActScore** (Min et al., 2023) for biographical and entity accuracy.
   - Evaluate against **RAGTruth** (Niu et al., 2024) for hallucination detection in retrieval-augmented setups.
2. **Calibration Metrics**:
   - Compute **Brier Score** and **Expected Calibration Error (ECE)** to ensure that a reported Reliability Score of $80\%$ genuinely correlates with an $80\%$ statistical probability of factual correctness.

---

## 5. Milestone Timeline & Phased Implementation

| Phase | Milestone Objective | Target Deliverable | Expected Performance Impact |
| :---: | :--- | :--- | :--- |
| **Q1** | **Atomic Claim Splitter** | `app/claims.py` upgrade using dependency parsing (spaCy) | Eliminates compound claim conflation |
| **Q2** | **Corporate & Gov Connectors** | SEC EDGAR + `data.gov.in` API connectors in `app/sources/` | Verifies commercial & economic claims |
| **Q3** | **Hybrid NLI Integration** | ONNX-quantized DeBERTa cross-encoder in `app/evaluator.py` | Semantic entailment / zero keyword mismatch |
| **Q4** | **Benchmark Evaluation** | Precision/Recall report on TruthfulQA & FActScore | Calibrated reliability metrics & research paper |
