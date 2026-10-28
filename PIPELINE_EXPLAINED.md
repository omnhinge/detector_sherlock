# Detector Sherlock — End-to-End Pipeline Explained

> **Document Type:** System Pipeline Reference  
> **Audience:** Beginners, Project Reviewers, and Viva Committees  
> **Repository:** [`Detector Sherlock`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock)

---

## 1. What Does This Pipeline Do? (The Big Picture)

When an AI like ChatGPT or Claude gives an answer, how do you know if it is telling the truth or **hallucinating** (making things up)?

Instead of asking another AI to check it, **Detector Sherlock** passes the AI's answer through an automated **6-step verification pipeline**. It takes the statement, breaks it into facts, searches real world databases (like scientific research, corporate filings, and government statistics), and evaluates the truthfulness using Natural Language Processing (NLP).

---

## 2. Visual Pipeline Flowchart

```
                 [ User highlights text on web / Chrome Extension ]
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. INPUT ACQUISITION                                                   │
   │    Extension sends selected text to FastAPI backend (POST /verify)     │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 2. ATOMIC CLAIM DECOMPOSITION (app/claims.py)                          │
   │    spaCy splits compound sentences into independent atomic facts        │
   │    Example: "A and B" -> Claim 1: "A", Claim 2: "B"                    │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 3. HYBRID SEMANTIC DOMAIN ROUTING (app/domain_classifier.py & router)  │
   │    Combines sentence embeddings (all-MiniLM-L6-v2) & lexical signals:  │
   │    - Biomedical -> PubMed, Wikipedia, Crossref, OpenAlex               │
   │    - Financial  -> SEC EDGAR, News, Wikipedia, Crossref, OpenAlex      │
   │    - Government -> World Bank / GovData, News, Wikipedia, Crossref     │
   │    - Computing  -> Wikipedia, Crossref, OpenAlex                       │
   │    - Multi-Domain -> Merges target sources without duplicate queries   │
   │    - Fallback   -> Offline keyword router & broad cross-index search   │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 4. CONCURRENT EVIDENCE SEARCH (app/sources/search_all.py)              │
   │    Fires asynchronous API calls in parallel; merges & deduplicates     │
   │    papers by DOI and title to create clean evidence records            │
   └────────────────────────────────────┬───────────────────────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 5. DUAL-TIER NLP VERIFICATION ENGINE (app/evaluator.py)                │
   │    • Tier 1 (Lexical Filter, <5ms):                                    │
   │      Tokenizes, removes stopwords, scans abstract with sliding window   │
   │    • Entity Gate:                                                      │
   │      Ensures evidence actually mentions the subject (e.g., "Tesla")    │
   │    • Tier 2 (Neural NLI Transformer):                                  │
   │      DeBERTa Cross-Encoder calculates semantic Entailment/Contradiction│
   └────────────────────────────────────┬───────────────────────────────────┘
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 6. VERDICT ASSIGNMENT & RISK SCORING (app/main.py)                     │
   │    • Assigns Verdict: SUPPORTED / CONTRADICTED / INCONCLUSIVE          │
   │    • Calculates Reliability Score (%) & Hallucination Risk             │
   │    • Displays evidence quote & link in Chrome Extension card           │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Breakdown (In Simple Words)

### Step 1: Input Acquisition
- **Where:** [`extension/`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/extension/) & [`app/main.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/main.py)
- **What happens:** You read an article or ChatGPT response in your browser. You highlight the paragraph, right-click, and click **"Verify with Evidence"** (or paste it into the Chrome extension).
- **The Code:** The Chrome Extension sends an HTTP POST request with the text to `http://127.0.0.1:8000/verify`.

---

### Step 2: Atomic Claim Decomposition
- **Where:** [`app/claims.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/claims.py)
- **The Problem:** People write compound sentences with multiple facts:
  > *"Aspirin reduces fever and prevents heart attacks."*
- **What happens:** If you search that whole sentence at once, you might not find a paper that discusses both fever and heart attacks in the exact same sentence.
- **The Solution:** We use **spaCy's grammar dependency parser** to break compound sentences into individual **atomic facts**:
  - Fact 1: *"Aspirin reduces fever"*
  - Fact 2: *"Aspirin prevents heart attacks"* (Notice how the second fact automatically inherited the subject *"Aspirin"*!)

---

### Step 3: Domain Detection & Smart Routing
- **Where:** [`app/router.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/router.py)
- **The Problem:** It makes no sense to search medical research (PubMed) for questions about Tesla's car deliveries or India's GDP.
- **What happens:** Our router scans the words in the claim and identifies its subject domain:
  - 🧬 **Biomedical** (drugs, diseases, cells) $\rightarrow$ Queries **PubMed + Crossref + OpenAlex**
  - 🏢 **Financial** (revenue, deliveries, stocks) $\rightarrow$ Queries **SEC EDGAR (official corporate filings)**
  - 🏛 **Government** (GDP, inflation, economy) $\rightarrow$ Queries **World Bank Open Data**
  - 💻 **Computing / Space / General** $\rightarrow$ Queries **Crossref + OpenAlex**

---

### Step 4: Multi-Source Evidence Retrieval & Deduplication
- **Where:** [`app/sources/search_all.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/search_all.py)
- **What happens:**
  1. Python fires search queries to all targeted sources **at the exact same time** using `asyncio.gather()`.
  2. Because different databases might return the exact same research paper, we run **deduplication**: we normalize the **DOI (Digital Object Identifier)** and paper titles to merge duplicates together.

---

### Step 5: Dual-Tier NLP Verification Engine
- **Where:** [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) & [`app/nli_engine.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/nli_engine.py)
- **What happens (The Brain of the System):**

We use a **Dual-Tier verification strategy**:

#### Tier 1: Fast Lexical Sieve ($< 5\text{ms}$)
- **Tokenize & Remove Stopwords:** Drops filler words (*"the", "is", "a"*) so only factual keywords remain.
- **Entity Gate:** Verifies that the document actually mentions the core subject entity (e.g. *"Tesla"* or *"Metformin"*). If an article doesn't even mention the subject, it gets discarded immediately as search noise.
- **Sliding Window Search:** Scans the paper sentence-by-sentence to pick the single best proof quote with the highest keyword overlap.

#### Tier 2: Neural Transformer NLI (DeBERTa)
- If the best proof quote is relevant, we run a **Natural Language Inference (NLI)** cross-encoder model:
  - **Premise:** The sentence from the peer-reviewed paper.
  - **Hypothesis:** The candidate claim from the AI.
- The neural model understands **semantics and synonyms** (even if the claim uses different words like *"lowers body temperature"* instead of *"reduces fever"*).
- It calculates:
  - **Entailment Probability:** How strongly the paper proves the claim ($\ge 65\% \rightarrow$ `SUPPORTED`).
  - **Contradiction Probability:** How strongly the paper disproves the claim ($\ge 65\% \rightarrow$ `CONTRADICTED`).

---

### Step 6: Risk Scoring & Visual Results
- **Where:** [`app/main.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/main.py) & [`extension/popup.js`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/extension/popup.js)
- **What happens:**
  1. **Verdict assigned to each claim:**
     - 🟢 **SUPPORTED:** Authoritative evidence confirms the statement.
     - 🔴 **CONTRADICTED:** Authoritative evidence proves the opposite.
     - 🟡 **INCONCLUSIVE:** No authoritative literature was found to corroborate it.
  2. **Overall Summary Metrics:**
     $$\text{Reliability Score} = \left(\frac{\text{Supported Claims}}{\text{Total Claims}}\right) \times 100$$
     $$\text{Evidence Coverage} = \left(\frac{\text{Claims with Literature Found}}{\text{Total Claims}}\right) \times 100$$
  3. **Hallucination Risk Level:**
     - **HIGH:** If even 1 claim is contradicted, or reliability is $< 40\%$.
     - **MEDIUM:** Reliability between $40\%$ and $75\%$.
     - **LOW:** Reliability $\ge 75\%$ with zero contradictions.
  4. The Chrome Extension displays a clean visual card showing the verdict badge, the domain icon, the exact proof quote from the paper, and a link to view the official record!

---

## 4. Real Walkthrough Example

Let's trace an AI response through the pipeline:

### Input Text:
> *"Metformin is a first-line medication for type 2 diabetes, but smoking cigarettes reduces the risk of lung cancer."*

| Step | Action Taken | Result |
|:---:|:---|:---|
| **1. Input** | Text captured from Chrome | Sent to `/verify` |
| **2. Decomposition** | spaCy splits at the conjunction *"but"* | **Claim 1:** *"Metformin is a first-line medication for type 2 diabetes"*<br>**Claim 2:** *"smoking cigarettes reduces the risk of lung cancer"* |
| **3. Router** | Keywords (`diabetes`, `cancer`) detected | Both mapped to `🧬 biomedical` $\rightarrow$ Queries PubMed + Crossref |
| **4. Retrieval** | Async search across databases | Retrieves top clinical papers on Metformin and Smoking |
| **5. Verification** | Evaluator checks passages & NLI | **Claim 1:** 94% NLI Entailment with clinical paper $\rightarrow$ 🟢 **SUPPORTED**<br>**Claim 2:** Literature proves smoking *increases* cancer $\rightarrow$ 🔴 **CONTRADICTED** |
| **6. Summary** | Main calculates overall metrics | **Reliability:** 50%<br>**Hallucination Risk:** 🔴 **HIGH** (because of the contradiction!) |

---

## 5. Pipeline Component Reference Table

| Pipeline Stage | Python / JS File | Key Technology |
|:---|:---|:---|
| **UI & Capture** | [`extension/popup.js`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/extension/popup.js) | Manifest V3, Chrome Extension API |
| **API Entrypoint** | [`app/main.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/main.py) | FastAPI, Pydantic, CORS Middleware |
| **Claim Splitting** | [`app/claims.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/claims.py) | spaCy (`en_core_web_sm`), Dependency Parsing |
| **Domain Routing** | [`app/router.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/router.py) | Lexical keyword set intersection |
| **Medical Search** | [`app/sources/pubmed.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/pubmed.py) | NCBI E-Utilities API (PubMed) |
| **Academic Search** | [`app/sources/crossref.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/crossref.py) | Crossref REST API (DOI registry) |
| **Knowledge Graph** | [`app/sources/openalex.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/openalex.py) | OpenAlex API + Inverted Index Rebuilder |
| **Corporate Filings**| [`app/sources/edgar.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/edgar.py) | SEC EDGAR Full-Text Search API |
| **Economic Records** | [`app/sources/datagov.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/datagov.py) | World Bank Open Data API |
| **Deduplication** | [`app/sources/search_all.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/sources/search_all.py) | Asyncio, DOI normalization, title hashing |
| **Lexical Evaluator**| [`app/evaluator.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/evaluator.py) | Sliding window passage search, Polarity pairs |
| **Neural Inference** | [`app/nli_engine.py`](file:///c:/Users/Lenovo/Desktop/Detector%20Sherlock/app/nli_engine.py) | DeBERTa Transformer Cross-Encoder |
