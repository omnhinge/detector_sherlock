# Semantic Domain Detection & Hybrid Routing

## Overview

Detector Sherlock's domain routing has been upgraded from a brittle keyword-only lookup to an offline-capable, **hybrid semantic, confidence-aware, multi-domain classification and evidence routing system**.

### Architecture & Routing Flow

```
                                      Claim Text
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    app/domain_classifier.py            │
                      │  - SBERT: all-MiniLM-L6-v2 (lazy load) │
                      │  - Dot product vs multi-facet anchors  │
                      │  - Lexical keyword supplementary boost │
                      │  - Margin & floor confidence analysis  │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
            High Confidence                             Low Confidence
        (Margin >= 0.05, Sim >= 0.18)                (Sim < 0.18 or Failure)
                    │                                           │
       ┌────────────┴────────────┐                              │
       ▼                         ▼                              ▼
  Single Domain             Multi-Domain                Fallback Router
(Top domain clear)     (Runner-up <= 0.10 margin)    (General sources / keyword fallback)
       │                         │                              │
       └────────────┬────────────┘                              │
                    │                                           │
                    ▼                                           ▼
   Targeted Authoritative Sources              Broad Cross-Index Search
  (PubMed, SEC EDGAR, World Bank, etc.)     (Wikipedia, FactCheck, News, Crossref)
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                               Concurrent Evidence Search
                                (app/sources/search_all.py)
                                  Deduplicated by DOI/Title
```

---

## Supported Domain Labels & Connectors

| Domain Label | Target Connectors | Fallback / Academic |
| :--- | :--- | :--- |
| `biomedical` | PubMed, Wikipedia | Crossref, OpenAlex |
| `financial` | SEC EDGAR, News, Wikipedia | Crossref, OpenAlex |
| `government_economic` *(aliases: `government`)* | World Bank / GovData, News, Wikipedia | Crossref |
| `computing` | Wikipedia | Crossref, OpenAlex |
| `space` | Wikipedia | Crossref, OpenAlex |
| `news` | News, FactCheck, Wikipedia | — |
| `general` / `unknown` | Wikipedia, FactCheck, News | Crossref, OpenAlex |

---

## Multi-Domain Routing & Deduplication

When a claim legitimately spans multiple disciplines (e.g., pharmaceutical vaccine earnings bridging `biomedical` and `financial`):
- Both domains are detected when the runner-up score is within `MULTI_DOMAIN_MARGIN = 0.10` and meets `MULTI_DOMAIN_MIN_SCORE = 0.24`.
- `get_sources_for_domains(domains)` aggregates required connectors across both domains (e.g., `["PubMed", "SEC EDGAR", "News", "Wikipedia", "Crossref", "OpenAlex"]`).
- Connectors are strictly deduplicated so that no API or repository is queried redundantly.

---

## Offline Safety & Fallback Mechanics

1. **Lazy Loading**: The model `sentence-transformers/all-MiniLM-L6-v2` is loaded lazily on first access and cached as a thread-safe singleton.
2. **Offline-First**: Defaults to `local_files_only=True` and sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, ensuring execution in sandboxed environments with zero outbound network calls.
3. **Keyword-Only Fallback**: If sentence-transformers or neural model weights are missing or fail to load, the router falls back to `keyword_only_classify(...)`.
4. **General Search Fallback**: Claims that score below the minimum semantic similarity threshold (`MIN_SEMANTIC_THRESHOLD = 0.18`) or exhibit high ambiguity are safely routed to broad encyclopedia, news, and academic indexes (`general`).

---

## Benchmark Evaluation Results

Evaluated on the deterministic 68-claim benchmark (`evaluation/routing_benchmark.json`):

| Metric | Keyword Baseline | Hybrid Semantic | Delta |
| :--- | :--- | :--- | :--- |
| **Single-Domain Accuracy** | 81.03% | **98.28%** | **+17.25%** |
| **Overall Accuracy** | 75.00% | **92.65%** | **+17.65%** |
| **Macro-F1 Score** | 75.53% | **92.91%** | **+17.38%** |
| **Multi-Domain Full Match** | 40.00% | **70.00%** | **+30.00%** |
| **Multi-Domain Any Match** | 90.00% | **100.00%** | **+10.00%** |
| **p95 Latency** | 0.04 ms | **31.47 ms** | CPU inference |

---

## Running Tests and Evaluation

### Run Unit and Integration Tests:
```bash
python -m pytest tests/test_domain_classifier.py tests/test_semantic_router.py -v
```

### Run Benchmark Evaluation Script:
```bash
python evaluation/evaluate_router.py
```
