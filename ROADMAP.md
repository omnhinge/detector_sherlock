# Detector Sherlock — Project Roadmap

## Phase Status Summary

- [x] **Phase 1: Atomic Claim Extraction & NLP Decomposition**
  - spaCy dependency parsing for clause separation and possessive pronoun coreference resolution.
  - Granular numerical grounding constraints.

- [x] **Phase 2: Public Record & Academic Connectors**
  - PubMed, Crossref, OpenAlex connectors.
  - SEC EDGAR company filings (10-K, 10-Q).
  - World Bank and data.gov.in macroeconomic indicators.
  - Wikipedia, FactCheck, and Real-Time Google News RSS feeds.

- [x] **Phase 3: Hybrid Semantic Domain Detection & Multi-Domain Routing**
  - Replaced keyword-only routing with hybrid sentence embedding similarity against multi-facet anchors (`all-MiniLM-L6-v2`).
  - Score separation margin and confidence-aware filtering.
  - Multi-domain detection with non-redundant connector merging.
  - Offline-first execution with automated keyword and general fallbacks.
  - Labeled 68-claim evaluation benchmark with 98.28% single-domain accuracy and 92.91% Macro-F1.

- [ ] **Phase 4: Claim Verdict Calibration & Domain-Adaptive NLI (Future Work)**
  - Fine-tuning cross-encoder thresholds per domain.
  - Temporal consistency verification for dynamic events.
