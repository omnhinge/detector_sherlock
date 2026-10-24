# Detector Sherlock — System Architecture & Data Flow Specification (for tldraw)

**Document Version:** 1.0  
**Target Visualizer:** ChatGPT / tldraw MCP Plugin / `@tldraw/editor`  
**Project:** Detector Sherlock (AI Response Factuality and Hallucination Scoring System)

---

## 1. Executive System Overview

**Detector Sherlock** is an evidence-grounded AI hallucination detection and claim verification engine. It decomposes unstructured LLM outputs into candidate atomic claims, routes each claim to authoritative research repositories (PubMed, Crossref, OpenAlex), reconstructs scientific literature abstracts, extracts matching evidence passages, and applies directional polarity analysis to render evidence-backed verdicts (`supported`, `contradicted`, `unverified`).

### System Tiers & Component Taxonomy:
- **Client Tier (`blue`)**: Manifest V3 Chrome Extension (`popup.html/js`, background worker, context menus) and Swagger/REST clients.
- **Gateway & Ingestion Tier (`violet` / `light-violet`)**: FastAPI ASGI application with CORS middleware, Pydantic schema validation, and concurrent dispatch orchestrator.
- **Decomposition & Routing Tier (`light-violet`)**:
  - `claims.py`: Sentence boundary detection and length-based fragment pruning.
  - `router.py`: Domain categorization heuristics (Biomedical, Computing, Space, General) controlling source targeting.
- **Evidence Retrieval & Ingestion Tier (`light-blue` & `grey`)**:
  - `search_all.py`: Asynchronous concurrent fan-out via `asyncio.gather`.
  - `pubmed.py`: E-Utilities integration (`esearch.fcgi` & `efetch.fcgi`).
  - `crossref.py`: Works API metadata and DOI resolver.
  - `openalex.py`: Works API and topological inverted-index abstract reconstruction.
  - Deduplication Engine: DOI canonicalization with normalized title fallback.
- **Evaluation & Scoring Engine (`yellow` & `orange`)**:
  - `evaluator.py`: Lexical keyword distillation, sentence-level sliding-window passage matching, directional polarity-flip detection, and confidence calculation.
- **Verdict & Synthesis Tier (`green` / `red` / `yellow`)**:
  - Reliability scoring ($0-100\%$), discrete hallucination risk classification (`low`, `medium`, `high`), evidence coverage calculation, and explainability payloads.

---

## 2. End-to-End Data Flow Pipeline

1. **Input Ingestion**: User highlights text or pastes LLM response into the Chrome Extension. Payload: `{ "response": str }`.
2. **API Dispatch**: `POST /verify` receives the payload, validates constraints (1–20,000 characters) via `VerifyRequest`.
3. **Atomic Claim Extraction**: `extract_claims()` tokenizes input into candidate sentences and removes non-substantive fragments ($<15$ chars). Returns: `list[str]`.
4. **Domain Classification**: For each claim, `detect_domain()` extracts vocabulary intersections against domain dictionaries and selects optimal target sources (`get_sources_for_domain()`).
5. **Concurrent Evidence Retrieval**:
   - `search_all_sources()` dispatches parallel HTTP GET queries to target sources via `httpx.AsyncClient`.
   - OpenAlex abstracts are topologically reconstructed from inverted position indexes.
   - Cross-source records are merged and deduplicated using normalized DOIs and titles.
6. **Passage-Level Semantic Matching & Polarity Check**:
   - Candidate claims are stripped of stopwords.
   - Paper abstracts are split into sentences; each sentence is evaluated for keyword intersection ratio.
   - Highest-scoring sentence is extracted as `best_evidence.snippet`.
   - `check_for_contradiction()` analyzes assertion polarity between claim and snippet/title across antagonistic keyword sets (e.g., `reduces/prevents` vs. `causes/increases`).
7. **Verdict Resolution**:
   - `contradicted`: Overlap $\ge 0.35$ with opposing semantic polarity.
   - `supported`: Overlap $\ge 0.60$ with corroborating passage.
   - `unverified`: Overlap $< 0.60$ or no records found.
8. **Aggregation & Metrics Formatting**:
   - Computes `reliability_score = (supported / total) * 100`.
   - Assesses `hallucination_risk` (`high` if any contradiction or score $<40\%$).
   - Returns complete structured report to the Chrome Extension popup.

---

## 3. tldraw Canvas Blueprint

### A. Coordinate Grid & Component Hierarchy

| Frame / Group | Component ID | Label | Subtext / Tech | Color | Coordinates `[X, Y, W, H]` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Frame 1: Clients** | `client_ext` | **Chrome Extension** | Manifest V3 / Popup / ContextMenu | `blue` | `[60, 160, 220, 100]` |
| | `client_docs` | **Swagger / REST Client** | OpenAPI 3.1 / HTTP Client | `blue` | `[60, 290, 220, 80]` |
| **Frame 2: Gateway** | `api_gateway` | **FastAPI Gateway** | `app/main.py` + CORS Middleware | `violet` | `[340, 200, 240, 140]` |
| **Frame 3: Analysis** | `claim_extractor` | **Claim Decomposer** | `app/claims.py` (Sentence Splitter) | `light-violet` | `[640, 100, 240, 90]` |
| | `domain_router` | **Domain Router** | `app/router.py` (Biomed/Comp/Space) | `light-violet` | `[640, 240, 240, 90]` |
| **Frame 4: Retrieval** | `search_aggregator` | **Source Aggregator** | `app/sources/search_all.py` (asyncio) | `light-blue` | `[940, 180, 260, 100]` |
| | `src_pubmed` | **PubMed API** | E-Utilities NCBI (Biomedical) | `grey` | `[1260, 80, 190, 65]` |
| | `src_crossref` | **Crossref Works API** | DOI & Metadata Registry | `grey` | `[1260, 175, 190, 65]` |
| | `src_openalex` | **OpenAlex API** | Works API + Abstract Reconstructor | `grey` | `[1260, 270, 190, 75]` |
| | `dedup_engine` | **Deduplication Engine** | DOI Normalizer & Title Hash Fallback | `light-blue` | `[940, 320, 260, 80]` |
| **Frame 5: Evaluation**| `eval_engine` | **Sherlock Evaluator** | `app/evaluator.py` (Passage Matcher) | `yellow` | `[640, 460, 260, 120]` |
| | `polarity_checker` | **Contradiction Detector**| Directional Polarity Flip Matrix | `orange` | `[940, 460, 260, 120]` |
| **Frame 6: Verdicts** | `verdict_supported` | **Verdict: SUPPORTED** | Green Badge / Corroborating Passage | `green` | `[340, 660, 210, 70]` |
| | `verdict_contradicted`| **Verdict: CONTRADICTED**| Red Badge / Elevated Hallucination Risk | `red` | `[590, 660, 210, 70]` |
| | `verdict_unverified` | **Verdict: UNVERIFIED** | Amber Badge / Insufficient Coverage | `yellow` | `[840, 660, 210, 70]` |
| | `summary_metrics` | **Summary Card** | Reliability Score + Risk + Caveat | `violet` | `[60, 460, 220, 120]` |

---

### B. Directed Arrows & Data Payloads

| Arrow ID | Source Component | Target Component | Label (Payload Traveling on Edge) |
| :--- | :--- | :--- | :--- |
| `e1` | `client_ext` | `api_gateway` | `POST /verify { response }` |
| `e2` | `client_docs` | `api_gateway` | `POST /verify { response }` |
| `e3` | `api_gateway` | `claim_extractor` | `Raw Response String` |
| `e4` | `claim_extractor` | `domain_router` | `Candidate Claims list[str]` |
| `e5` | `domain_router` | `search_aggregator` | `Claim Query + Targeted Sources` |
| `e6` | `search_aggregator` | `src_pubmed` | `async search_pubmed(query)` |
| `e7` | `search_aggregator` | `src_crossref` | `async search_crossref(query)` |
| `e8` | `search_aggregator` | `src_openalex` | `async search_openalex(query)` |
| `e9` | `src_pubmed` | `dedup_engine` | `Raw Medical Records` |
| `e10` | `src_crossref` | `dedup_engine` | `Metadata & DOIs` |
| `e11` | `src_openalex` | `dedup_engine` | `Reconstructed Abstracts` |
| `e12` | `dedup_engine` | `eval_engine` | `Deduplicated Canonical Evidence Records` |
| `e13` | `claim_extractor` | `eval_engine` | `Claim Assertion & Keywords` |
| `e14` | `eval_engine` | `polarity_checker` | `Extracted Best Snippet vs Claim Terms` |
| `e15` | `polarity_checker` | `eval_engine` | `Contradiction Flag & Opposing Assertions` |
| `e16` | `eval_engine` | `verdict_supported` | `Keyword Overlap >= 60%` |
| `e17` | `eval_engine` | `verdict_contradicted`| `Opposing Polarity Detected (Risk: HIGH)` |
| `e18` | `eval_engine` | `verdict_unverified` | `Overlap < 60% / No Evidence` |
| `e19` | `eval_engine` | `summary_metrics` | `Claim Verdicts & Confidence Scores` |
| `e20` | `summary_metrics` | `client_ext` | `JSON Verification Report & Visual Cards` |

---

## 4. tldraw MCP / SDK Programmatic Creation Script

```typescript
// Detector Sherlock - tldraw Canvas Generation Script

export function renderSherlockArchitecture(editor: any) {
  // 1. Define Components
  const shapes = [
    // Clients
    { id: 'client_ext', x: 60, y: 160, w: 220, h: 100, text: 'Chrome Extension (V3)\n[Popup & Context Menu]', color: 'blue' },
    { id: 'client_docs', x: 60, y: 290, w: 220, h: 80, text: 'Swagger UI / REST\n[API Documentation]', color: 'blue' },
    
    // Gateway
    { id: 'api_gateway', x: 340, y: 200, w: 240, h: 140, text: 'FastAPI Gateway\n(app/main.py)\n[CORS, Pydantic, Dispatch]', color: 'violet' },
    
    // Claims & Routing
    { id: 'claim_extractor', x: 640, y: 100, w: 240, h: 90, text: 'Claim Decomposer\n(app/claims.py)\n[Sentence Splitter]', color: 'light-violet' },
    { id: 'domain_router', x: 640, y: 240, w: 240, h: 90, text: 'Domain Router\n(app/router.py)\n[Biomed/Comp/Space Filter]', color: 'light-violet' },
    
    // Aggregator & Sources
    { id: 'search_aggregator', x: 940, y: 180, w: 260, h: 100, text: 'Source Aggregator\n(search_all.py)\n[asyncio.gather Fan-out]', color: 'light-blue' },
    { id: 'src_pubmed', x: 1260, y: 80, w: 190, h: 65, text: 'PubMed E-Utilities\n[Biomedical Papers]', color: 'grey' },
    { id: 'src_crossref', x: 1260, y: 175, w: 190, h: 65, text: 'Crossref Works API\n[DOI & Metadata]', color: 'grey' },
    { id: 'src_openalex', x: 1260, y: 270, w: 190, h: 75, text: 'OpenAlex Works API\n[Abstract Reconstructor]', color: 'grey' },
    { id: 'dedup_engine', x: 940, y: 320, w: 260, h: 80, text: 'Deduplication Engine\n[DOI & Title Normalization]', color: 'light-blue' },
    
    // Evaluation Tier
    { id: 'eval_engine', x: 640, y: 460, w: 260, h: 120, text: 'Sherlock Evaluator\n(app/evaluator.py)\n[Passage Search & Overlap]', color: 'yellow' },
    { id: 'polarity_checker', x: 940, y: 460, w: 260, h: 120, text: 'Contradiction Detector\n[Opposing Polarity Matrix]', color: 'orange' },
    
    // Verdicts & Summary
    { id: 'verdict_supported', x: 340, y: 660, w: 210, h: 70, text: 'SUPPORTED\n[Passage Overlap >= 60%]', color: 'green' },
    { id: 'verdict_contradicted', x: 590, y: 660, w: 210, h: 70, text: 'CONTRADICTED\n[Opposing Polarity Claimed]', color: 'red' },
    { id: 'verdict_unverified', x: 840, y: 660, w: 210, h: 70, text: 'UNVERIFIED\n[Insufficient Evidence]', color: 'yellow' },
    { id: 'summary_metrics', x: 60, y: 460, w: 220, h: 120, text: 'Summary Metrics Card\n[Score %, Risk, Coverage]', color: 'violet' }
  ];

  // 2. Add Component Boxes
  shapes.forEach((s) => {
    editor.createShape({
      id: s.id,
      type: 'geo',
      x: s.x,
      y: s.y,
      props: {
        w: s.w,
        h: s.h,
        geo: 'rectangle',
        color: s.color,
        fill: 'semi',
        text: s.text,
        font: 'sans',
        size: 's',
        align: 'middle'
      }
    });
  });

  // 3. Define Connectors with Edge Labels
  const connectors = [
    { from: 'client_ext', to: 'api_gateway', label: 'POST /verify' },
    { from: 'client_docs', to: 'api_gateway', label: 'POST /verify' },
    { from: 'api_gateway', to: 'claim_extractor', label: 'Raw text' },
    { from: 'claim_extractor', to: 'domain_router', label: 'Claims list[str]' },
    { from: 'domain_router', to: 'search_aggregator', label: 'Targeted Sources' },
    { from: 'search_aggregator', to: 'src_pubmed', label: 'E-Search' },
    { from: 'search_aggregator', to: 'src_crossref', label: 'Query Works' },
    { from: 'search_aggregator', to: 'src_openalex', label: 'Inverted Index' },
    { from: 'src_pubmed', to: 'dedup_engine', label: 'Articles' },
    { from: 'src_crossref', to: 'dedup_engine', label: 'Metadata' },
    { from: 'src_openalex', to: 'dedup_engine', label: 'Abstracts' },
    { from: 'dedup_engine', to: 'eval_engine', label: 'Clean Evidence' },
    { from: 'claim_extractor', to: 'eval_engine', label: 'Claim Text' },
    { from: 'eval_engine', to: 'polarity_checker', label: 'Snippet vs Claim' },
    { from: 'polarity_checker', to: 'eval_engine', label: 'Polarity Status' },
    { from: 'eval_engine', to: 'verdict_supported', label: 'Corroborated' },
    { from: 'eval_engine', to: 'verdict_contradicted', label: 'Refuted' },
    { from: 'eval_engine', to: 'verdict_unverified', label: 'Low Overlap' },
    { from: 'eval_engine', to: 'summary_metrics', label: 'Aggregated Scores' },
    { from: 'summary_metrics', to: 'client_ext', label: 'Render UI Cards' }
  ];

  // 4. Create Arrows
  connectors.forEach((c, idx) => {
    editor.createShape({
      id: `arrow_${idx}`,
      type: 'arrow',
      props: {
        start: { type: 'binding', boundShapeId: c.from, normalizedAnchor: { x: 0.5, y: 0.5 }, isExact: false },
        end: { type: 'binding', boundShapeId: c.to, normalizedAnchor: { x: 0.5, y: 0.5 }, isExact: false },
        text: c.label,
        size: 's',
        color: 'grey'
      }
    });
  });
}
```
