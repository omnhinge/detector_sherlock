const API_BASE = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
  const responseInput = document.getElementById("responseInput");
  const verifyBtn = document.getElementById("verifyBtn");
  const pasteBtn = document.getElementById("pasteBtn");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const errorMessage = document.getElementById("errorMessage");
  const resultsSection = document.getElementById("resultsSection");

  // Check if text was saved from right-click context menu
  chrome.storage?.local?.get(["pendingVerification"], (result) => {
    if (result && result.pendingVerification) {
      responseInput.value = result.pendingVerification;
      chrome.storage.local.remove("pendingVerification");
      triggerVerification(result.pendingVerification);
    }
  });

  // Paste button
  pasteBtn.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        responseInput.value = text;
        responseInput.focus();
      }
    } catch (err) {
      console.warn("Clipboard access denied or unavailable", err);
    }
  });

  // Verify button click
  verifyBtn.addEventListener("click", () => {
    const text = responseInput.value.trim();
    if (!text) {
      showError("Please enter or paste an AI response to verify.");
      return;
    }
    triggerVerification(text);
  });

  async function triggerVerification(text) {
    clearState();
    loadingState.classList.remove("hidden");
    verifyBtn.disabled = true;

    try {
      const response = await fetch(`${API_BASE}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ response: text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with error status ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error("Verification failed:", err);
      showError(
        `Could not connect to Sherlock backend at ${API_BASE}. Make sure the FastAPI server is running with: python -m uvicorn app.main:app --reload`
      );
    } finally {
      loadingState.classList.add("hidden");
      verifyBtn.disabled = false;
    }
  }

  function renderResults(data) {
    const summary = data.summary || {};
    const claims = data.claims || [];

    // Summary scores
    document.getElementById("reliabilityScore").textContent = Math.round(summary.reliability_score || 0);
    
    const riskBadge = document.getElementById("hallucinationRisk");
    const risk = (summary.hallucination_risk || "unknown").toLowerCase();
    riskBadge.textContent = risk;
    riskBadge.className = `badge ${risk}`;

    document.getElementById("evidenceCoverage").textContent = Math.round(summary.evidence_coverage || 0);
    document.getElementById("countSupported").textContent = summary.supported || 0;
    document.getElementById("countContradicted").textContent = summary.contradicted || 0;

    const inconclusiveCount = summary.inconclusive ?? summary.unverified ?? 0;
    const incEl = document.getElementById("countInconclusive") || document.getElementById("countUnverified");
    if (incEl) incEl.textContent = inconclusiveCount;

    document.getElementById("claimsTotalCount").textContent = claims.length;

    document.getElementById("summaryCaveat").textContent = summary.caveat || "";

    // Claims list
    const claimsList = document.getElementById("claimsList");
    claimsList.innerHTML = "";

    if (claims.length === 0) {
      claimsList.innerHTML = `<p class="state-container">No distinct candidate claims were identified in the text.</p>`;
    } else {
      claims.forEach((c) => {
        const card = createClaimCard(c);
        claimsList.appendChild(card);
      });
    }

    resultsSection.classList.remove("hidden");
  }

  function createClaimCard(claimData) {
    const card = document.createElement("div");
    const rawVerdict = (claimData.verdict || "inconclusive").toLowerCase();
    const verdict = rawVerdict === "unverified" ? "inconclusive" : rawVerdict;
    card.className = `claim-card ${verdict}`;

    const bestEvidence = claimData.best_evidence;
    const hasEvidence = !["unverified", "inconclusive"].includes(verdict) && bestEvidence && bestEvidence.snippet;

    let evidenceHtml = "";
    if (hasEvidence) {
      const sourcesText = bestEvidence.sources ? bestEvidence.sources.join(", ") : "Credible Source";
      const linkHtml = bestEvidence.url
        ? `<a href="${bestEvidence.url}" target="_blank" class="evidence-link">View Record ↗</a>`
        : "";

      evidenceHtml = `
        <div class="evidence-box">
          <p class="evidence-quote">"${escapeHtml(bestEvidence.snippet)}"</p>
          <div class="evidence-meta">
            <span>Source: <strong>${sourcesText}</strong></span>
            ${linkHtml}
          </div>
        </div>
      `;
    } else {
      evidenceHtml = `
        <div class="evidence-box">
          <p class="evidence-quote" style="font-style: normal; color: #94a3b8;">${escapeHtml(claimData.explanation || "No corroborating record found.")}</p>
        </div>
      `;
    }

    const domainIcons = {
      biomedical: "🧬 biomedical",
      computing: "💻 computing",
      space: "🚀 space",
      financial: "🏢 financial",
      government: "🏛 government",
      general: "🌐 general",
    };
    const domainText = domainIcons[claimData.domain] || claimData.domain || "general";

    let nliBadgeHtml = "";
    if (claimData.nli && claimData.nli.available) {
      if (claimData.nli.entailment >= 0.5) {
        nliBadgeHtml = `<span class="domain-tag" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">🧠 NLI Entailment: ${Math.round(claimData.nli.entailment * 100)}%</span>`;
      } else if (claimData.nli.contradiction >= 0.5) {
        nliBadgeHtml = `<span class="domain-tag" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">🧠 NLI Contradiction: ${Math.round(claimData.nli.contradiction * 100)}%</span>`;
      }
    }

    card.innerHTML = `
      <div class="claim-top">
        <span class="claim-badge ${verdict}">${verdict}</span>
        <span class="domain-tag">${escapeHtml(domainText)}</span>
        ${nliBadgeHtml}
      </div>
      <p class="claim-text">${escapeHtml(claimData.claim)}</p>
      ${evidenceHtml}
    `;

    return card;
  }

  function showError(msg) {
    clearState();
    errorMessage.textContent = msg;
    errorState.classList.remove("hidden");
  }

  function clearState() {
    errorState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultsSection.classList.add("hidden");
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
