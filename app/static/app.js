const $ = (selector) => document.querySelector(selector);
const elements = {
  input: $("#image-input"), chooseButton: $("#choose-image-button"),
  replaceButton: $("#replace-image-button"), analyseButton: $("#analyse-button"),
  newSessionButton: $("#start-new-session-button"), captureHeading: $("#capture-heading"),
  resultHeading: $("#result-heading"), emptyState: $("#empty-state"),
  previewState: $("#preview-state"), preview: $("#image-preview"),
  imageName: $("#image-name"), progress: $("#progress-panel"),
  requestError: $("#request-error"), resultPlaceholder: $("#result-placeholder"),
  status: $("#result-status"), measuredPanel: $("#measured-panel"),
  rejectedPanel: $("#rejected-panel"), diagnosticPanel: $("#diagnostic-panel"),
  baselineDiagnosticImage: $("#baseline-diagnostic-image"),
  diagnosticLink: $("#open-diagnostic"), baselineSessionPanel: $("#baseline-session-panel"),
  baselineSessionId: $("#baseline-session-id"), baselineCreatedAt: $("#baseline-created-at"),
  comparisonPanel: $("#comparison-panel"),
  comparisonBaselineImage: $("#comparison-baseline-diagnostic-image"),
  followUpDiagnosticImage: $("#follow-up-diagnostic-image"), followUpCount: $("#follow-up-count"),
  researchIndicator: $("#research-indicator"),
  researchIndicatorLabel: $("#research-indicator-label"),
  researchIndicatorMessage: $("#research-indicator-message"),
  researchIndicatorThreshold: $("#research-indicator-threshold"),
  followUpGuidance: $("#follow-up-guidance"),
  acquisitionConfirmation: $("#acquisition-confirmation"),
  confirmationHelp: $("#confirmation-help"),
  settingsForm: $("#settings-form"),
  settingsApplyNote: $("#settings-apply-note"),
  settingsError: $("#settings-error"),
  settingsSuccess: $("#settings-success"),
  settingsSaveButton: $("#settings-save-button"),
  settingsResetButton: $("#settings-reset-button"),
  sessionsLoading: $("#sessions-loading"),
  sessionsEmpty: $("#sessions-empty"),
  sessionsError: $("#sessions-error"),
  sessionList: $("#session-list"),
  sessionsRefreshButton: $("#sessions-refresh-button"),
};

const acquisitionChecks = Array.from(
  document.querySelectorAll(".acquisition-check"),
);

let selectedFile = null;
let previewUrl = null;
let activeSessionId = null;
const viewNames = ["analyse", "sessions", "settings", "guide"];

function activateView(viewName, updateHash = true) {
  const selected = viewNames.includes(viewName) ? viewName : "analyse";
  for (const name of viewNames) {
    const panel = $(`#${name}-view`);
    const active = name === selected;
    setHidden(panel, !active);
    document.querySelectorAll(`[data-view="${name}"]`).forEach((control) => {
      control.classList.toggle("active", active);
      if (control.getAttribute("role") === "tab") {
        control.setAttribute("aria-selected", String(active));
        control.tabIndex = active ? 0 : -1;
      }
    });
  }
  if (updateHash) location.hash = selected;
  if (selected === "settings") loadSettings();
  if (selected === "sessions") loadSessions();
}

function setHidden(element, hidden) { element.hidden = hidden; }
function diagnosticUrl(result) {
  return result?.diagnostic_available && result?.diagnostic_url
    ? `${result.diagnostic_url}?v=${Date.now()}` : null;
}
function showRequestError(message) {
  elements.requestError.textContent = message;
  setHidden(elements.requestError, false);
}
function updateAnalysisReadiness() {
  const ready = Boolean(selectedFile)
    && acquisitionChecks.every((checkbox) => checkbox.checked);
  elements.analyseButton.disabled = !ready;
  elements.confirmationHelp.textContent = ready
    ? "Image confirmed and ready for analysis."
    : "Confirm all three items to enable analysis.";
  elements.acquisitionConfirmation.classList.toggle("confirmation-ready", ready);
}
function resetAcquisitionConfirmation() {
  acquisitionChecks.forEach((checkbox) => { checkbox.checked = false; });
  updateAnalysisReadiness();
}
function clearSelectedImage() {
  selectedFile = null;
  elements.input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  elements.preview.removeAttribute("src");
  elements.imageName.textContent = "";
  setHidden(elements.previewState, true);
  setHidden(elements.emptyState, false);
  resetAcquisitionConfirmation();
}
function resetForNewSession() {
  if (activeSessionId && !window.confirm("Start a new session and leave the active baseline?")) return;
  activeSessionId = null;
  clearSelectedImage();
  setHidden(elements.resultPlaceholder, false);
  setHidden(elements.measuredPanel, true);
  setHidden(elements.rejectedPanel, true);
  setHidden(elements.diagnosticPanel, true);
  setHidden(elements.comparisonPanel, true);
  setHidden(elements.requestError, true);
  setHidden(elements.baselineSessionPanel, true);
  setHidden(elements.researchIndicator, true);
  setHidden(elements.followUpGuidance, true);
  elements.captureHeading.textContent = "Choose a baseline image";
  elements.resultHeading.textContent = "Baseline result";
  elements.chooseButton.textContent = "Choose image";
  elements.analyseButton.textContent = "Establish baseline";
  elements.status.textContent = "Waiting for image";
  elements.status.className = "status-badge status-idle";
  $("#technical-settings").textContent = "—";
  $("#technical-session-id").textContent = "—";
  $("#technical-timestamp").textContent = "—";
  elements.settingsApplyNote.textContent = "Changes apply to new baseline sessions.";
}
function prepareFollowUpCapture() {
  clearSelectedImage();
  elements.captureHeading.textContent = "Choose a follow-up image";
  elements.resultHeading.textContent = "Latest comparison";
  elements.chooseButton.textContent = "Choose follow-up image";
  elements.analyseButton.textContent = "Compare with baseline";
  setHidden(elements.followUpGuidance, false);
}
function selectImage(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    showRequestError("Please choose a JPG or PNG image.");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.preview.src = previewUrl;
  elements.imageName.textContent = file.name;
  setHidden(elements.emptyState, true);
  setHidden(elements.previewState, false);
  setHidden(elements.progress, true);
  setHidden(elements.requestError, true);
  resetAcquisitionConfirmation();
}
function setMetric(selector, value, suffix = "") {
  $(selector).textContent = value == null ? "Not available" : `${value}${suffix}`;
}
function showMeasurement(result) {
  setHidden(elements.resultPlaceholder, true);
  setHidden(elements.rejectedPanel, true);
  setHidden(elements.measuredPanel, false);
  $("#external-length").textContent = Number(result.external_length_cm).toFixed(3);
  setMetric("#pivc-detected", result.pivc_detected ? "Yes" : "No");
  setMetric("#pivc-confidence", result.pivc_confidence == null ? null : Number(result.pivc_confidence).toFixed(4));
  setMetric("#marks-detected", result.marks_detected);
  setMetric("#centreline-length", result.corrected_centreline_px == null ? null : Number(result.corrected_centreline_px).toFixed(2), " px");
  setMetric("#pixels-per-cm", result.pixels_per_cm == null ? null : Number(result.pixels_per_cm).toFixed(2), " px/cm");
  const spacings = Array.isArray(result.consecutive_mark_spacings_px)
    ? result.consecutive_mark_spacings_px.map((value) => `${Number(value).toFixed(2)} px`).join(", ") : "";
  setMetric("#mark-spacings", spacings || null);
}
function showBaselineDiagnostic(result) {
  const url = diagnosticUrl(result);
  if (!url) { setHidden(elements.diagnosticPanel, true); return; }
  elements.baselineDiagnosticImage.src = url;
  elements.diagnosticLink.href = result.diagnostic_url;
  setHidden(elements.diagnosticPanel, false);
}
function showRejected(result, followUp = false) {
  setHidden(elements.resultPlaceholder, true);
  if (!followUp) setHidden(elements.measuredPanel, true);
  setHidden(elements.rejectedPanel, false);
  elements.status.textContent = followUp ? "Follow-up rejected" : "Measurement rejected";
  elements.status.className = "status-badge status-rejected";
  $("#rejection-reason").textContent = result.rejection_reason || "The image could not produce a safe measurement.";
  $("#rejection-stage").textContent = result.rejection_stage ? `Rejected during: ${result.rejection_stage}` : "";
}
function showBaselineSession(result, makeActive = true) {
  if (makeActive) activeSessionId = result.session_id;
  elements.baselineSessionId.textContent = result.session_id;
  const createdAt = new Date(result.created_at);
  elements.baselineCreatedAt.textContent = Number.isNaN(createdAt.getTime()) ? result.created_at : createdAt.toLocaleString();
  setHidden(elements.baselineSessionPanel, false);
  $("#technical-session-id").textContent = result.session_id;
  $("#technical-timestamp").textContent = result.created_at;
  const settings = result.settings;
  $("#technical-settings").textContent = settings
    ? `conf ${settings.confidence}, IoU ${settings.iou}, tolerance ±${settings.tolerance_cm} cm, ${settings.imgsz}px`
    : "Not available";
  elements.settingsApplyNote.textContent = "An active session keeps its original settings. Changes apply to the next baseline.";
}
function showResearchIndicator(indicator) {
  if (!indicator) {
    setHidden(elements.researchIndicator, true);
    return;
  }

  elements.researchIndicatorLabel.textContent = indicator.label;
  elements.researchIndicatorMessage.textContent = indicator.message;
  elements.researchIndicatorThreshold.textContent =
    `Research tolerance: ±${Number(indicator.threshold_cm).toFixed(2)} cm`;

  const classByCode = {
    WITHIN_TOLERANCE: "indicator-stable",
    OUTWARD_INCREASE: "indicator-increase",
    INWARD_DECREASE: "indicator-decrease",
    UNABLE_TO_ASSESS: "indicator-unavailable",
  };

  elements.researchIndicator.className =
    `research-indicator ${classByCode[indicator.code] || "indicator-neutral"}`;

  const symbolByCode = {
    WITHIN_TOLERANCE: "✓",
    OUTWARD_INCREASE: "↑",
    INWARD_DECREASE: "↓",
    UNABLE_TO_ASSESS: "?",
  };

  const symbol = elements.researchIndicator.querySelector(
    ".indicator-symbol",
  );

  symbol.textContent = symbolByCode[indicator.code] || "↔";

  setHidden(elements.researchIndicator, false);
}
function showComparison(result) {
  const signed = Number(result.comparison.signed_change_cm);
  $("#baseline-length-comparison").textContent = Number(result.baseline.external_length_cm).toFixed(3);
  $("#follow-up-length-comparison").textContent = Number(result.follow_up.external_length_cm).toFixed(3);
  $("#signed-change").textContent = `${signed > 0 ? "+" : ""}${signed.toFixed(3)}`;
  $("#absolute-change").textContent = Number(result.comparison.absolute_change_cm).toFixed(3);
  elements.followUpCount.textContent = `Follow-up ${result.successful_follow_up_count}`;
  const baselineUrl = diagnosticUrl(result.baseline);
  const followUpUrl = diagnosticUrl(result.follow_up);
  if (baselineUrl) elements.comparisonBaselineImage.src = baselineUrl;
  else elements.comparisonBaselineImage.removeAttribute("src");
  if (followUpUrl) elements.followUpDiagnosticImage.src = followUpUrl;
  else elements.followUpDiagnosticImage.removeAttribute("src");
  setHidden(elements.comparisonPanel, false);
  setHidden(elements.rejectedPanel, true);
  elements.status.textContent = "Follow-up compared";
  elements.status.className = "status-badge status-measured";
  showResearchIndicator(result.research_indicator);
}

async function loadSettings() {
  setHidden(elements.settingsError, true);
  try {
    const response = await fetch("/api/v1/settings");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Settings could not be loaded.");
    $("#setting-confidence").value = payload.editable.confidence;
    $("#setting-iou").value = payload.editable.iou;
    $("#setting-tolerance").value = payload.editable.tolerance_cm;
    $("#runtime-imgsz").textContent = `${payload.runtime.imgsz} px`;
    $("#runtime-device").textContent = payload.runtime.device;
    $("#runtime-model-status").textContent = payload.runtime.model_loaded ? "Loaded" : "Not loaded";
    $("#runtime-model-name").textContent = payload.runtime.model_filename;
    $("#runtime-model-task").textContent = payload.runtime.model_task || "Unavailable";
    $("#runtime-app-version").textContent = payload.runtime.application_version;
  } catch (error) {
    elements.settingsError.textContent = error.message;
    setHidden(elements.settingsError, false);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  elements.settingsSaveButton.disabled = true;
  setHidden(elements.settingsError, true);
  setHidden(elements.settingsSuccess, true);
  const update = {
    confidence: Number($("#setting-confidence").value),
    iou: Number($("#setting-iou").value),
    tolerance_cm: Number($("#setting-tolerance").value),
  };
  try {
    const response = await fetch("/api/v1/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(payload.detail)
        ? payload.detail.map((item) => `${item.loc.at(-1)}: ${item.msg}`).join("; ")
        : payload.detail;
      throw new Error(detail || "Settings were rejected.");
    }
    localStorage.setItem("pivc-settings", JSON.stringify(payload.editable));
    elements.settingsSuccess.textContent = activeSessionId
      ? "Saved. These defaults apply to the next baseline session."
      : "Research defaults saved.";
    setHidden(elements.settingsSuccess, false);
    await loadSettings();
  } catch (error) {
    elements.settingsError.textContent = error.message;
    setHidden(elements.settingsError, false);
  } finally {
    elements.settingsSaveButton.disabled = false;
  }
}

async function resetSettings() {
  if (!window.confirm("Restore confidence 0.50, IoU 0.70 and tolerance 0.10 cm?")) return;
  try {
    const response = await fetch("/api/v1/settings/reset", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Defaults could not be reset.");
    localStorage.setItem("pivc-settings", JSON.stringify(payload.editable));
    await loadSettings();
    elements.settingsSuccess.textContent = "Validated defaults restored.";
    setHidden(elements.settingsSuccess, false);
  } catch (error) {
    elements.settingsError.textContent = error.message;
    setHidden(elements.settingsError, false);
  }
}

function sessionCard(summary) {
  const indicator = summary.latest_indicator?.label || "No follow-up yet";
  const change = summary.latest_signed_change_cm == null
    ? "—" : `${summary.latest_signed_change_cm > 0 ? "+" : ""}${Number(summary.latest_signed_change_cm).toFixed(3)} cm`;
  return `<article class="session-card">
    <div><span class="session-time">${new Date(summary.created_at).toLocaleString()}</span><h3>${Number(summary.baseline_length_cm).toFixed(3)} cm baseline</h3><p>${summary.successful_follow_up_count} follow-up(s) · ${change} · ${indicator}</p></div>
    <div class="session-actions"><button class="button button-secondary" type="button" data-open-session="${summary.session_id}">Open</button><a class="button button-secondary" href="/api/v1/sessions/${summary.session_id}/export.json">JSON</a><a class="button button-secondary" href="/api/v1/sessions/${summary.session_id}/export.csv">CSV</a></div>
  </article>`;
}

async function loadSessions() {
  setHidden(elements.sessionsLoading, false);
  setHidden(elements.sessionsError, true);
  try {
    const response = await fetch("/api/v1/sessions");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Sessions could not be loaded.");
    elements.sessionList.innerHTML = payload.sessions.map(sessionCard).join("");
    setHidden(elements.sessionsEmpty, payload.sessions.length !== 0);
  } catch (error) {
    elements.sessionsError.textContent = error.message;
    setHidden(elements.sessionsError, false);
  } finally {
    setHidden(elements.sessionsLoading, true);
  }
}

async function openSession(sessionId) {
  try {
    const response = await fetch(`/api/v1/sessions/${sessionId}`);
    const session = await response.json();
    if (!response.ok) throw new Error(session.detail || "Session could not be opened.");
    showMeasurement(session.baseline);
    showBaselineDiagnostic(session.baseline);
    showBaselineSession(session, false);
    const latest = session.follow_ups.at(-1);
    if (latest) {
      showComparison({
        baseline: session.baseline,
        follow_up: latest.measurement,
        comparison: latest,
        research_indicator: latest.research_indicator,
        successful_follow_up_count: session.follow_ups.length,
      });
    }
    activateView("analyse");
    if (window.confirm("Continue this session with another follow-up image?")) {
      activeSessionId = session.session_id;
      prepareFollowUpCapture();
    } else {
      activeSessionId = null;
    }
  } catch (error) {
    elements.sessionsError.textContent = error.message;
    setHidden(elements.sessionsError, false);
  }
}

async function analyseImage() {
  if (!selectedFile) { showRequestError("Choose an image before starting the analysis."); return; }
  setHidden(elements.previewState, true);
  setHidden(elements.progress, false);
  setHidden(elements.requestError, true);
  elements.analyseButton.disabled = true;
  const formData = new FormData();
  formData.append("image", selectedFile, selectedFile.name);
  const endpoint = activeSessionId
    ? `/api/v1/sessions/${activeSessionId}/follow-up`
    : "/api/v1/sessions/baseline";
  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    let result;
    try { result = await response.json(); }
    catch { throw new Error("The server returned an unreadable response."); }
    if (!response.ok) throw new Error(result.detail || `Analysis failed with status ${response.status}.`);
    if (result.session_status === "BASELINE_ESTABLISHED") {
      showMeasurement(result.baseline);
      showBaselineDiagnostic(result.baseline);
      showBaselineSession(result);
      elements.status.textContent = "Baseline established";
      elements.status.className = "status-badge status-measured";
      prepareFollowUpCapture();
      loadSessions();
    } else if (result.session_status === "BASELINE_REJECTED") {
      showRejected(result.baseline);
    } else if (result.session_status === "FOLLOW_UP_MEASURED") {
      showMeasurement(result.follow_up);
      showComparison(result);
      prepareFollowUpCapture();
      loadSessions();
    } else if (result.session_status === "FOLLOW_UP_REJECTED") {
      showRejected(result.follow_up, true);
      showResearchIndicator(result.research_indicator);
      prepareFollowUpCapture();
    } else {
      throw new Error("The server returned an unknown session status.");
    }
  } catch (error) {
    showRequestError(error.message || "The image could not be analysed.");
    setHidden(elements.previewState, false);
  } finally {
    setHidden(elements.progress, true);
    updateAnalysisReadiness();
  }
}

elements.chooseButton.addEventListener("click", () => elements.input.click());
elements.replaceButton.addEventListener("click", () => elements.input.click());
elements.analyseButton.addEventListener("click", analyseImage);
elements.newSessionButton.addEventListener("click", resetForNewSession);
elements.input.addEventListener("change", (event) => selectImage(event.target.files?.[0]));
elements.settingsForm.addEventListener("submit", saveSettings);
elements.settingsResetButton.addEventListener("click", resetSettings);
elements.sessionsRefreshButton.addEventListener("click", loadSessions);
elements.sessionList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-open-session]");
  if (button) openSession(button.dataset.openSession);
});
document.querySelectorAll("[data-view]").forEach((control) => {
  control.addEventListener("click", () => activateView(control.dataset.view));
});
document.querySelectorAll('[role="tab"]').forEach((tab) => {
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const current = viewNames.indexOf(tab.dataset.view);
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const next = viewNames[(current + offset + viewNames.length) % viewNames.length];
    activateView(next);
    $(`#${next}-tab`).focus();
  });
});
window.addEventListener("hashchange", () => activateView(location.hash.slice(1), false));
acquisitionChecks.forEach((checkbox) => {
  checkbox.addEventListener("change", updateAnalysisReadiness);
});
for (const eventName of ["dragenter", "dragover"]) {
  elements.emptyState.addEventListener(eventName, (event) => {
    event.preventDefault(); elements.emptyState.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.emptyState.addEventListener(eventName, (event) => {
    event.preventDefault(); elements.emptyState.classList.remove("dragging");
  });
}
elements.emptyState.addEventListener("drop", (event) => selectImage(event.dataTransfer?.files?.[0]));
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/static/service-worker.js").catch(() => {});
  });
}
activateView(location.hash.slice(1) || "analyse", false);
