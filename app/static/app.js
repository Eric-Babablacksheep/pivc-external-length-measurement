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
};

const acquisitionChecks = Array.from(
  document.querySelectorAll(".acquisition-check"),
);

let selectedFile = null;
let previewUrl = null;
let activeSessionId = null;

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
function showBaselineSession(result) {
  activeSessionId = result.session_id;
  elements.baselineSessionId.textContent = result.session_id;
  const createdAt = new Date(result.created_at);
  elements.baselineCreatedAt.textContent = Number.isNaN(createdAt.getTime()) ? result.created_at : createdAt.toLocaleString();
  setHidden(elements.baselineSessionPanel, false);
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
    } else if (result.session_status === "BASELINE_REJECTED") {
      showRejected(result.baseline);
    } else if (result.session_status === "FOLLOW_UP_MEASURED") {
      showMeasurement(result.follow_up);
      showComparison(result);
      prepareFollowUpCapture();
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
