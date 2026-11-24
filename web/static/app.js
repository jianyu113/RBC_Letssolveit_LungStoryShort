"use strict";

const $ = id => document.getElementById(id);
const DISPLAY_THRESHOLD = 0.30;
let items = [];
let selectedKey = null;
let showBoxes = true;
let running = false;
let modelsReady = false;
let toastTimer;
let nextKey = 0;

function notify(message) {
  $("toast").textContent = message;
  $("toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("visible"), 5000);
}

async function requestJSON(url, options) {
  const response = await fetch(url, options);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`The service returned HTTP ${response.status} without a result.`);
  }
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail);
  }
  return body;
}

function updateControls() {
  $("choose").disabled = running;
  $("input-kind").disabled = running;
  $("run").disabled = running || !modelsReady || !items.some(item => !item.result);
  $("run-label").textContent = running ? "Analyzing…" : "Analyze images";
  $("workspace-grid").classList.toggle("processing", running);
  $("list").setAttribute("aria-busy", String(running));
}

async function checkHealth() {
  try {
    const health = await requestJSON("/api/health");
    modelsReady = Boolean(health.ready);
    $("health-label").textContent = modelsReady ? "Both models ready" : "Models not ready";
    $("health-dot").classList.toggle("ready-state", modelsReady);
    $("service-message").replaceChildren();
    $("service-message").hidden = modelsReady;
    if (!modelsReady) {
      $("service-message").append(document.createTextNode(health.detail || "The model weights must be loaded before analysis."));
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "retry-health";
      retry.textContent = "Check again";
      retry.onclick = checkHealth;
      $("service-message").append(retry);
    }
  } catch (error) {
    modelsReady = false;
    $("health-label").textContent = "Inference service unavailable";
    $("service-message").hidden = false;
    $("service-message").textContent = `Unable to reach the inference service. ${error.message}`;
  }
  updateControls();
}

function rankedItems() {
  return [...items].sort((a, b) => {
    const difference = (b.result?.fusion_score ?? -1) - (a.result?.fusion_score ?? -1);
    if (difference) return difference;
    const aId = a.result?.id || a.name;
    const bId = b.result?.id || b.name;
    return aId < bId ? -1 : aId > bId ? 1 : a.key - b.key;
  });
}

function renderList() {
  const list = $("list");
  list.replaceChildren();
  const complete = items.filter(item => item.result).length;
  $("list-count").textContent = `${items.length} image${items.length === 1 ? "" : "s"}`;
  $("sort-state").textContent = complete ? "Highest fusion score first ↓" : "Waiting for analysis";
  $("rank-note").textContent = `${complete} / ${items.length} analyzed`;
  $("ready").textContent = running
    ? `Analyzing ${items.filter(item => item.result || item.error).length} / ${items.length}`
    : `${items.length} image${items.length === 1 ? "" : "s"} selected`;
  if (!items.length) {
    const note = document.createElement("p");
    note.className = "empty-list";
    note.textContent = "Select images to build a review queue.";
    list.append(note);
  }
  rankedItems().forEach((item, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "image-row" + (item.key === selectedKey ? " active" : "");
    row.setAttribute("aria-pressed", String(item.key === selectedKey));
    const preview = item.result?.preview_url || item.localPreview;
    if (preview) {
      const image = document.createElement("img");
      image.className = "thumb";
      image.src = preview;
      image.alt = "";
      row.append(image);
    } else {
      const placeholder = document.createElement("span");
      placeholder.className = "thumb placeholder";
      placeholder.textContent = "DICOM";
      row.append(placeholder);
    }
    const label = document.createElement("div");
    label.className = "row-label";
    const name = document.createElement("div");
    name.className = "row-id";
    name.textContent = item.name;
    name.title = item.name;
    const meta = document.createElement("div");
    meta.className = "row-meta" + (item.error ? " row-error" : "");
    meta.textContent = item.result ? `Rank ${index + 1} · ${item.result.elapsed_seconds.toFixed(1)} s` : item.error || (item.processing ? "Analyzing…" : "Ready for analysis");
    label.append(name, meta);
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = item.result ? item.result.fusion_score.toFixed(3) : "—";
    row.append(label, score);
    row.onclick = () => {
      selectedKey = item.key;
      renderList();
      renderViewer();
    };
    list.append(row);
  });
  updateControls();
}

function renderViewer() {
  const item = items.find(entry => entry.key === selectedKey);
  const result = item?.result;
  const preview = result?.preview_url || item?.localPreview;
  $("image-wrap").hidden = !preview;
  $("empty-stage").hidden = Boolean(preview);
  $("stage-badge").hidden = !result;
  $("empty-stage").textContent = item
    ? item.error || (item.processing ? "Analyzing image…" : "Preview available after analysis")
    : "Select an image to review";
  if (preview) $("main-image").src = preview;
  else $("main-image").removeAttribute("src");
  $("main-image").alt = item ? `Chest X-ray: ${item.name}` : "Selected chest X-ray";
  $("viewer-id").textContent = item?.name || "No image selected";
  $("viewer-id").title = item?.name || "";
  $("viewer-dim").textContent = result ? `${result.width} × ${result.height} prepared image` : item ? "Not analyzed" : "";
  $("viewer-score").textContent = result ? result.fusion_score.toFixed(3) : "—";
  $("detector-score").textContent = result ? result.detector_score.toFixed(3) : "—";
  $("classifier-score").textContent = result ? result.classifier_score.toFixed(3) : "—";
  $("overlay").replaceChildren();
  const boxes = result?.boxes.filter(box => box.score >= DISPLAY_THRESHOLD) || [];
  if (result && showBoxes) {
    for (const box of boxes) {
      const [x1, y1, x2, y2] = box.xyxy;
      const region = document.createElement("div");
      region.className = "region";
      region.style.left = `${100 * x1 / result.width}%`;
      region.style.top = `${100 * y1 / result.height}%`;
      region.style.width = `${100 * (x2 - x1) / result.width}%`;
      region.style.height = `${100 * (y2 - y1) / result.height}%`;
      const score = document.createElement("span");
      score.className = "box-score";
      score.textContent = box.score.toFixed(2);
      region.append(score);
      $("overlay").append(region);
    }
  }
  $("legend-text").textContent = !result ? "Detector boxes · score ≥ 0.30"
    : !showBoxes ? "Boxes hidden · ranking unchanged"
      : boxes.length ? `${boxes.length} detector box${boxes.length === 1 ? "" : "es"} · score ≥ 0.30`
        : "No detector boxes at score ≥ 0.30";
  $("annotation-note").textContent = result && !boxes.length
    ? "No displayed box does not establish a normal image. Fusion ranks images; the detector supplies unchanged boxes and box scores."
    : "Fusion changes the review order. Box locations and scores come directly from the detector.";
}

function replaceItems(next) {
  items.forEach(item => { if (item.localPreview) URL.revokeObjectURL(item.localPreview); });
  items = next;
  selectedKey = items[0]?.key ?? null;
  $("progress").style.width = "0%";
  document.querySelector(".progressbar").setAttribute("aria-valuenow", "0");
  renderList();
  renderViewer();
}

async function loadFiles(fileList) {
  if (running) return;
  const files = Array.from(fileList);
  if (!files.length) return;
  if (files.length > 20) {
    notify("Choose no more than 20 images per batch.");
    return;
  }
  const inputKind = $("input-kind").value;
  const supported = files.every(file => inputKind === "prepared_png"
    ? /\.png$/i.test(file.name) : /\.dcm$/i.test(file.name) || file.type === "application/dicom");
  if (!supported) {
    notify(inputKind === "prepared_png" ? "Select prepared 512 × 512 PNG files in PNG mode." : "Select .dcm files, or switch the format to prepared PNG.");
    return;
  }
  replaceItems(files.map(file => ({
    key: nextKey++, file, name: file.name, inputKind,
    localPreview: inputKind === "prepared_png" ? URL.createObjectURL(file) : null,
  })));
}

$("run").onclick = async () => {
  if (running || !modelsReady) return;
  running = true;
  updateControls();
  for (const item of items.filter(entry => !entry.result)) {
    item.error = null;
    item.processing = true;
    renderList();
    renderViewer();
    try {
      const data = new FormData();
      data.append("files", item.file);
      data.append("input_kind", item.inputKind);
      const response = await requestJSON("/api/analyze", {method: "POST", body: data});
      if (!response.results?.length) throw new Error("The service did not return a prediction.");
      item.result = response.results[0];
    } catch (error) {
      item.error = error.message;
    }
    item.processing = false;
    const progress = 100 * items.filter(entry => entry.result || entry.error).length / items.length;
    $("progress").style.width = `${progress}%`;
    document.querySelector(".progressbar").setAttribute("aria-valuenow", String(Math.round(progress)));
    renderList();
    renderViewer();
  }
  running = false;
  const ranked = rankedItems();
  selectedKey = ranked[0]?.key ?? null;
  renderList();
  renderViewer();
  const failed = items.filter(item => item.error).length;
  notify(failed ? `${items.length - failed} analyzed; ${failed} failed. Select a failed image for details.` : `${items.length} images analyzed and ranked.`);
};

$("toggle").onclick = () => {
  showBoxes = !showBoxes;
  $("toggle").setAttribute("aria-pressed", String(showBoxes));
  $("toggle-label").textContent = showBoxes ? "Boxes on" : "Boxes off";
  renderViewer();
};
$("choose").onclick = () => $("files").click();
$("files").onchange = event => { loadFiles(event.target.files); event.target.value = ""; };
$("input-kind").onchange = () => {
  const png = $("input-kind").value === "prepared_png";
  $("files").accept = png ? ".png,image/png" : ".dcm,application/dicom";
  $("format-hint").textContent = png ? "512 × 512 PNG from the preparation pipeline" : "Up to 20 DICOM images";
};
for (const eventName of ["dragenter", "dragover"]) {
  $("dropzone").addEventListener(eventName, event => { event.preventDefault(); if (!running) $("dropzone").classList.add("dragging"); });
}
for (const eventName of ["dragleave", "drop"]) {
  $("dropzone").addEventListener(eventName, event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); });
}
$("dropzone").addEventListener("drop", event => loadFiles(event.dataTransfer.files));

async function initialize() {
  renderList();
  renderViewer();
  await checkHealth();
}
initialize();
