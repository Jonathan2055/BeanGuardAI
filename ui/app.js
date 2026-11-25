const state = {
  user: null,
  status: null,
  insights: null,
  tab: "overview",
  authMode: "login",
  prediction: null,
  predictFile: null,
  previewUrl: null,
  files: [],
  zipFile: null,
  uploadLabel: "",
  busy: false,
  message: "",
};

const page = document.querySelector("#page");
const pretty = (value = "") => String(value).replaceAll("_", " ");
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
}[character]));

function uptime(seconds = 0) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { credentials: "same-origin", ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || response.statusText);
  }
  return data;
}

async function refreshData() {
  const [statusResult, insightResult, userResult] = await Promise.allSettled([
    api("/status"),
    api("/insights"),
    api("/auth/me"),
  ]);
  if (statusResult.status === "fulfilled") state.status = statusResult.value;
  if (insightResult.status === "fulfilled") state.insights = insightResult.value;
  state.user = userResult.status === "fulfilled" ? userResult.value : null;
  updateHeader();
}

function updateHeader() {
  const pill = document.querySelector("#health-pill");
  const online = state.status?.status === "online";
  pill.className = `health-pill ${online ? "online" : "offline"}`;
  pill.innerHTML = `<i></i>${online ? `Online · ${uptime(state.status.uptime_seconds)}` : "API unavailable"}`;
  document.querySelector("#account-area").innerHTML = state.user
    ? `<span class="account-chip"><span class="avatar">${escapeHtml(state.user.display_name?.slice(0, 1).toUpperCase())}</span><span class="account-name">${escapeHtml(state.user.display_name)}</span><button class="text-button signout" data-action="logout">Sign out</button></span>`
    : `<button class="button outline compact" data-action="login">Log in</button>`;
  const classes = state.status?.classes || [];
  document.querySelector("#model-label").innerHTML = `
    <span class="eyebrow">MODEL</span>
    <strong>${state.status?.model_available ? `BeanGuard v${escapeHtml(state.status.metrics?.version || 1)}` : "Model unavailable"}</strong>
    <span class="muted small">${escapeHtml(classes.map(pretty).join(" · ") || "Waiting for model metadata")}</span>`;
}

function heading(title, description, button = "") {
  return `<div class="page-heading"><div><span class="eyebrow">BEAN LEAF HEALTH · MONITORING</span><h1>${title}</h1><p class="muted">${description}</p></div>${button}</div>`;
}

function notice(text, type = "") {
  return text ? `<div class="notice ${type}" role="status">${escapeHtml(text)}</div>` : "";
}

function barRows(items, max, format = (value) => value, color = "green") {
  if (!items.length) return '<p class="muted">No values available yet.</p>';
  return `<div class="bar-rows">${items.map(({ name, value }) => `
    <div class="bar-row">
      <span class="bar-label">${escapeHtml(pretty(name))}</span>
      <span class="bar-track"><span class="bar-fill ${color}" style="width:${Math.min(100, (value / (max || 1)) * 100)}%"></span></span>
      <span class="bar-value">${escapeHtml(format(value))}</span>
    </div>`).join("")}</div>`;
}

function runRows(runs) {
  if (!state.user) return '<p class="muted">Log in to view your own retraining history.</p>';
  if (!runs.length) return '<p class="muted">No retraining runs for this account yet.</p>';
  return `<div class="table-scroll"><table><thead><tr><th>Finished</th><th>Images</th><th>Accuracy before</th><th>Accuracy after</th><th>F1 after</th><th>Outcome</th></tr></thead><tbody>
    ${runs.map((run) => `<tr>
      <td>${escapeHtml(run.finished_at || "In progress")}</td>
      <td>${escapeHtml(run.new_images)}</td>
      <td>${run.old_accuracy == null ? "—" : percent(run.old_accuracy)}</td>
      <td>${run.new_accuracy == null ? "—" : percent(run.new_accuracy)}</td>
      <td>${run.new_metrics?.f1 == null ? "—" : percent(run.new_metrics.f1)}</td>
      <td><span class="tag ${run.promoted ? "success-tag" : ""}">${run.promoted ? "MODEL PROMOTED" : "BASELINE KEPT"}</span></td>
    </tr>`).join("")}
    </tbody></table></div>`;
}

function overview() {
  const status = state.status || {};
  const metrics = status.metrics || {};
  const classes = status.classes || [];
  const history = status.recent_runs || [];
  const matrix = metrics.confusion_matrix;
  const matrixTable = matrix
    ? `<div class="table-scroll"><table><thead><tr><th>Actual / Predicted</th>${classes.map((name) => `<th>${escapeHtml(pretty(name))}</th>`).join("")}</tr></thead><tbody>
        ${matrix.map((row, i) => `<tr><th>${escapeHtml(pretty(classes[i]))}</th>${row.map((cell, j) => `<td class="${i === j ? "correct-cell" : ""}">${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}
       </tbody></table></div>`
    : '<p class="muted">Evaluation metrics will appear after training.</p>';
  const metricRows = [["Accuracy", metrics.accuracy], ["Precision · macro", metrics.precision], ["Recall · macro", metrics.recall], ["F1 score · macro", metrics.f1]];
  return `${heading("Model overview", "Monitor availability, evaluation results and your account's retraining activity.", '<button class="button primary" data-tab="predict">New prediction</button>')}
    ${notice(state.message)}
    <div class="stats-grid">
      <article class="stat-card"><span class="eyebrow">SERVICE UPTIME</span><strong>${uptime(status.uptime_seconds)}</strong><span class="muted small">Since this API process started</span></article>
      <article class="stat-card"><span class="eyebrow">MODEL VERSION</span><strong>${metrics.version ? `v${escapeHtml(metrics.version)}` : "—"}</strong><span class="muted small">${escapeHtml(metrics.trained_at || "Training date unavailable")}</span></article>
      <article class="stat-card"><span class="eyebrow">TEST ACCURACY</span><strong>${metrics.accuracy == null ? "—" : percent(metrics.accuracy)}</strong><span class="muted small">${escapeHtml(metrics.images || 0)} held-out images</span></article>
      <article class="stat-card"><span class="eyebrow">MACRO F1</span><strong>${metrics.f1 == null ? "—" : percent(metrics.f1)}</strong><span class="muted small">Average across classes</span></article>
    </div>
    <div class="content-grid">
      <section class="panel wide-panel"><div class="panel-heading"><div><span class="eyebrow">EVALUATION</span><h2>Model quality</h2></div><span class="tag">HELD-OUT TEST SET</span></div>
        <div class="metric-list">${metricRows.map(([name, value]) => `<div class="metric-line"><span>${name}</span><div class="metric-track"><i style="width:${Math.min(100, Number(value || 0) * 100)}%"></i></div><b>${value == null ? "—" : percent(value)}</b></div>`).join("")}</div>
        <h3 class="subheading">Confusion matrix <span class="muted small">Rows = actual; columns = predicted</span></h3>${matrixTable}
      </section>
      <section class="panel"><div class="panel-heading"><div><span class="eyebrow">OPERATIONS</span><h2>System status</h2></div><span class="status-dot ${status.model_available && status.input_check_ready ? "good" : "warn"}"></span></div>
        <div class="status-row"><span>Prediction API</span><b class="${status.status === "online" ? "positive" : "negative"}">${status.status === "online" ? "Online" : "Unavailable"}</b></div>
        <div class="status-row"><span>Classifier</span><b>${status.model_available ? "Ready" : "Missing"}</b></div>
        <div class="status-row"><span>Bean-leaf input check</span><b>${status.input_check_ready ? "Ready" : "Not calibrated"}</b></div>
        <div class="status-row"><span>Retraining</span><b>${escapeHtml(pretty(status.retrain?.state || "idle"))}</b></div>
        ${!status.input_check_ready ? notice("Run the notebook to create input-check statistics before prediction.", "warning") : ""}
        ${state.user ? `<div class="personal-counts"><span>Pending images <b>${escapeHtml(status.pending_uploads ?? 0)}</b></span><span>Your uploads <b>${escapeHtml(status.total_uploads ?? 0)}</b></span></div>` : '<p class="muted small">Log in to see your upload totals and retraining history.</p>'}
      </section>
    </div>
    <section class="panel history-panel"><div class="panel-heading"><div><span class="eyebrow">YOUR ACCOUNT</span><h2>Retraining history</h2></div>${state.user ? '<button class="button outline compact" data-tab="retrain">Upload data</button>' : ""}</div>${runRows(history)}</section>`;
}

function prediction() {
  const result = state.prediction;
  const output = !result
    ? '<div class="empty-result"><span>✳</span><h2>Waiting for an image</h2><p class="muted">The model checks that an image looks like a bean leaf before classifying it.</p></div>'
    : result.error
      ? notice(result.error, "error")
      : `<div class="prediction-label"><span class="result-leaf">✓</span><div><span class="eyebrow">MOST LIKELY CLASS</span><h2>${escapeHtml(pretty(result.class))}</h2></div></div>
         <p class="muted">Confidence: <b>${percent(result.confidence)}</b>${result.uncertain ? " · Try a clearer image." : ""}</p>
         <div class="probability-list">${Object.entries(result.probabilities || {}).map(([name, value]) => `<div><div class="probability-label"><span>${escapeHtml(pretty(name))}</span><b>${percent(value)}</b></div><div class="metric-track"><i style="width:${value * 100}%"></i></div></div>`).join("")}</div>
         <p class="notice warning">For education only; confirm important decisions with an agronomist.</p>`;
  return `${heading("Inspect a leaf", "Choose one clear bean-leaf photo. Images outside the bean-leaf domain are rejected.")}
    <div class="content-grid prediction-grid">
      <section class="panel"><span class="eyebrow">SINGLE IMAGE CLASSIFICATION</span><h2>Upload an image</h2><p class="muted">JPEG or PNG · Maximum 10 MB</p>
        <label class="upload-zone"><input id="predict-file" type="file" accept="image/jpeg,image/png"><span class="upload-symbol">↑</span><strong id="predict-name">${escapeHtml(state.predictFile?.name || "Choose an image")}</strong><small>Choose a clear, close-up photo with one leaf in focus.</small></label>
        <img id="image-preview" class="image-preview" alt="Selected image preview" ${state.previewUrl ? `src="${state.previewUrl}"` : "hidden"}>
        <button class="button primary" data-action="predict" ${state.busy || !state.status?.model_available || !state.status?.input_check_ready ? "disabled" : ""}>${state.busy ? "Checking image…" : "Analyze image"}</button>
        ${!state.status?.input_check_ready ? notice("Input check is not calibrated; run the notebook first.", "warning") : ""}
      </section>
      <section class="panel prediction-result"><span class="eyebrow">PREDICTION RESULT</span>${output}</section>
    </div>`;
}

function insights() {
  const classes = state.status?.classes || [];
  const trainCounts = state.insights?.counts?.train || {};
  const colors = state.insights?.colours || {};
  const f1 = state.status?.metrics?.per_class_f1 || {};
  const maxCount = Math.max(1, ...Object.values(trainCounts));
  const maxF1 = Math.max(1, ...Object.values(f1));
  const colourItems = ["red", "green", "blue"].map((channel) => ({
    name: channel,
    value: classes.length ? classes.reduce((sum, name) => sum + Number(colors[name]?.[channel] || 0), 0) / classes.length : 0,
  }));
  const splitCounts = state.insights?.counts || {};
  return `${heading("Data insights", "Explore class balance, image colour and test performance.")}
    <div class="insight-grid">
      <section class="panel insight-card"><span class="eyebrow">FEATURE 01 · DATA COVERAGE</span><h2>Class balance</h2><p class="muted">Training images in each class.</p>
        ${barRows(classes.map((name) => ({ name, value: trainCounts[name] || 0 })), maxCount)}
        <p class="interpretation"><b>Interpretation:</b> similar class counts reduce bias from unequal class representation; per-class scores still matter.</p>
      </section>
      <section class="panel insight-card"><span class="eyebrow">FEATURE 02 · PIXEL COLOUR</span><h2>Average RGB intensity</h2><p class="muted">Mean channel values across training images, from 0 to 255.</p>
        ${barRows(colourItems, 255, (value) => value.toFixed(1), "blue")}
        ${classes.map((name) => `<div class="colour-line"><span>${escapeHtml(pretty(name))}</span><span>${["red", "green", "blue"].map((channel) => `<i title="${channel}: ${escapeHtml(colors[name]?.[channel] ?? "—")}" style="background-color:${channel};opacity:${Math.max(0.15, Number(colors[name]?.[channel] || 0) / 255)}"></i>`).join("")}</span></div>`).join("")}
        <p class="interpretation"><b>Interpretation:</b> colour can reflect yellowing or rust-like tones, but lighting and background also affect RGB. The model must learn leaf texture and lesions too.</p>
      </section>
      <section class="panel insight-card"><span class="eyebrow">FEATURE 03 · CLASS-WISE ERROR</span><h2>Per-class F1</h2><p class="muted">Lower scores show which classes need more review.</p>
        ${barRows(classes.map((name) => ({ name, value: f1[name] || 0 })), maxF1, percent, "gold")}
        <p class="interpretation"><b>Interpretation:</b> the lowest F1 class is the current challenge; inspect its precision, recall and confusion-matrix errors before retraining.</p>
      </section>
      <section class="panel insight-card"><span class="eyebrow">DATASET SPLITS</span><h2>Images by split</h2>
        <div class="split-list">${["train", "val", "test"].map((split) => `<div><span>${split}</span><b>${Object.values(splitCounts[split] || {}).reduce((sum, value) => sum + value, 0)}</b></div>`).join("")}</div>
        <p class="interpretation"><b>Interpretation:</b> validation supports model selection; the held-out test split is reserved for final evaluation.</p>
      </section>
    </div>`;
}

function authPanel() {
  const registering = state.authMode === "register";
  return `<div class="auth-layout">
    <div class="auth-explainer panel"><span class="eyebrow">PRIVATE TRAINING WORKSPACE</span><h2>Sign in to upload and retrain.</h2><p class="muted">Your uploads and retraining history are linked to your account.</p><div class="auth-points"><span>✓ Passwords are hashed</span><span>✓ Session cookie is protected</span><span>✓ Uploads and run history are account-scoped</span></div></div>
    <form class="auth-form" id="auth-form"><span class="eyebrow">ACCOUNT ACCESS</span><h2>${registering ? "Create your account" : "Welcome back"}</h2>
      ${registering ? '<label>Display name<input name="display_name" autocomplete="name" maxlength="80" required></label>' : ""}
      <label>Email<input name="email" type="email" autocomplete="email" required></label>
      <label>Password<input name="password" type="password" autocomplete="${registering ? "new-password" : "current-password"}" minlength="${registering ? 10 : 1}" maxlength="128" required></label>
      ${notice(state.message, "error")}
      <button class="button primary wide" ${state.busy ? "disabled" : ""}>${state.busy ? "Please wait…" : registering ? "Create account" : "Log in"}</button>
      <p class="muted switch-auth">${registering ? "Already registered?" : "New to BeanGuard?"} <button type="button" class="text-button" data-action="toggle-auth">${registering ? "Log in" : "Create an account"}</button></p>
    </form>
  </div>`;
}

function retrain() {
  const status = state.status || {};
  const classes = status.classes || [];
  const runs = status.recent_runs || [];
  const jobState = status.retrain?.state || "idle";
  const content = !state.user
    ? authPanel()
    : `<div class="content-grid retrain-grid">
        <section class="panel"><span class="eyebrow">STEP 01 · CONTRIBUTE DATA</span><h2>Upload labeled images</h2>
          <p class="muted">Only upload bean-leaf photos with accurate labels. Files are checked before being saved for training.</p>
          <label>Class for selected images<select id="upload-label">${classes.map((name, index) => `<option value="${escapeHtml(name)}" ${state.uploadLabel ? state.uploadLabel === name ? "selected" : "" : index === 0 ? "selected" : ""}>${escapeHtml(pretty(name))}</option>`).join("")}</select></label>
          <label class="field-label">Multiple images · same class<input id="upload-files" type="file" accept="image/jpeg,image/png" multiple></label>
          <div class="divider"><span>OR</span></div>
          <label class="field-label">One ZIP · one folder per class<input id="upload-zip" type="file" accept=".zip,application/zip"></label>
          <p class="small muted">ZIP folders: ${escapeHtml(classes.map(pretty).join(", "))}.</p>
          <p id="upload-selection" class="small muted">${state.zipFile ? escapeHtml(state.zipFile.name) : state.files.length ? `${state.files.length} image(s) selected` : "No files selected."}</p>
          <button class="button primary" data-action="upload" ${state.busy || (!state.zipFile && !state.files.length) ? "disabled" : ""}>${state.busy ? "Saving…" : "Save uploaded data"}</button>
          ${notice(state.message)}
        </section>
        <section class="panel"><span class="eyebrow">STEP 02 · RETRAIN</span><h2>Train from your new examples</h2>
          <div class="queue-number" id="pending-count">${escapeHtml(status.pending_uploads ?? 0)}<span>images waiting in your queue</span></div>
          <p class="muted">Fine-tuning starts from the saved model and uses the original training set plus your uploads. The test set remains separate. Only a candidate with no lower test accuracy replaces the current model.</p>
          <button id="retrain-button" class="button primary" data-action="retrain" ${state.busy || !status.pending_uploads || jobState === "running" ? "disabled" : ""}>${jobState === "running" ? "Retraining in progress…" : "Start retraining"}</button>
          <p class="small muted">Automatic trigger: ${escapeHtml(status.retrain_threshold ?? 30)} pending images. Job state: <b id="job-state">${escapeHtml(pretty(jobState))}</b>.</p>
        </section>
        <section class="panel history-panel"><div class="panel-heading"><div><span class="eyebrow">YOUR ACCOUNT</span><h2>Your model updates</h2></div></div>${runRows(runs)}</section>
      </div>`;
  return `${heading(state.user ? "Improve the model" : "Account access", state.user ? "Upload examples and trigger a private retraining run." : "Log in or register to upload training data.")}${content}`;
}

function render() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("selected", item.dataset.tab === state.tab);
  });
  const views = {
    overview,
    predict: prediction,
    insights,
    retrain,
  };
  page.innerHTML = views[state.tab]();
  updateHeader();
}

function selectTab(tab) {
  state.tab = tab;
  state.message = "";
  render();
}

async function submitPrediction() {
  if (!state.predictFile) {
    state.prediction = { error: "Choose an image first." };
    render();
    return;
  }
  state.busy = true;
  state.prediction = null;
  render();
  try {
    const body = new FormData();
    body.append("file", state.predictFile);
    state.prediction = await api("/predict", { method: "POST", body });
  } catch (error) {
    state.prediction = { error: error.message };
  } finally {
    state.busy = false;
    render();
  }
}

async function submitUpload() {
  const label = document.querySelector("#upload-label")?.value || state.uploadLabel;
  state.busy = true;
  state.message = "";
  render();
  try {
    let response;
    if (state.zipFile) {
      const body = new FormData();
      body.append("file", state.zipFile);
      response = await api("/upload-zip", { method: "POST", body });
    } else {
      if (!state.files.length) throw new Error("Choose images or one class-folder ZIP first.");
      const body = new FormData();
      body.append("label", label);
      state.files.forEach((file) => body.append("files", file));
      response = await api("/upload", { method: "POST", body });
    }
    const rejected = response.rejected?.length || response.broken_files || 0;
    state.message = `Saved ${response.saved} image(s).${rejected ? ` Skipped ${rejected} invalid file(s).` : ""}${response.auto_retrain_triggered ? " Automatic retraining started." : ""}`;
    state.files = [];
    state.zipFile = null;
    await refreshData();
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

async function startRetraining() {
  state.busy = true;
  state.message = "";
  render();
  try {
    const result = await api("/retrain", { method: "POST" });
    state.message = result.message;
    await refreshData();
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

document.addEventListener("click", async (event) => {
  const tabButton = event.target.closest("[data-tab]");
  if (tabButton) {
    selectTab(tabButton.dataset.tab);
    return;
  }
  const button = event.target.closest("[data-action]");
  if (!button) return;
  switch (button.dataset.action) {
    case "login":
      state.tab = "retrain";
      state.authMode = "login";
      render();
      break;
    case "toggle-auth":
      state.authMode = state.authMode === "login" ? "register" : "login";
      state.message = "";
      render();
      break;
    case "predict":
      await submitPrediction();
      break;
    case "upload":
      await submitUpload();
      break;
    case "retrain":
      await startRetraining();
      break;
    case "logout":
      try {
        await api("/auth/logout", { method: "POST" });
        state.user = null;
        state.tab = "overview";
        state.message = "Signed out.";
        await refreshData();
        render();
      } catch (error) {
        state.message = error.message;
        render();
      }
      break;
  }
});

document.addEventListener("change", (event) => {
  const target = event.target;
  if (target.id === "predict-file") {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.predictFile = target.files?.[0] || null;
    state.prediction = null;
    state.previewUrl = state.predictFile ? URL.createObjectURL(state.predictFile) : null;
    document.querySelector("#predict-name").textContent = state.predictFile?.name || "Choose an image";
    const preview = document.querySelector("#image-preview");
    if (state.predictFile) {
      preview.src = state.previewUrl;
      preview.hidden = false;
    } else {
      preview.removeAttribute("src");
      preview.hidden = true;
    }
  } else if (target.id === "upload-files") {
    state.files = Array.from(target.files || []);
    state.zipFile = null;
    document.querySelector("#upload-zip").value = "";
    document.querySelector("#upload-selection").textContent = state.files.length
      ? `${state.files.length} image(s) selected`
      : "No files selected.";
    document.querySelector('[data-action="upload"]').disabled = !state.files.length;
  } else if (target.id === "upload-zip") {
    state.zipFile = target.files?.[0] || null;
    state.files = [];
    document.querySelector("#upload-files").value = "";
    document.querySelector("#upload-selection").textContent = state.zipFile?.name || "No files selected.";
    document.querySelector('[data-action="upload"]').disabled = !state.zipFile;
  } else if (target.id === "upload-label") {
    state.uploadLabel = target.value;
  }
});

document.addEventListener("submit", async (event) => {
  if (event.target.id !== "auth-form") return;
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.target));
  const endpoint = state.authMode === "register" ? "/auth/register" : "/auth/login";
  state.busy = true;
  state.message = "";
  render();
  try {
    state.user = await api(endpoint, { method: "POST", body: JSON.stringify(values) });
    state.message = `Signed in as ${state.user.display_name}.`;
    await refreshData();
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
});

document.addEventListener("DOMContentLoaded", async () => {
  await refreshData();
  render();
  window.setInterval(async () => {
    await refreshData();
    if (state.tab === "overview") {
      render();
    } else if (state.tab === "retrain") {
      const count = document.querySelector("#pending-count");
      if (count && state.user) {
        count.innerHTML = `${escapeHtml(state.status?.pending_uploads ?? 0)}<span>images waiting in your queue</span>`;
        const jobState = state.status?.retrain?.state || "idle";
        document.querySelector("#job-state").textContent = pretty(jobState);
        const retrainButton = document.querySelector("#retrain-button");
        retrainButton.disabled = state.busy || !state.status?.pending_uploads || jobState === "running";
        retrainButton.textContent = jobState === "running" ? "Retraining in progress…" : "Start retraining";
      }
    }
  }, 5000);
});
