(function () {
  "use strict";

  const cfg = window.APP_CONFIG;
  const $ = (id) => document.getElementById(id);

  const drop = $("drop");
  const fileInput = $("file");
  const dropInner = $("drop-inner");
  const preview = $("preview");
  const predictBtn = $("predict");
  const clearBtn = $("clear");
  const result = $("result");
  const topLabel = $("top-label");
  const topConf = $("top-conf");
  const bars = $("bars");
  const meta = $("meta");
  const errorBox = $("error");
  const status = $("status");

  let currentFile = null;
  let previewUrl = null;

  // ---- api health ---------------------------------------------------------
  async function checkHealth() {
    try {
      const r = await fetch(`${cfg.apiBase}/health`, { cache: "no-store" });
      const body = await r.json();
      if (!r.ok || body.status !== "ok") throw new Error(body.detail || `HTTP ${r.status}`);
      status.textContent = `API online · classes: ${body.classes.join(", ")}`;
      status.className = "status ok";
    } catch (err) {
      status.textContent = `API unreachable (${cfg.apiBase}) - ${err.message}`;
      status.className = "status bad";
    }
  }

  // ---- file handling ------------------------------------------------------
  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = !msg;
  }

  function setFile(file) {
    showError("");
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("That doesn't look like an image.");
      return;
    }
    if (file.size > cfg.maxFileMb * 1024 * 1024) {
      showError(`Image is too large (max ${cfg.maxFileMb} MB).`);
      return;
    }
    currentFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    dropInner.hidden = true;
    predictBtn.disabled = false;
    clearBtn.disabled = false;
    result.hidden = true;
  }

  function clearAll() {
    currentFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.hidden = true;
    preview.removeAttribute("src");
    dropInner.hidden = false;
    fileInput.value = "";
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    result.hidden = true;
    showError("");
  }

  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
  );
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });

  // paste an image from the clipboard
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  clearBtn.addEventListener("click", clearAll);

  // ---- prediction ---------------------------------------------------------
  function render(data, roundTripMs) {
    topLabel.textContent = data.label;
    topConf.textContent = `${(data.confidence * 100).toFixed(1)}% confident`;
    bars.innerHTML = "";
    for (const p of data.predictions) {
      const li = document.createElement("li");
      li.innerHTML =
        `<span class="name"></span><div class="track"><div class="fill"></div></div><span class="pct"></span>`;
      li.querySelector(".name").textContent = p.label;
      li.querySelector(".pct").textContent = `${(p.confidence * 100).toFixed(1)}%`;
      bars.appendChild(li);
      // set width on the next frame so the css transition plays
      requestAnimationFrame(() => (li.querySelector(".fill").style.width = `${p.confidence * 100}%`));
    }
    meta.textContent = `model inference ${data.inference_ms} ms · round trip ${roundTripMs} ms`;
    result.hidden = false;
  }

  async function classify() {
    if (!currentFile) return;
    showError("");
    predictBtn.disabled = true;
    predictBtn.textContent = "Classifying…";
    const form = new FormData();
    form.append("file", currentFile, currentFile.name);
    const t0 = performance.now();
    try {
      const r = await fetch(`${cfg.apiBase}/predict?top_k=5`, { method: "POST", body: form });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
      render(body, Math.round(performance.now() - t0));
    } catch (err) {
      showError(`Prediction failed: ${err.message}`);
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Classify";
    }
  }

  predictBtn.addEventListener("click", classify);

  checkHealth();
})();
