/* ============================================================
   RetinaScan — frontend logic
   ============================================================ */

(() => {
  const $ = (id) => document.getElementById(id);

  // State
  const state = {
    models: {},
    selectedModel: null,
    file: null,
    lastResult: null,
    gradcamLoaded: false,
    ddOpen: false,
  };

  // Elements
  const els = {
    dropdownWrap: $('model-dropdown-wrap'),
    infoStrip:    $('model-info-strip'),
    misName:      $('mis-name'),
    misTag:       $('mis-tag'),
    misStats:     $('mis-stats'),
    fileInput: $('file-input'),
    dropzone: $('dropzone'),
    preview: $('preview'),
    previewImg: $('preview-img'),
    previewName: $('preview-name'),
    previewSize: $('preview-size'),
    resetBtn: $('reset-btn'),
    analyzeBtn: $('analyze-btn'),
    stateIdle: $('state-idle'),
    stateAnalyzing: $('state-analyzing'),
    stateResult: $('state-result'),
    stateError: $('state-error'),
    scannerImg: $('scanner-img'),
    resultModel: $('result-model'),
    resultTitle: $('result-title'),
    resultRec: $('result-rec'),
    resultNote: $('result-note'),
    confidenceRing: $('confidence-ring'),
    confidencePct: $('confidence-pct'),
    probList: $('prob-list'),
    gradcamBtn: $('gradcam-btn'),
    gradcamBtnLabel: $('gradcam-btn-label'),
    gradcamView: $('gradcam-view'),
    gradcamOriginal: $('gradcam-original'),
    gradcamHeatmap: $('gradcam-heatmap'),
    newBtn: $('new-btn'),
    errorTitle: $('error-title'),
    errorMessage: $('error-message'),
    errorRetry: $('error-retry'),
    themeToggle: $('theme-toggle'),
  };

  // ---------- Init ----------
  document.addEventListener('DOMContentLoaded', async () => {
    loadTheme();
    els.themeToggle.addEventListener('click', toggleTheme);

    await loadModels();
    wireDropzone();
    els.resetBtn.addEventListener('click', resetUpload);
    els.analyzeBtn.addEventListener('click', runPredict);
    els.gradcamBtn.addEventListener('click', toggleGradcam);
    els.newBtn.addEventListener('click', resetAll);
    els.errorRetry.addEventListener('click', resetAll);
  });

  // ---------- Theme ----------
  function loadTheme() {
    const saved = localStorage.getItem('rs_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
  }
  function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('rs_theme', next);
  }

  // ---------- Models ----------
  async function loadModels() {
    try {
      const resp = await fetch('/api/models');
      const data = await resp.json();
      state.models = data.models || {};
      renderDropdown();
    } catch (err) {
      els.dropdownWrap.innerHTML = `<p style="color:var(--text-dim)">Failed to load models: ${err.message}</p>`;
    }
  }

  function renderDropdown() {
    const keys = Object.keys(state.models);
    if (!keys.length) {
      els.dropdownWrap.innerHTML = `<p style="color:var(--text-dim)">No models configured.</p>`;
      return;
    }

    // Determine pre-selection: URL param > localStorage > first available
    const urlParam = new URLSearchParams(window.location.search).get('model');
    const lsKey    = localStorage.getItem('rs_preselect_model');
    localStorage.removeItem('rs_preselect_model'); // consume once
    const preselect = [urlParam, lsKey].find(k => k && state.models[k]?.available)
                   || keys.find(k => state.models[k].available)
                   || null;

    // Build dropdown HTML
    els.dropdownWrap.innerHTML = `
      <div class="model-dropdown" id="model-dropdown">
        <button type="button" class="model-dropdown__trigger" id="dd-trigger"
                aria-haspopup="listbox" aria-expanded="false">
          <div class="dd-trigger__info">
            <div class="dd-trigger__name" id="dd-trigger-name">Select a model</div>
            <div class="dd-trigger__sub"  id="dd-trigger-sub">Choose one of the trained architectures</div>
          </div>
          <svg class="dd-chevron" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" stroke-width="2.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        <ul class="model-dropdown__menu" id="dd-menu" role="listbox">
          ${keys.map(key => {
            const m = state.models[key];
            const valAcc = (m.val_accuracy * 100).toFixed(1);
            return `
              <li class="model-dropdown__item${m.available ? '' : ' is-disabled'}"
                  data-key="${escapeHtml(key)}" role="option"
                  aria-disabled="${!m.available}"
                  title="${m.available ? '' : 'Weights not yet uploaded'}">
                <div class="ddi__left">
                  <span class="ddi__name">${escapeHtml(m.name)}</span>
                  <span class="ddi__fw">${escapeHtml(m.framework)}</span>
                </div>
                <div class="ddi__right">
                  ${m.best ? `<span class="ddi__best">★</span>` : ''}
                  ${m.available
                    ? `<span class="ddi__acc">${valAcc}%</span>`
                    : `<span class="ddi__noweights">No weights</span>`}
                </div>
              </li>`;
          }).join('')}
        </ul>
      </div>`;

    // Wire trigger
    const trigger = $('dd-trigger');
    const menu    = $('dd-menu');
    trigger.addEventListener('click', () => toggleDropdown(trigger, menu));

    // Wire items
    menu.querySelectorAll('.model-dropdown__item:not(.is-disabled)').forEach(item => {
      item.addEventListener('click', () => {
        selectModel(item.dataset.key);
        closeDropdown(trigger, menu);
      });
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!$('model-dropdown')?.contains(e.target)) closeDropdown(trigger, menu);
    }, true);

    // Keyboard: Escape closes
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeDropdown(trigger, menu);
    });

    if (preselect) selectModel(preselect);
  }

  function toggleDropdown(trigger, menu) {
    const open = trigger.getAttribute('aria-expanded') === 'true';
    open ? closeDropdown(trigger, menu) : openDropdown(trigger, menu);
  }
  function openDropdown(trigger, menu) {
    trigger.setAttribute('aria-expanded', 'true');
    menu.classList.add('is-open');
    $('model-dropdown')?.classList.add('is-open');
  }
  function closeDropdown(trigger, menu) {
    trigger.setAttribute('aria-expanded', 'false');
    menu.classList.remove('is-open');
    $('model-dropdown')?.classList.remove('is-open');
  }

  function selectModel(key) {
    state.selectedModel = key;
    const m = state.models[key];
    if (!m) return;

    // Update trigger text
    const nameEl = $('dd-trigger-name');
    const subEl  = $('dd-trigger-sub');
    if (nameEl) nameEl.textContent = m.name;
    if (subEl)  subEl.textContent  = m.framework;

    // Highlight selected item
    document.querySelectorAll('.model-dropdown__item').forEach(li => {
      li.classList.toggle('is-selected', li.dataset.key === key);
      li.setAttribute('aria-selected', String(li.dataset.key === key));
    });

    // Info strip
    if (els.misName) els.misName.textContent = m.name;
    if (els.misTag)  els.misTag.textContent  = m.tagline;
    if (els.misStats) {
      const valAcc  = (m.val_accuracy * 100).toFixed(1);
      const macroF1 = (m.macro_f1  ?? 0).toFixed(2);
      const wtF1    = (m.weighted_f1 ?? 0).toFixed(2);
      els.misStats.innerHTML = `
        <span class="stat stat--accent"><strong>${valAcc}%</strong>Val Acc</span>
        <span class="stat"><strong>${macroF1}</strong>Macro F1</span>
        <span class="stat"><strong>${wtF1}</strong>Wtd F1</span>`;
    }
    els.infoStrip?.classList.remove('hidden');

    updateAnalyzeButton();
  }

  // ---------- Upload ----------
  function wireDropzone() {
    els.fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    ['dragenter', 'dragover'].forEach(evt =>
      els.dropzone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        els.dropzone.classList.add('is-dragover');
      })
    );
    ['dragleave', 'drop'].forEach(evt =>
      els.dropzone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        els.dropzone.classList.remove('is-dragover');
      })
    );
    els.dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files);
    });
  }

  function handleFiles(files) {
    if (!files || !files.length) return;
    const file = files[0];
    const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!allowed.includes(file.type) && !/\.(jpe?g|png|bmp|tiff?|webp)$/i.test(file.name)) {
      showError('Unsupported file type', 'Please upload a JPG, PNG, BMP, TIFF, or WEBP image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError('File too large', 'Maximum size is 10 MB.');
      return;
    }

    state.file = file;
    els.previewName.textContent = file.name;
    els.previewSize.textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type || 'image'}`;

    const reader = new FileReader();
    reader.onload = (e) => {
      els.previewImg.src = e.target.result;
      els.scannerImg.src = e.target.result;
      els.gradcamOriginal.src = e.target.result;
      els.preview.classList.remove('hidden');
      updateAnalyzeButton();
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    state.file = null;
    els.fileInput.value = '';
    els.preview.classList.add('hidden');
    updateAnalyzeButton();
  }

  function updateAnalyzeButton() {
    els.analyzeBtn.disabled = !(state.file && state.selectedModel);
  }

  // ---------- Predict ----------
  async function runPredict() {
    if (!state.file || !state.selectedModel) return;
    setState('analyzing');

    const fd = new FormData();
    fd.append('image', state.file);
    fd.append('model', state.selectedModel);

    try {
      const resp = await fetch('/api/predict', { method: 'POST', body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      state.lastResult = data;
      renderResult(data);
      setState('result');
    } catch (err) {
      showError('Prediction failed', err.message);
    }
  }

  function renderResult(r) {
    const modelMeta = state.models[r.model] || {};
    els.resultModel.textContent = modelMeta.name || r.model;
    els.resultTitle.textContent = r.class_name;
    els.resultTitle.style.color = r.severity_color || '';
    els.resultRec.textContent = r.recommendation || '';
    els.resultNote.textContent = r.note || '';

    // Confidence ring
    const pct = Math.round((r.confidence || 0) * 100);
    const circ = 2 * Math.PI * 52; // r=52
    els.confidenceRing.style.strokeDasharray = `${circ}`;
    // start full, animate to remaining
    els.confidenceRing.style.strokeDashoffset = `${circ}`;
    els.confidenceRing.style.stroke = r.severity_color || 'var(--primary)';
    requestAnimationFrame(() => {
      els.confidenceRing.style.strokeDashoffset = `${circ * (1 - pct / 100)}`;
    });

    // Animate the % number
    animateNumber(els.confidencePct, 0, pct, 1100);

    // Probability bars
    els.probList.innerHTML = '';
    const probs = r.all_probs || [];
    const names = r.class_names || [];
    const topIdx = r.class_idx;
    probs.forEach((p, i) => {
      const pctI = (p * 100).toFixed(1);
      const li = document.createElement('li');
      li.className = 'prob' + (i === topIdx ? ' is-top' : '');
      li.innerHTML = `
        <span class="prob__name">${escapeHtml(names[i] || `Class ${i}`)}</span>
        <span class="prob__bar"><span class="prob__fill" style="background: ${getSeverityColor(i)}"></span></span>
        <span class="prob__pct">${pctI}%</span>
      `;
      els.probList.appendChild(li);
      requestAnimationFrame(() => {
        li.querySelector('.prob__fill').style.width = `${p * 100}%`;
      });
    });

    // Reset gradcam state
    state.gradcamLoaded = false;
    els.gradcamView.classList.add('hidden');
    els.gradcamBtnLabel.textContent = 'Show Grad-CAM';
    els.gradcamBtn.disabled = false;
  }

  function getSeverityColor(i) {
    const colors = ['#27AE60', '#F1C40F', '#E67E22', '#E74C3C', '#922B21'];
    return colors[i] || '#1ABC9C';
  }

  // ---------- GradCAM ----------
  async function toggleGradcam() {
    if (state.gradcamLoaded) {
      els.gradcamView.classList.toggle('hidden');
      els.gradcamBtnLabel.textContent = els.gradcamView.classList.contains('hidden')
        ? 'Show Grad-CAM' : 'Hide Grad-CAM';
      return;
    }
    if (!state.lastResult) return;

    els.gradcamBtn.disabled = true;
    els.gradcamBtnLabel.textContent = 'Generating…';

    try {
      const resp = await fetch('/api/gradcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: state.lastResult.model,
          image_filename: state.lastResult.image_filename,
          class_idx: state.lastResult.class_idx,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      els.gradcamHeatmap.src = data.gradcam_url + `?t=${Date.now()}`;
      els.gradcamOriginal.src = state.lastResult.image_url;
      els.gradcamView.classList.remove('hidden');
      state.gradcamLoaded = true;
      els.gradcamBtnLabel.textContent = 'Hide Grad-CAM';
    } catch (err) {
      els.gradcamBtnLabel.textContent = 'Show Grad-CAM';
      alert(`Grad-CAM failed: ${err.message}`);
    } finally {
      els.gradcamBtn.disabled = false;
    }
  }

  // ---------- State machine ----------
  function setState(name) {
    [els.stateIdle, els.stateAnalyzing, els.stateResult, els.stateError]
      .forEach(el => el.classList.remove('state--active'));
    const map = {
      idle: els.stateIdle,
      analyzing: els.stateAnalyzing,
      result: els.stateResult,
      error: els.stateError,
    };
    map[name]?.classList.add('state--active');
    if (name === 'result' || name === 'error') {
      setTimeout(() => map[name].scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    }
  }

  function resetAll() {
    resetUpload();
    state.lastResult = null;
    state.gradcamLoaded = false;
    setState('idle');
  }

  function showError(title, message) {
    els.errorTitle.textContent = title;
    els.errorMessage.textContent = message;
    setState('error');
  }

  // ---------- Helpers ----------
  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = Math.round(from + (to - from) * eased);
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
})();
