/* ============================================================
   RetinaScan — Models page logic
   ============================================================ */

(() => {
  // ---- Theme ----
  const savedTheme = localStorage.getItem('rs_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('theme-toggle').addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('rs_theme', next);
    });
  });

  // ---- Load & render ----
  document.addEventListener('DOMContentLoaded', async () => {
    const grid = document.getElementById('cards-grid');
    try {
      const resp = await fetch('/api/models');
      const data = await resp.json();
      renderCards(data.models || {}, grid);
    } catch (e) {
      grid.innerHTML = `<p style="color:var(--text-dim);padding:24px">Failed to load models: ${e.message}</p>`;
    }
  });

  function renderCards(models, grid) {
    grid.innerHTML = '';
    const order = ['vit', 'swin', 'inceptionresnetv2', 'two_stage'];
    const keys = order.filter(k => models[k]).concat(
      Object.keys(models).filter(k => !order.includes(k))
    );

    keys.forEach((key) => {
      const m = models[key];
      if (!m) return;
      const valAcc  = (m.val_accuracy  * 100).toFixed(1);
      const macroF1 = (m.macro_f1  ?? 0).toFixed(2);
      const wtF1    = (m.weighted_f1 ?? 0).toFixed(2);

      const card = document.createElement('article');
      card.className = 'mcard' + (m.available ? '' : ' mcard--unavail');
      if (m.best) card.classList.add('mcard--best');

      card.innerHTML = `
        <div class="mcard__badges">
          ${m.best      ? `<span class="mcard__badge mcard__badge--best">★ Best Accuracy</span>` : ''}
          ${!m.available ? `<span class="mcard__badge mcard__badge--none">Weights Not Uploaded</span>` : `<span class="mcard__badge mcard__badge--ready">Ready</span>`}
        </div>
        <div class="mcard__head">
          <div>
            <h2 class="mcard__name">${esc(m.name)}</h2>
            <div class="mcard__fw">${esc(m.framework)}</div>
          </div>
        </div>
        <p class="mcard__tag">${esc(m.tagline)}</p>
        <div class="mcard__stats">
          <div class="mcard__stat mcard__stat--accent">
            <strong>${valAcc}%</strong>
            <span>Val Accuracy</span>
          </div>
          <div class="mcard__stat">
            <strong>${macroF1}</strong>
            <span>Macro F1</span>
          </div>
          <div class="mcard__stat">
            <strong>${wtF1}</strong>
            <span>Weighted F1</span>
          </div>
        </div>
        <div class="mcard__foot">
          ${m.available
            ? `<button class="btn btn--primary mcard__select-btn" data-key="${esc(key)}" type="button">
                 Select for Prediction
                 <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M5 12h14m0 0l-6-6m6 6l-6 6" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
               </button>`
            : `<div class="mcard__hint">Drop weights in <code>backend/models/</code> to enable</div>`
          }
        </div>
      `;

      if (m.available) {
        card.querySelector('.mcard__select-btn').addEventListener('click', () => {
          localStorage.setItem('rs_preselect_model', key);
          window.location.href = '/';
        });
      }
      grid.appendChild(card);
    });
  }

  function esc(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
})();
