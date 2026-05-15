let selectedModel = null;

// Toast Notification
function toast(msg, ms=2800) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}

// Set Loading State
function setLoading(id) {
  const el = document.getElementById(id); if (!el) return;
  el.innerHTML = `<div class="loading"><div class="spinner"></div> Loading...</div>`;
}

// Escape HTML
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// Load Models List from API
async function loadModels() {
  const data = await fetch('/api/models_list').then(r=>r.json());
  const sel  = document.getElementById('deployModelSelect');
  const noCard = document.getElementById('noModelCard');

  if (!data.models?.length) {
    if (noCard) noCard.style.display = 'block';
    return;
  }

  data.models.slice().reverse().forEach(m => {
    const opt = document.createElement('option');
    opt.value       = m.model_path;
    opt.textContent = `${m.model_name} — Acc: ${(m.accuracy*100).toFixed(1)}% — ${m.timestamp}`;
    opt.dataset.modelId   = m.model_id;
    opt.dataset.modelName = m.model_name;
    opt.dataset.accuracy  = m.accuracy;
    opt.dataset.f1        = m.f1_score;
    opt.dataset.feats     = JSON.stringify(m.features || []);
    sel.appendChild(opt);
  });
}

// Handle Model Selection
async function onModelSelect(modelPath) {
  const sel = document.getElementById('deployModelSelect');
  const opt = sel.options[sel.selectedIndex];

  if (!modelPath || !opt.value) {
    document.getElementById('modelInfo').style.display  = 'none';
    document.getElementById('inputCard').style.display  = 'none';
    document.getElementById('resultCard').style.display = 'none';
    selectedModel = null;
    return;
  }

  const stateData = await fetch('/api/columns').then(r=>r.json());
  const features  = stateData.feature_cols || JSON.parse(opt.dataset.feats || '[]');

  selectedModel = {
    model_path: modelPath,
    model_id:   opt.dataset.modelId,
    model_name: opt.dataset.modelName,
    accuracy:   parseFloat(opt.dataset.accuracy || 0),
    f1:         parseFloat(opt.dataset.f1 || 0),
    features:   features,
  };

  document.getElementById('modelInfoName').textContent = selectedModel.model_name;
  document.getElementById('modelInfoGrid').innerHTML = `
    <div class="model-info-item">
      <div class="model-info-val">${(selectedModel.accuracy*100).toFixed(2)}%</div>
      <div class="model-info-label">Accuracy</div>
    </div>
    <div class="model-info-item">
      <div class="model-info-val">${(selectedModel.f1*100).toFixed(2)}%</div>
      <div class="model-info-label">F1 Score</div>
    </div>
    <div class="model-info-item">
      <div class="model-info-val">${selectedModel.features.length}</div>
      <div class="model-info-label">Features</div>
    </div>`;
  document.getElementById('modelInfo').style.display = 'block';

  buildFeatureInputs(selectedModel.features);
  document.getElementById('inputCard').style.display  = 'block';
  document.getElementById('resultCard').style.display = 'none';
}

// Build Feature Input Fields
function buildFeatureInputs(features) {
  const grid = document.getElementById('featureInputs');
  grid.innerHTML = '';

  const badge = document.getElementById('featCountBadge');
  if (badge) badge.textContent = `${features.length} features`;

  features.forEach(feat => {
    const div = document.createElement('div');
    div.className = 'feature-input-item';
    div.innerHTML = `
      <label>${esc(feat)}</label>
      <input type="number" id="feat_${esc(feat)}" placeholder="0" step="any" value="0"/>`;
    grid.appendChild(div);
  });
}

// Run Prediction
async function runPredict() {
  if (!selectedModel) { toast('Select a model first'); return; }
  if (!selectedModel.features.length) { toast('No features found for this model'); return; }

  const input_data = {};
  let hasError = false;
  selectedModel.features.forEach(feat => {
    const el = document.getElementById(`feat_${feat}`);
    if (!el) { hasError = true; return; }
    input_data[feat] = parseFloat(el.value) || 0;
  });
  if (hasError) { toast('Some feature inputs not found'); return; }

  setLoading('predResult');
  document.getElementById('resultCard').style.display = 'block';

  const data = await fetch('/api/predict', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      model_path: selectedModel.model_path,
      model_id:   selectedModel.model_id,
      input_data: input_data,
    })
  }).then(r=>r.json());

  if (data.error) {
    document.getElementById('predResult').innerHTML =
      `<div style="color:#e53e3e;font-weight:600;padding:20px">❌ ${esc(data.error)}</div>`;
    return;
  }

  const pred  = data.prediction;
  const feats = data.features_used || selectedModel.features;

  document.getElementById('predResult').innerHTML = `
    <div class="pred-value-box">
      <div class="pred-value-label">PREDICTION</div>
      <div class="pred-value">${pred}</div>
      <div class="pred-value-sub">Class / Label</div>
    </div>
    <div class="pred-features-used">
      <b>Features used (${feats.length}):</b><br>
      <div style="margin-top:6px">
        ${feats.map(f => `<span class="pred-feat-tag">${esc(f)}: ${input_data[f]??0}</span>`).join('')}
      </div>
    </div>`;
}

// Initialize on Page Load
document.addEventListener('DOMContentLoaded', loadModels);