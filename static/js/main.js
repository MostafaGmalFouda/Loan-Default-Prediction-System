// Application State
const S = {
  orig_columns: [], orig_cat_cols: [], orig_num_cols: [],
  proc_columns: [], proc_cat_cols: [], proc_num_cols: [],
  columns: [], cat_cols: [], num_cols: [],
  num_nan_cols: [], cat_nan_cols: [],
  processed_num_nan_cols: [],
  processed_cat_nan_cols: [],
  processed_num_cols: [],
  applied_steps: [], has_duplicates: false,
  vector_ready: false, processing_done: false,
  data_loaded: false, feature_cols: [], target_col: null,
  diProcessed: false, vizProcessed: false,
  chartType: null, chartColType: null,
  mlModel: 'lr', diAction: null, modalAction: null,
};

let _refreshTimer = null;

// Auto Refresh Columns
function startAutoRefresh() {
  if (_refreshTimer) return;
  _refreshTimer = setInterval(() => { if (S.data_loaded) refreshColNames(); }, 3000);
}

// Navigation
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById('sec-' + name)?.classList.add('active');
  document.querySelectorAll('.topnav button, .topnav a.btn-upload').forEach(b => b.classList.remove('active-nav'));
  const map = { upload:'btnUpload', datainfo:'btnDataInfo', processing:'btnProcessing', visualization:'btnViz', ml:'btnML' };
  document.getElementById(map[name])?.classList.add('active-nav');
  if (name === 'processing') initProcTab();
  if (name === 'visualization') initVizTab();
  if (name === 'ml') initMLTab();
}

function navClick(name) {
  if (!S.data_loaded) { toast('⚠️ Upload data first!'); return; }
  if (name === 'ml' && !S.vector_ready) { toast('⚡ Run VectorAssembler first (Processing tab)'); return; }
  showSection(name);
}

function unlockNav(vecReady) {
  ['btnDataInfo','btnProcessing','btnViz'].forEach(id => {
    const b = document.getElementById(id); if (b) b.disabled = false;
  });
  const mlBtn = document.getElementById('btnML');
  if (mlBtn) {
    mlBtn.disabled = !vecReady;
    mlBtn.classList.toggle('btn-ml-locked', !vecReady);
    mlBtn.textContent = vecReady ? '⚡ ML Models' : '🔒 ML Models';
  }
  if (vecReady) {
    const dep = document.getElementById('btnDeploy');
    if (dep) dep.style.display = 'inline-flex';
  }
}

// Toast Notification
function toast(msg, ms=2800) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}

// Set Loading State
function setLoading(id) {
  const el = document.getElementById(id); if (!el) return;
  el.innerHTML = `<div class="loading"><div class="spinner"></div> Processing with Spark...</div>`;
}

// API Helpers
async function api(url, body = {}) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
  return r.json();
}
async function apiGet(url) { return (await fetch(url)).json(); }
async function postForm(url, fd) { return (await fetch(url, { method:'POST', body:fd })).json(); }

// Update State from API Response
function updateS(data) {
  if (!data) return;
  const keys = ['orig_columns','orig_cat_cols','orig_num_cols',
                 'proc_columns','proc_cat_cols','proc_num_cols',
                 'columns','cat_cols','num_cols','num_nan_cols','cat_nan_cols',
                 'processed_num_nan_cols','processed_cat_nan_cols',
                 'processed_num_cols',
                 'applied_steps','has_duplicates','vector_ready',
                 'processing_done','data_loaded','feature_cols','target_col'];
  keys.forEach(k => { if (data[k] !== undefined) S[k] = data[k]; });

  const btn = document.getElementById('btnRemoveDup');
  if (btn) btn.style.display = S.has_duplicates ? 'block' : 'none';

  updateStepsBadge(S.applied_steps);
  setAfterEnabled(S.processing_done);
  unlockNav(S.vector_ready);
  updateVecInfo(S.feature_cols);
  updateTargetInfo(S.target_col);
}

// Refresh Column Names
async function refreshColNames() {
  const data = await apiGet('/api/col_names');
  if (data) updateS(data);
}

function updateStepsBadge(steps) {
  const bar = document.getElementById('appliedSteps'); if (!bar) return;
  if (!steps?.length) { bar.style.display='none'; return; }
  bar.style.display = 'flex';
  bar.innerHTML = '<b style="font-size:11px;margin-right:6px">Applied:</b>' +
    steps.map(s => `<span class="step-chip">${esc(s.replace(/_/g,' '))}</span>`).join('');
}

function setAfterEnabled(enabled) {
  ['di-tog-after','viz-tog-after'].forEach(id => {
    const b = document.getElementById(id); if (!b) return;
    b.disabled = !enabled;
    if (!enabled) b.classList.remove('active-toggle');
  });
  if (!enabled) {
    S.diProcessed = false; S.vizProcessed = false;
    document.getElementById('di-tog-before')?.classList.add('active-toggle');
    document.getElementById('viz-tog-before')?.classList.add('active-toggle');
  }
}

// File Upload
async function uploadFile(input) {
  const file = input.files[0]; if (!file) return;
  document.getElementById('uploadStatus').innerHTML = '<span style="color:#888">⚡ Loading with Spark...</span>';
  const fd = new FormData(); fd.append('file', file);
  const data = await postForm('/api/upload', fd);
  if (data.error) { document.getElementById('uploadStatus').innerHTML = '❌ ' + data.error; return; }
  updateS(data);
  document.getElementById('uploadStatus').innerHTML =
    `✅ <b>${file.name}</b> — <b>${fmt(data.shape[0])}</b> rows × <b>${data.shape[1]}</b> cols`;
  showSection('datainfo');
  document.getElementById('dataInfoResult').innerHTML = buildTable(data.table);
  startAutoRefresh();
  toast('Data loaded ✅');
}

const uploadBox = document.getElementById('uploadBox');
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.style.borderColor='#1a7abf'; });
uploadBox.addEventListener('dragleave', () => uploadBox.style.borderColor='');
uploadBox.addEventListener('drop', e => {
  e.preventDefault(); uploadBox.style.borderColor='';
  const f = e.dataTransfer.files[0];
  if (f) { const dt=new DataTransfer(); dt.items.add(f); document.getElementById('fileInput').files=dt.files; uploadFile(document.getElementById('fileInput')); }
});

// Table Builder
function buildTable(t) {
  if (!t?.columns) return '<p class="placeholder-text">No data</p>';
  let h = '';
  if (t.total_rows && t.shown_rows < t.total_rows)
    h = `<div class="sample-notice">Showing ${fmt(t.shown_rows)} of ${fmt(t.total_rows)} rows</div>`;
  h += '<div class="data-table-wrap"><table class="data-table"><thead><tr>';
  t.columns.forEach(c => { h += `<th>${esc(c)}</th>`; });
  h += ' hilab</thead><tbody>';
  t.rows.forEach(r => { h += '<tr>'; r.forEach(c => { h += `<td>${esc(String(c))}</td>`; }); h += '</tr>'; });
  return h + '</tbody></table></div>';
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmt(n) { return Number(n).toLocaleString(); }
function showMsg(id, msg, isErr=false) {
  document.getElementById(id).innerHTML =
    `<div class="msg-box ${isErr?'msg-error':''}">${isErr?'❌ ':'✅ '}${esc(msg)}</div>`;
}
function fillSel(id, cols=[], ph='--') {
  const s = document.getElementById(id); if (!s) return;
  s.innerHTML = `<option value="">${ph}</option>`;
  (cols||[]).forEach(c => { const o=document.createElement('option'); o.value=c; o.textContent=c; s.appendChild(o); });
}

// Data Info
async function dataInfoAction(action, btn) {
  S.diAction = action;
  document.querySelectorAll('#sec-datainfo .btn-side').forEach(b=>b.classList.remove('active-side'));
  if (btn) btn.classList.add('active-side');
  setLoading('dataInfoResult');
  const data = await api('/api/data_info', { action, processed: S.diProcessed });
  if (data.error) { showMsg('dataInfoResult', data.error, true); return; }
  if (data.table) document.getElementById('dataInfoResult').innerHTML = buildTable(data.table);
  else if (data.message) { showMsg('dataInfoResult', data.message); updateS(data); }
}

function setDIView(val, btn) {
  if (val && !S.processing_done) { toast('Apply processing first'); return; }
  S.diProcessed = val;
  document.querySelectorAll('#sec-datainfo .toggle-btn').forEach(b=>b.classList.remove('active-toggle'));
  btn.classList.add('active-toggle');
  if (S.diAction) dataInfoAction(S.diAction, null);
}

// Modal
async function openColModal(action) {
  S.modalAction = action;
  document.getElementById('modalTitle').textContent =
    action==='value_counts' ? '📊 Count Val Col — Top 10' : 'Select Column';
  await refreshColNames();
  fillSel('modalColSelect', S.columns, '-- Select Column --');
  document.getElementById('modalOverlay').classList.add('open');
}
function closeModal() { document.getElementById('modalOverlay').classList.remove('open'); }

async function confirmModal() {
  const col = document.getElementById('modalColSelect').value;
  closeModal(); if (!col) return;
  setLoading('dataInfoResult');
  const data = await api('/api/data_info', { action: S.modalAction, column: col, processed: S.diProcessed });
  if (data.error) { showMsg('dataInfoResult', data.error, true); return; }
  if (S.modalAction === 'value_counts') {
    let h = `<div class="msg-box" style="margin-bottom:10px">
      Column: <b>${esc(data.column)}</b> &nbsp;|&nbsp; Total unique: <b>${fmt(data.total_unique)}</b> &nbsp;|&nbsp; Top <b>${data.top_n}</b>
    </div>`;
    document.getElementById('dataInfoResult').innerHTML = h + buildTable(data.table);
  } else {
    document.getElementById('dataInfoResult').innerHTML = buildTable(data.table);
  }
}

// Processing Tab
function onProcType(radio) {
  const selMap = { num_nan:'selNumNan', cat_nan:'selCatNan', encoding:'selEncoding', normalize:'selNormalize', imbalance:'selImbalance' };
  Object.values(selMap).forEach(id => { const el=document.getElementById(id); if(el){el.disabled=true;el.value='';} });
  document.getElementById(selMap[radio.value]).disabled = false;

  let cols = S.columns;
  if (radio.value === 'num_nan') {
    cols = S.processing_done ? (S.processed_num_nan_cols || []) : (S.num_nan_cols || []);
  }
  else if (radio.value === 'cat_nan') {
    cols = S.processing_done ? (S.processed_cat_nan_cols || []) : (S.cat_nan_cols || []);
  }
  else if (radio.value === 'encoding') {
    cols = S.cat_cols;
  }
  else if (radio.value === 'normalize') {
    cols = S.processing_done ? (S.processed_num_cols || S.num_cols) : S.num_cols;
  }
  fillSel('procColSelect', cols, '-- All Relevant Columns --');
}

async function initProcTab() {
  await refreshColNames();
  fillSel('procColSelect', S.columns, '-- All Relevant --');

  const vs = document.getElementById('vecFeatSelect'); if (!vs) return;
  vs.innerHTML = '';
  const numCols = S.processing_done ? S.proc_num_cols : S.orig_num_cols;
  numCols.forEach(c => { const o=document.createElement('option'); o.value=c; o.textContent=c; vs.appendChild(o); });

  const targetCols = S.processing_done ? S.proc_columns : S.orig_columns;
  fillSel('vecTargetSelect', targetCols, '-- Select Target --');
  
  const targetStatus = document.getElementById('targetStatus');
  if (targetStatus) targetStatus.textContent = '';

  const btn = document.getElementById('btnRemoveDup');
  if (btn) btn.style.display = S.has_duplicates ? 'block' : 'none';
  updateVecInfo(S.feature_cols);
}

async function applyProcessing() {
  const radio = document.querySelector('input[name="procType"]:checked');
  if (!radio) { toast('Select a processing type first'); return; }
  const selMap = { num_nan:'selNumNan', cat_nan:'selCatNan', encoding:'selEncoding', normalize:'selNormalize', imbalance:'selImbalance' };
  const action = document.getElementById(selMap[radio.value])?.value;
  if (!action) { toast('Select a method from the dropdown'); return; }
  await applyAction(action);
}

async function applyAction(action) {
  const col = document.getElementById('procColSelect')?.value || null;
  setLoading('procResult');
  const data = await api('/api/processing', { action, column: col||null });
  if (data.error) { showMsg('procResult', data.error, true); return; }
  updateS(data);

  await refreshColNames();
  const vs = document.getElementById('vecFeatSelect');
  if (vs) {
    vs.innerHTML = '';
    const numCols = S.processing_done ? S.proc_num_cols : S.orig_num_cols;
    numCols.forEach(c => { const o=document.createElement('option'); o.value=c; o.textContent=c; vs.appendChild(o); });
  }

  const targetCols = S.processing_done ? S.proc_columns : S.orig_columns;
  fillSel('vecTargetSelect', targetCols, '-- Select Target --');

  document.getElementById('procResult').innerHTML =
    `<div class="msg-box">✅ ${esc(data.message||'Done')} — Shape: <b>${fmt(data.shape[0])} × ${data.shape[1]}</b></div>` +
    buildTable(data.table);
  toast('Applied ✅');
}

// Target Selection
function onTargetSelect() {
  const targetEl = document.getElementById('vecTargetSelect');
  const target = targetEl ? targetEl.value : '';
  const statusEl = document.getElementById('targetStatus');
  
  if (target && target.trim() !== '') {
    statusEl.innerHTML = `<span style="color:#fbbf24">ℹ️ Selected: <b>${esc(target)}</b> (click Set to confirm)</span>`;
  } else {
    statusEl.innerHTML = `<span style="color:#92400e">⚠️ Select target column</span>`;
  }
}

async function setTargetColumn() {
  const targetEl = document.getElementById('vecTargetSelect');
  const target = targetEl ? targetEl.value : '';
  
  if (!target || target.trim() === '') {
    toast('⚠️ Select a target column first');
    return;
  }

  const data = await api('/api/set_target', { target });
  if (data.error) {
    toast('❌ ' + data.error);
    const statusEl = document.getElementById('targetStatus');
    if (statusEl) statusEl.innerHTML = `<span style="color:#dc2626">❌ ${esc(data.error)}</span>`;
    return;
  }

  S.target_col = target;
  updateTargetInfo(target);
  updateFeatDisplay(S.feature_cols, target);
  const statusEl = document.getElementById('targetStatus');
  statusEl.innerHTML = `<span style="color:#065f46">✅ Target set: <b>${esc(target)}</b></span>`;
  toast('✅ Target column set');
}

// VectorAssembler
async function runVectorAssembler() {
  const targetEl = document.getElementById('vecTargetSelect');
  const target = targetEl ? targetEl.value : S.target_col;
  
  if (!target) {
    toast('⚠️ Select target column first (step 1)');
    return;
  }

  const vs = document.getElementById('vecFeatSelect');
  const feats = Array.from(vs.selectedOptions).map(o => o.value);
  if (!feats.length) { toast('Select feature columns (step 2 — hold Ctrl for multiple)'); return; }
  
  S.target_col = target;
  
  setLoading('procResult');
  const data = await api('/api/vector_assemble', { feature_cols: feats });
  if (data.error) { showMsg('procResult', data.error, true); return; }
  updateS(data);
  updateVecInfo(feats);
  updateFeatDisplay(feats, target);
  unlockNav(true);
  document.getElementById('procResult').innerHTML =
    `<div class="msg-box">⚡ VectorAssembler done!<br>
    Target: <b>${esc(target)}</b> → renamed to <b>label</b><br>
    Features (${feats.length}): ${feats.map(f=>`<span class="feat-chip">${esc(f)}</span>`).join(' ')}</div>`;
  toast(`VectorAssembler done ⚡ — ${feats.length} features. ML unlocked!`);
}

function updateVecInfo(feats) {
  const f = feats || S.feature_cols || [];
  const ready = S.vector_ready && f.length > 0;
  const vs = document.getElementById('vecStatus');
  const ti = document.getElementById('vecInfoML');
  if (ready) {
    if (vs) { vs.style.display='block'; vs.textContent=`✅ ${f.length} features assembled`; }
    if (ti) { ti.textContent=`✅ ${f.length} features ready`; ti.style.color='#065f46'; }
  } else {
    if (vs) vs.style.display='none';
    if (ti) { ti.textContent='⚡ Run VectorAssembler last (after all processing)'; ti.style.color='#92400e'; }
  }
}

function updateTargetInfo(target) {
  const t = target || S.target_col || '';
  const ts = document.getElementById('targetStatus');
  const ti = document.getElementById('targetInfoML');

  if (t && t.trim() !== '') {
    if (ts) {
      ts.style.display = 'block';
      ts.innerHTML = `🎯 Target: <b>${esc(t)}</b>`;
      ts.style.color = '#065f46';
    }
    if (ti) {
      ti.innerHTML = `✅ Target ready: <b>${esc(t)}</b>`;
      ti.style.color = '#065f46';
    }
  } else {
    if (ts) {
      ts.style.display = 'block';
      ts.innerHTML = '⚠️ Select target column';
      ts.style.color = '#92400e';
    }
    if (ti) {
      ti.textContent = '⚡ Target not selected';
      ti.style.color = '#92400e';
    }
  }
}

// Visualization Tab
async function initVizTab() {
  await refreshColNames();
  fillSel('vizCol1', S.orig_cat_cols.length ? S.orig_cat_cols : S.orig_columns, '-- Select Column --');
  fillSel('vizCol2', S.orig_num_cols, '-- Select Column 2 --');
}

function setVizView(val, btn) {
  if (val && !S.processing_done) { toast('Apply processing first'); return; }
  S.vizProcessed = val;
  document.querySelectorAll('#sec-visualization .toggle-btn').forEach(b=>b.classList.remove('active-toggle'));
  btn.classList.add('active-toggle');
  loadVizColsByType(S.chartColType);
}

function doChart(type, colType, evt) {
  S.chartType = type; S.chartColType = colType;
  document.querySelectorAll('#sec-visualization .btn-side').forEach(b=>b.classList.remove('active-side'));
  if (evt?.currentTarget) evt.currentTarget.classList.add('active-side');

  const isHeatmap = type === 'heatmap';
  document.getElementById('vizColLabel').style.display = isHeatmap ? 'none' : '';
  document.getElementById('vizCol1').style.display = isHeatmap ? 'none' : '';
  document.getElementById('vizCol2').style.display = (type==='scatter') ? '' : 'none';

  if (!isHeatmap) loadVizColsByType(colType);
}

function loadVizColsByType(colType) {
  const isAfter = S.vizProcessed && S.processing_done;

  let cols;
  if (colType === 'cat') cols = isAfter ? S.proc_cat_cols : S.orig_cat_cols;
  else if (colType === 'num') cols = isAfter ? S.proc_num_cols : S.orig_num_cols;
  else cols = isAfter ? S.proc_columns : S.orig_columns;

  const lbl = document.getElementById('vizColLabel');
  if (lbl) lbl.textContent = colType==='cat' ? 'Categorical Column' : colType==='num' ? 'Numerical Column' : 'Column';

  fillSel('vizCol1', cols, '-- Select Column --');

  const num2 = isAfter ? S.proc_num_cols : S.orig_num_cols;
  fillSel('vizCol2', num2, '-- Select Column 2 --');
}

async function applyViz() {
  if (!S.chartType) { toast('Select a chart type'); return; }
  const col = document.getElementById('vizCol1')?.value || null;
  const col2 = document.getElementById('vizCol2')?.value || null;
  if (S.chartType !== 'heatmap' && !col) { toast('Select a column'); return; }
  setLoading('vizResult');
  const data = await api('/api/visualization', {
    chart_type: S.chartType, column: col, column2: col2, processed: S.vizProcessed
  });
  if (data.error) { showMsg('vizResult', data.error, true); return; }
  document.getElementById('vizResult').innerHTML =
    `<img class="viz-img" src="data:image/png;base64,${data.image}" alt="${S.chartType}"/>`;
}

// ML Tab
async function initMLTab() {
  await refreshColNames();
  updateFeatDisplay(S.feature_cols, S.target_col);
  updateVecInfo(S.feature_cols);
  updateTargetInfo(S.target_col);
  loadScoresHistory();
}

function updateFeatDisplay(feats, target) {
  let h = '';
  if (feats?.length) h += `<div><b>Features (${feats.length}):</b><div class="feat-list">${feats.map(f=>`<span class="feat-chip">${esc(f)}</span>`).join('')}</div></div>`;
  if (target) h += `<div style="margin-top:8px"><b>Target:</b> <span class="target-chip">${esc(target)}</span></div>`;
  document.getElementById('mlFeaturesDisplay').innerHTML = h || '<p class="placeholder-text">Features and Target will appear here</p>';
}

function selectML(id, btn) {
  S.mlModel = id;
  document.querySelectorAll('.ml-tab').forEach(b=>b.classList.remove('active-mltab'));
  btn.classList.add('active-mltab');
}

async function trainModel() {
  if (!S.vector_ready) { toast('⚡ Run VectorAssembler first'); return; }
  setLoading('mlResultDisplay');
  const data = await api('/api/train', { model_id: S.mlModel });
  if (data.error) { showMsg('mlResultDisplay', data.error, true); return; }

  let html = `
    <div class="result-header">
      <b>${esc(data.model)}</b>
      <span class="elapsed-badge">⏱ ${data.elapsed_s}s</span>
    </div>
    <div class="ml-scores">
      <div class="score-card"><div class="score-val">${(data.accuracy*100).toFixed(2)}%</div><div class="score-label">Accuracy</div></div>
      <div class="score-card"><div class="score-val">${(data.f1_score*100).toFixed(2)}%</div><div class="score-label">F1 Score</div></div>
    </div>
    <div class="result-actions">
      <button class="btn-save-model" onclick="saveModel()">💾 Save Model</button>
    </div>`;
  
  if (data.roc_curve) {
    html += `<div style="margin-top:14px"><b>ROC Curve:</b><br>
      <img class="viz-img" style="max-width:600px;margin-top:8px"
           src="data:image/png;base64,${data.roc_curve}"/></div>`;
  }

  if (data.confusion_matrix) {
    html += `<div style="margin-top:14px"><b>Confusion Matrix:</b><br>
      <img class="viz-img" style="max-width:380px;margin-top:8px"
           src="data:image/png;base64,${data.confusion_matrix}"/></div>`;
  }

  html += `<div style="margin-top:14px"><b>Classification Report:</b>
    <pre class="report-box">${esc(data.report)}</pre></div>`;

  document.getElementById('mlResultDisplay').innerHTML = html;
  loadScoresHistory();
  toast(`Training done ✅ (${data.elapsed_s}s)`);
}

async function saveModel() {
  const data = await api('/api/save_model', {});
  if (data.error) { toast('❌ '+data.error); return; }
  toast('💾 Model saved ✅');
  document.getElementById('btnDeploy').style.display = 'inline-flex';
  const actionDiv = document.querySelector('.result-actions');
  if (actionDiv && !actionDiv.querySelector('.btn-deploy-link')) {
    const link = document.createElement('a');
    link.href = '/deploy';
    link.className = 'btn-deploy-link';
    link.textContent = '🚀 Deploy & Test';
    link.style.marginLeft = '8px';
    link.target = '_blank';
    actionDiv.appendChild(link);
  }
}

async function loadScoresHistory() {
  const history = await apiGet('/api/scores_history');
  const div = document.getElementById('scoresHistory'); if (!div) return;
  if (!history?.length) { div.innerHTML='<p class="placeholder-text" style="margin-top:20px">No scores saved yet</p>'; return; }
  let h = '<div class="scores-table-wrap"><table class="scores-table"><thead><tr>' +
    '<th>#</th><th>Model</th><th>Accuracy</th><th>F1</th><th>⏱</th><th>Date</th>' +
    '</tr></thead><tbody>';
  history.slice().reverse().forEach((s,i) => {
    h += `<tr>
      <td>${history.length-i}</td>
      <td><b>${esc(s.model_name)}</b></td>
      <td class="score-cell">${(s.accuracy*100).toFixed(2)}%</td>
      <td class="score-cell">${(s.f1_score*100).toFixed(2)}%</td>
      <td>${s.elapsed_s}s</td>
      <td>${esc(s.timestamp)}</td>
    </tr>`;
  });
  div.innerHTML = h + '</tbody></table></div>';
}

// Initialize on Page Load
document.addEventListener('DOMContentLoaded', () => {
  ['btnDataInfo','btnProcessing','btnViz','btnML'].forEach(id => {
    const b = document.getElementById(id); if (b) b.disabled = true;
  });
});