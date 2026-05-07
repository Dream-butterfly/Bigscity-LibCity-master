/* ═══════════════════════════════════════════════════════
   GNNTP Web Console — v2 Main Script
   基于 main.js 迁移，保留核心功能，适配 sidebar 新布局
   ═══════════════════════════════════════════════════════ */

"use strict";

/* ═══════════════════════ GLOBALS ═══════════════════════ */
const TAB_MAP = {
  data:    { title: '数据处理',    sub: '数据工件生成 · 版本管理 · 数据集预览',      id: 'tab-data' },
  train:   { title: '训练',        sub: '参数配置 · 实时日志 · Loss/指标可视化',     id: 'tab-train' },
  resume:  { title: '继续训练',    sub: 'Checkpoint续训 · 历史运行恢复',             id: 'tab-resume' },
  compare: { title: '模型对比',    sub: '多模型指标对比 · 预测曲线叠加',             id: 'tab-compare' },
  history: { title: '运行历史',    sub: '训练记录查询 · 指标追踪',                   id: 'tab-history' },
};

/* ── ECharts 实例池 ── */
const CHART_IDS = ['chartModelParams','chartLoss','chartPred','chartCompare'];

/* ═══════════════════════ TAB SWITCHING ═══════════════════════ */
function switchTab(name) {
  // 隐藏所有 tab
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  // 取消所有 sidebar 高亮
  document.querySelectorAll('.sidebar-nav-item').forEach(el => el.classList.remove('active'));
  // 激活目标 tab
  const tabEl = document.getElementById(TAB_MAP[name].id);
  if (tabEl) tabEl.classList.add('active');
  const btn = document.querySelector(`[data-tab="${name}"]`);
  if (btn) btn.classList.add('active');
  // 更新 header
  document.getElementById('tabTitle').textContent = TAB_MAP[name].title;
  document.querySelector('.page-header-left .sub').textContent = TAB_MAP[name].sub;
  // 延迟 resize 图表
  setTimeout(resizeAllCharts, 300);
}

/* ═══════════════════════ THEME ═══════════════════════ */
function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  const tweakSelect = document.getElementById('tweaksTheme');
  if (tweakSelect) tweakSelect.value = mode;
  setTimeout(resizeAllCharts, 100);
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  setTheme(cur === 'dark' ? 'light' : 'dark');
}

/* ═══════════════════════ TWEAKS ═══════════════════════ */
function setupTweaks() {
  const toggle = document.getElementById('tweaksToggle');
  const panel = document.getElementById('tweaksPanel');
  if (toggle && panel) {
    toggle.addEventListener('click', () => panel.classList.toggle('visible'));
  }

  const themeSelect = document.getElementById('tweaksTheme');
  if (themeSelect) {
    themeSelect.addEventListener('change', e => setTheme(e.target.value));
  }

  // 间距调节
  document.getElementById('tweaksSpacing')?.addEventListener('change', e => {
    const scales = { compact: .75, default: 1, relaxed: 1.35 };
    const s = scales[e.target.value] || 1;
    const root = document.documentElement;
    root.style.setProperty('--spacing-xs',  `${4*s}px`);
    root.style.setProperty('--spacing-sm', `${8*s}px`);
    root.style.setProperty('--spacing-md', `${12*s}px`);
    root.style.setProperty('--spacing-lg', `${16*s}px`);
    root.style.setProperty('--spacing-xl', `${20*s}px`);
    root.style.setProperty('--spacing-2xl',`${24*s}px`);
    root.style.setProperty('--spacing-3xl',`${32*s}px`);
    root.style.setProperty('--spacing-4xl',`${40*s}px`);
  });

  // 圆角调节
  document.getElementById('tweaksRadius')?.addEventListener('change', e => {
    const radii = {
      sharp:   { sm:'2px', md:'4px', lg:'6px',  xl:'10px' },
      default: { sm:'6px', md:'10px',lg:'14px', xl:'20px' },
      round:   { sm:'10px',md:'16px',lg:'22px', xl:'28px' },
    };
    const r = radii[e.target.value] || radii.default;
    const root = document.documentElement;
    root.style.setProperty('--radius-sm', r.sm);
    root.style.setProperty('--radius-md', r.md);
    root.style.setProperty('--radius-lg', r.lg);
    root.style.setProperty('--radius-xl', r.xl);
  });

  // 密度调节
  document.getElementById('tweaksDensity')?.addEventListener('change', e => {
    const dense = e.target.value === 'dense';
    document.querySelectorAll('.card').forEach(c => c.style.padding = dense ? '10px 14px' : '');
    document.querySelectorAll('.group-grid').forEach(g => g.style.gap = dense ? '6px' : '');
  });
}

/* ═══════════════════════ ECHARTS ═══════════════════════ */
function getEChartsTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    textStyle: { color: isDark ? '#9ba3b0' : '#4a4f5a' },
    legend: { textStyle: { color: isDark ? '#9ba3b0' : '#4a4f5a' } },
    tooltip: {
      backgroundColor: isDark ? 'rgba(14,18,26,.94)' : 'rgba(255,255,255,.94)',
      borderColor: isDark ? '#252b36' : '#e2e5ea',
      textStyle: { color: isDark ? '#e2e6ed' : '#1a1d23' },
    },
  };
}

function resizeAllCharts() {
  CHART_IDS.forEach(id => {
    const dom = document.getElementById(id);
    if (dom && dom._echart && dom.offsetParent !== null) dom._echart.resize();
  });
}
window.addEventListener('resize', resizeAllCharts);

/* ═══════════════════════ API HELPERS ═══════════════════════ */
async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `POST ${url} failed: ${res.status}`);
  }
  return res.json();
}

/* ═══════════════════════ METADATA LOAD ═══════════════════════ */
async function loadMeta() {
  try {
    const data = await apiGet('/api/meta');
    // 填充 task
    const taskSel = document.getElementById('data_task');
    if (taskSel && data.models) {
      taskSel.innerHTML = Object.keys(data.models).map(t =>
        `<option value="${t}">${t}</option>`
      ).join('');
    }
    // 填充 model (依赖 task 变化)
    if (data.models) {
      setupModelDatasetSelects(data.models, data.datasets || []);
    }
    // 填充数据版本
    if (data.data_versions) {
      populateDataVersionSelects(data.data_versions);
    }
  } catch (e) {
    console.error('loadMeta failed:', e);
  }
}

function setupModelDatasetSelects(models, datasets) {
  const taskSel = document.getElementById('data_task');
  const modelSel = document.getElementById('data_model');
  const datasetSel = document.getElementById('data_dataset');
  if (!taskSel || !modelSel || !datasetSel) return;

  function updateModelDataset() {
    const task = taskSel.value;
    const modelList = models[task] || [];
    modelSel.innerHTML = modelList.map(m => `<option value="${m}">${m}</option>`).join('');
    datasetSel.innerHTML = datasets.map(d => `<option value="${d}">${d}</option>`).join('');
    // 触发 populateTrainParams
    if (modelSel.value) populateTrainParams(modelSel.value);
  }
  taskSel.addEventListener('change', updateModelDataset);
  modelSel.addEventListener('change', () => populateTrainParams(modelSel.value));
  updateModelDataset();
}

function populateDataVersionSelects(versions) {
  ['train_data_version', 'resume_data_version'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = versions.map(v =>
      `<option value="${v.version_id}">${v.version_id} (${v.status} | ${v.model}/${v.dataset})</option>`
    ).join('');
  });
}

/* ═══════════════════════ DEFAULT CONFIG ═══════════════════════ */
async function populateTrainParams(model) {
  // 后续 Phase 2 实现完整参数表
}

/* ═══════════════════════ DATA PREVIEW ═══════════════════════ */
async function loadDataPreview() {
  const fileType = document.getElementById('data_preview_file_type')?.value || 'dyna';
  const entityId = document.getElementById('data_preview_entity_id')?.value || '';
  const timeStart = document.getElementById('data_preview_time_start')?.value || '';
  const timeEnd = document.getElementById('data_preview_time_end')?.value || '';
  const columns = document.getElementById('data_preview_columns')?.value || '';
  const rows = document.getElementById('data_preview_rows')?.value || '20';
  const dataset = document.getElementById('data_dataset')?.value || '';

  const params = new URLSearchParams({ file_type: fileType, rows, dataset });
  if (entityId) params.set('entity_id', entityId);
  if (timeStart) params.set('time_start', timeStart);
  if (timeEnd) params.set('time_end', timeEnd);
  if (columns) params.set('columns', columns);

  try {
    const data = await apiGet(`/api/data/preview?${params}`);
    renderPreviewTable(data);
  } catch (e) {
    console.error('loadDataPreview failed:', e);
  }
}

function renderPreviewTable(data) {
  const meta = document.getElementById('data_preview_meta');
  const table = document.getElementById('data_preview_table');
  if (!table) return;
  if (meta) meta.textContent = data.meta || '';
  if (!data.data || data.data.length === 0) {
    table.innerHTML = '';
    return;
  }
  const cols = Object.keys(data.data[0]);
  table.innerHTML =
    `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>` +
    `<tbody>${data.data.map(row =>
      `<tr>${cols.map(c => `<td>${row[c] ?? ''}</td>`).join('')}</tr>`
    ).join('')}</tbody>`;
}

/* ═══════════════════════ DATA VERSIONS ═══════════════════════ */
async function loadDataVersions() {
  try {
    const versions = await apiGet('/api/data/versions');
    populateDataVersionSelects(versions);
  } catch (e) {
    console.error('loadDataVersions failed:', e);
  }
}

/* ═══════════════════════ DATA PREP ═══════════════════════ */
async function startDataPrep() {
  const task = document.getElementById('data_task')?.value;
  const model = document.getElementById('data_model')?.value;
  const dataset = document.getElementById('data_dataset')?.value;
  const seed = document.getElementById('data_seed')?.value || '0';
  const batchSize = document.getElementById('data_batch_size')?.value || '64';
  const datasetClass = document.getElementById('data_dataset_class')?.value || 'TrafficStatePointDataset';
  const trainRate = document.getElementById('data_train_rate')?.value || '0.6';
  const evalRate = document.getElementById('data_eval_rate')?.value || '0.2';
  const testRate = document.getElementById('data_test_rate')?.value || '0.2';
  const extraArgs = document.getElementById('data_extra_args')?.value || '';

  try {
    await apiPost('/api/data/start', {
      task, model, dataset, seed, batch_size: batchSize,
      dataset_class: datasetClass,
      train_rate: trainRate, eval_rate: evalRate,
      extra_args: extraArgs,
    });
    // 开始轮询日志
    pollDataLogs();
  } catch (e) {
    console.error('startDataPrep failed:', e);
  }
}

/* ═══════════════════════ LOG POLLING ═══════════════════════ */
let dataLogIndex = 0;
let trainLogIndex = 0;
let resumeLogIndex = 0;

async function pollDataLogs() {
  const el = document.getElementById('data_logs');
  if (!el) return;
  try {
    const data = await apiGet(`/api/data/status?since=${dataLogIndex}`);
    if (data.lines) {
      data.lines.forEach(line => appendLogLine(el, line));
      dataLogIndex = data.next_index || dataLogIndex;
    }
    if (data.running) {
      setTimeout(pollDataLogs, 1000);
    }
  } catch (e) {
    setTimeout(pollDataLogs, 2000);
  }
}

function appendLogLine(container, line) {
  const lvl = (line.level || 'INFO').toLowerCase();
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML =
    `<span class="log-index">${line.idx || ''}</span>` +
    `<span class="log-level log-level-${lvl}">${(line.level || 'INFO').toUpperCase()}</span>` +
    `<span class="log-message">${escapeHtml(line.message || '')}</span>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ═══════════════════════ TRAINING ═══════════════════════ */
async function startTrain() {
  const dataVersion = document.getElementById('train_data_version')?.value;
  const maxEpoch = document.getElementById('max_epoch')?.value || '10';
  const batchSize = document.getElementById('batch_size')?.value || '64';
  const learningRate = document.getElementById('learning_rate')?.value || '0.001';
  const savedModel = document.getElementById('saved_model')?.value || 'true';
  const train = document.getElementById('train')?.value || 'true';
  const expId = document.getElementById('exp_id')?.value || '';
  const gpu = document.getElementById('gpu')?.value || '';
  const gpuId = document.getElementById('gpu_id')?.value || '';
  const executor = document.getElementById('executor')?.value || '';
  const evaluator = document.getElementById('evaluator')?.value || '';
  const extraArgs = document.getElementById('extra_args')?.value || '';

  try {
    await apiPost('/api/start', {
      data_version_id: dataVersion, max_epoch: maxEpoch,
      batch_size: batchSize, learning_rate: learningRate,
      saved_model: savedModel, train,
      exp_id: expId, gpu, gpu_id: gpuId,
      executor, evaluator, extra_args: extraArgs,
    });
    pollTrainLogs();
  } catch (e) {
    console.error('startTrain failed:', e);
  }
}

async function stopTrain() {
  try { await apiPost('/api/stop', {}); } catch (e) { console.error(e); }
}

async function clearState() {
  try { await apiPost('/api/clear', {}); } catch (e) { console.error(e); }
}

async function pollTrainLogs() {
  const el = document.getElementById('logs');
  if (!el) return;
  try {
    const data = await apiGet(`/api/status?since=${trainLogIndex}`);
    if (data.lines) {
      data.lines.forEach(line => appendLogLine(el, line));
      trainLogIndex = data.next_index || trainLogIndex;
    }
    // 更新 status badge
    const badge = document.getElementById('status');
    if (badge) {
      badge.className = `badge badge-${data.status || 'idle'}`;
      badge.textContent = data.status || 'idle';
    }
    if (data.running) {
      setTimeout(pollTrainLogs, 1000);
    }
  } catch (e) {
    setTimeout(pollTrainLogs, 2000);
  }
}

/* ═══════════════════════ RESUME TRAINING ═══════════════════════ */
async function startResumeTrain() {
  // Phase 2 实现
}

/* ═══════════════════════ COMPARE ═══════════════════════ */
async function loadComparison() {
  // Phase 2 实现
}

/* ═══════════════════════ HISTORY ═══════════════════════ */
async function loadHistory() {
  // Phase 2 实现
}

/* ═══════════════════════ SPLIT PRESET ═══════════════════════ */
function applySplitPreset() {
  const sel = document.getElementById('data_split_preset');
  if (!sel || !sel.value) return;
  const parts = sel.value.split(',').map(Number);
  const trainInput = document.getElementById('data_train_rate');
  const evalInput = document.getElementById('data_eval_rate');
  const testInput = document.getElementById('data_test_rate');
  if (trainInput) trainInput.value = parts[0];
  if (evalInput) evalInput.value = parts[1];
  if (testInput) testInput.value = parts[2];
  updateSplitRatioHint();
}

function updateSplitRatioHint() {
  const hint = document.getElementById('data_split_ratio_hint');
  if (!hint) return;
  const t = document.getElementById('data_train_rate')?.value || '0';
  const e = document.getElementById('data_eval_rate')?.value || '0';
  const s = document.getElementById('data_test_rate')?.value || '0';
  hint.textContent = `train/eval/test = ${Number(t).toFixed(3)} / ${Number(e).toFixed(3)} / ${Number(s).toFixed(3)}`;
}

/* ═══════════════════════ DATA VERSION ACTIONS ═══════════════════════ */
function applySelectedDataVersionToTrain() {
  const sel = document.getElementById('data_version_select');
  const trainSel = document.getElementById('train_data_version');
  if (sel && trainSel && sel.value) trainSel.value = sel.value;
}

/* ═══════════════════════ ECHARTS INIT ═══════════════════════ */
function initCharts() {
  // Phase 2: 参照 main.js 实现具体图表渲染
}

/* ═══════════════════════ INIT ═══════════════════════ */
function init() {
  // Tab 切换绑定
  document.querySelectorAll('.sidebar-nav-item').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // 主题切换
  document.getElementById('themeToggleSidebar')?.addEventListener('click', toggleTheme);

  // Tweaks
  setupTweaks();

  // Split 预设
  const splitPreset = document.getElementById('data_split_preset');
  if (splitPreset) {
    splitPreset.addEventListener('change', applySplitPreset);
  }

  // 数据版本同步
  document.getElementById('btnSyncToTrain')?.addEventListener('click', applySelectedDataVersionToTrain);

  // 加载元数据
  loadMeta();

  // Charts
  initCharts();
}

document.addEventListener('DOMContentLoaded', init);
