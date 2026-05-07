/* ═══════════════════════════════════════════════════════
   GNNTP Web Console — v2 Main Script
   基于 main.js 迁移，保留核心功能，适配 sidebar 新布局
   ═══════════════════════════════════════════════════════ */

"use strict";

/* ═══════════════════════ GLOBALS ═══════════════════════ */
const DEFAULT_LANG = new URLSearchParams(window.location.search).get('lang')
  || window.localStorage.getItem('train_web_lang') || 'zh-CN';
const DEFAULT_THEME = new URLSearchParams(window.location.search).get('theme')
  || window.localStorage.getItem('train_web_theme') || 'dark';

const HIDDEN_TRAIN_PARAM_KEYS = new Set([
  'config_file', 'train_rate', 'eval_rate', 'dataset_class', 'task', 'model', 'dataset', 'seed',
]);
const CLI_FIELDS = [
  'config_file', 'exp_id', 'seed', 'gpu', 'gpu_id',
  'train_rate', 'eval_rate', 'batch_size', 'learning_rate',
  'max_epoch', 'dataset_class', 'executor', 'evaluator',
  'num_gpus', 'gpu_ids',
];

/* ── Module-level state ── */
let _state = {
  paramRowsConfig: [],
  paramRowsExecutor: [],
  dataVersions: [],
  modelPlotType: 'pie',
  modelPlot: {},
  modelPlotOptionPie: {},
  modelPlotOptionBar: {},
  lossPlot: {},
  predictionRanges: {horizon: 0, node: 0, feature: 0},
  predictionSelection: {horizon: 1, node: 1, feature: 1},
};
let _resumeRuns = [];
let _compareRuns = [];
let _charts = {};
let _trainLogLines = [];
let _resumeLogLines = [];
let trainLogIndex = 0;
let resumeLogIndex = 0;

const TAB_MAP = {
  data:    { title: '数据处理',    sub: '数据工件生成 · 版本管理 · 数据集预览',      id: 'tab-data' },
  train:   { title: '训练',        sub: '参数配置 · 实时日志 · Loss/指标可视化',     id: 'tab-train' },
  resume:  { title: '继续训练',    sub: 'Checkpoint续训 · 历史运行恢复',             id: 'tab-resume' },
  compare: { title: '模型对比',    sub: '多模型指标对比 · 预测曲线叠加',             id: 'tab-compare' },
  history: { title: '运行历史',    sub: '训练记录查询 · 指标追踪',                   id: 'tab-history' },
};

const TAB_I18N_KEYS = {
  data:    { title: 'tab_data_title',    sub: 'tab_data_sub' },
  train:   { title: 'tab_train_title',   sub: 'tab_train_sub' },
  resume:  { title: 'tab_resume_title',  sub: 'tab_resume_sub' },
  compare: { title: 'tab_compare_title', sub: 'tab_compare_sub' },
  history: { title: 'tab_history_title', sub: 'tab_history_sub' },
};

/* ── i18n State ── */
let _i18nUI = {};
let _i18nParams = {};

function t(key, fallback) {
  return _i18nUI[key] || fallback || key;
}

function getParamDisplayName(key) {
  return _i18nParams[key] || key;
}

function applyI18n() {
  // data-i18n 元素
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (!key) return;
    if (!el.dataset.i18nFallback) el.dataset.i18nFallback = el.innerText;
    el.innerText = t(key, el.dataset.i18nFallback);
  });
  // data-i18n-placeholder 元素
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (!key) return;
    if (!el.dataset.i18nPlaceholderFallback) {
      el.dataset.i18nPlaceholderFallback = el.getAttribute('placeholder') || '';
    }
    el.setAttribute('placeholder', t(key, el.dataset.i18nPlaceholderFallback));
  });
  // data-param-label 元素
  document.querySelectorAll('[data-param-label]').forEach(el => {
    const key = el.getAttribute('data-param-label');
    if (!key) return;
    el.innerText = getParamDisplayName(key);
  });
}

async function loadI18n(lang) {
  const normalized = lang.replace(/_/g, '-');
  window.localStorage.setItem('train_web_lang', normalized);
  try {
    const r = await fetch(`/static/i18n/${encodeURIComponent(normalized)}.json`);
    if (r.ok) {
      const data = await r.json();
      _i18nUI = data.ui || {};
      _i18nParams = data.params || {};
      document.documentElement.lang = data.lang || normalized;
    } else {
      _i18nUI = {}; _i18nParams = {};
    }
  } catch (_) {
    _i18nUI = {}; _i18nParams = {};
  }
  applyI18n();
  // 更新 lang 选择器
  const sel = document.getElementById('lang_select');
  if (sel) sel.value = normalized;
  // 更新 page header
  updatePageHeaderI18n();
}

function updatePageHeaderI18n() {
  const activeTab = document.querySelector('.sidebar-nav-item.active');
  if (!activeTab) return;
  const name = activeTab.dataset.tab;
  const keys = TAB_I18N_KEYS[name];
  if (!keys) return;
  const titleEl = document.getElementById('tabTitle');
  const subEl = document.getElementById('tabSub');
  if (titleEl && keys.title) {
    titleEl.textContent = t(keys.title, TAB_MAP[name].title);
  }
  if (subEl && keys.sub) {
    subEl.textContent = t(keys.sub, TAB_MAP[name].sub);
  }
}

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
  // 更新 header (优先用 i18n)
  const titleEl = document.getElementById('tabTitle');
  const subEl = document.querySelector('.page-header-left .sub');
  const i18nTitle = t(TAB_I18N_KEYS[name].title, TAB_MAP[name].title);
  const i18nSub = t(TAB_I18N_KEYS[name].sub, TAB_MAP[name].sub);
  if (titleEl) titleEl.textContent = i18nTitle;
  if (subEl) subEl.textContent = i18nSub;
  // 延迟 resize 图表
  setTimeout(resizeAllCharts, 300);
}

/* ═══════════════════════ THEME ═══════════════════════ */
function setTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  window.localStorage.setItem('train_web_theme', mode);
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
  Object.values(_charts).forEach(chart => {
    if (chart) chart.resize();
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
function buildTaskModelMap(modelsMeta) {
  const map = {};
  if (Array.isArray(modelsMeta)) {
    modelsMeta.forEach(item => {
      const task = String(item?.task || '').trim();
      const model = String(item?.model || '').trim();
      if (!task || !model) return;
      if (!map[task]) map[task] = [];
      if (!map[task].includes(model)) map[task].push(model);
    });
    return map;
  }
  if (modelsMeta && typeof modelsMeta === 'object') {
    Object.entries(modelsMeta).forEach(([task, models]) => {
      const taskName = String(task || '').trim();
      if (!taskName) return;
      const list = Array.isArray(models) ? models : [];
      map[taskName] = list
        .map(x => String(x || '').trim())
        .filter(Boolean);
    });
  }
  return map;
}

async function loadMeta() {
  try {
    const data = await apiGet('/api/meta');
    const taskModelMap = buildTaskModelMap(data.models);
    // 填充 task
    const taskSel = document.getElementById('data_task');
    if (taskSel) {
      const tasks = Object.keys(taskModelMap).sort();
      taskSel.innerHTML = tasks.map(t =>
        `<option value="${t}">${t}</option>`
      ).join('');
      const stgcnTask = Array.isArray(data.models)
        ? (data.models.find(x => String(x?.model || '') === 'STGCN')?.task || '')
        : '';
      if (stgcnTask && tasks.includes(stgcnTask)) {
        taskSel.value = stgcnTask;
      } else if (tasks.includes('traffic_state_pred')) {
        taskSel.value = 'traffic_state_pred';
      }
    }
    // 填充 model (依赖 task 变化)
    setupModelDatasetSelects(taskModelMap, data.datasets || []);
    // 填充数据版本
    if (data.data_versions) {
      const versions = Array.isArray(data.data_versions) ? data.data_versions : [];
      _state.dataVersions = versions;
      populateDataVersionSelects(versions);
      renderDataVersionTable(versions);
      populateDatasetFilter(versions);
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
    if (modelList.includes('STGCN')) modelSel.value = 'STGCN';
    datasetSel.innerHTML = datasets.map(d => `<option value="${d}">${d}</option>`).join('');
    // 触发 populateTrainParams
    if (modelSel.value) populateTrainParams(modelSel.value);
  }
  taskSel.addEventListener('change', updateModelDataset);
  modelSel.addEventListener('change', () => populateTrainParams(modelSel.value));
  updateModelDataset();
}

function populateDataVersionSelects(versions) {
  const selectedRunId = document.getElementById('resume_run_id')?.value || '';
  const selectedRun = (_resumeRuns || []).find(x => x.run_id === selectedRunId);
  const ready = versions.filter(v => String(v.status || '').toLowerCase() === 'ready');
  const resumeReady = selectedRun
    ? ready.filter(v => v.task === selectedRun.task && v.model === selectedRun.model && v.dataset === selectedRun.dataset)
    : ready;
  const renderSelect = (id, items) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = items.map(v =>
      `<option value="${escHtml(v.version_id)}">${escHtml(v.version_id)} (${escHtml(v.status || '-')} | ${escHtml(v.model || '-')}/${escHtml(v.dataset || '-')})</option>`
    ).join('');
    if (current && items.some(v => v.version_id === current)) {
      sel.value = current;
    } else if (items.length) {
      sel.value = items[0].version_id;
    }
  };
  renderSelect('train_data_version', ready);
  renderSelect('resume_data_version', resumeReady);
}

/* ═══════════════════════ PARAM TABLE UTILITIES ═══════════════════════ */
function inferType(v) {
  if (v === null || v === undefined) return 'str';
  if (typeof v === 'boolean') return 'bool';
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'float';
  if (typeof v === 'object') return 'json';
  return 'str';
}

function toInputString(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'object') return JSON.stringify(v, null, 2);
  return String(v);
}

function parseValue(type, raw) {
  const t = (type || 'str').toLowerCase();
  if (t === 'bool') {
    const x = String(raw).trim().toLowerCase();
    if (['true', '1', 'yes'].includes(x)) return true;
    if (['false', '0', 'no'].includes(x)) return false;
    throw new Error(`布尔值错误: ${raw}`);
  }
  if (t === 'int') {
    const n = parseInt(String(raw).trim(), 10);
    if (Number.isNaN(n)) throw new Error(`整数错误: ${raw}`);
    return n;
  }
  if (t === 'float') {
    const n = Number(String(raw).trim());
    if (Number.isNaN(n)) throw new Error(`浮点数错误: ${raw}`);
    return n;
  }
  if (t === 'json') {
    const txt = String(raw).trim();
    return txt ? JSON.parse(txt) : null;
  }
  return String(raw);
}

function renderParamCell(section, idx) {
  if (idx === null) return '';
  const rows = section === 'executor' ? _state.paramRowsExecutor : _state.paramRowsConfig;
  const row = rows[idx];
  const valueEditor = row.type === 'json'
    ? `<textarea class="param-value mono value-input value-input-json" data-section="${section}" data-idx="${idx}" rows="4">${escHtml(row.value)}</textarea>`
    : `<input class="param-value mono value-input" data-section="${section}" data-idx="${idx}" value="${escHtml(row.value)}"/>`;
  return `
    <td class="param-localized">${escHtml(getParamDisplayName(row.key))}</td>
    <td class="mono param-key">${escHtml(row.key)}</td>
    <td>
      <select class="type-select" data-section="${section}" data-idx="${idx}">
        <option value="str" ${row.type === 'str' ? 'selected' : ''}>str</option>
        <option value="bool" ${row.type === 'bool' ? 'selected' : ''}>bool</option>
        <option value="int" ${row.type === 'int' ? 'selected' : ''}>int</option>
        <option value="float" ${row.type === 'float' ? 'selected' : ''}>float</option>
        <option value="json" ${row.type === 'json' ? 'selected' : ''}>json</option>
      </select>
    </td>
    <td>${valueEditor}</td>
  `;
}

function renderParamTable() {
  const filter = document.getElementById('param_filter')?.value.trim().toLowerCase() || '';
  const configIds = _state.paramRowsConfig
    .map((r, i) => ({r, i}))
    .filter(x => !filter || x.r.key.toLowerCase().includes(filter))
    .map(x => x.i);
  const executorIds = _state.paramRowsExecutor
    .map((r, i) => ({r, i}))
    .filter(x => !filter || x.r.key.toLowerCase().includes(filter))
    .map(x => x.i);
  const countEl = document.getElementById('param_count');
  if (countEl) countEl.textContent = String(configIds.length + executorIds.length);
  const c1 = document.getElementById('param_count_config');
  if (c1) c1.textContent = String(configIds.length);
  const c2 = document.getElementById('param_count_executor');
  if (c2) c2.textContent = String(executorIds.length);
  const bodyConfig = document.getElementById('param_tbody_config');
  const bodyExecutor = document.getElementById('param_tbody_executor');
  if (bodyConfig) {
    bodyConfig.innerHTML = configIds.map(idx => `<tr>${renderParamCell('config', idx)}</tr>`).join('')
      || `<tr><td colspan="4" class="small">${escHtml(t('table_no_params', '无参数'))}</td></tr>`;
  }
  if (bodyExecutor) {
    bodyExecutor.innerHTML = executorIds.map(idx => `<tr>${renderParamCell('executor', idx)}</tr>`).join('')
      || `<tr><td colspan="4" class="small">${escHtml(t('table_no_params', '无参数'))}</td></tr>`;
  }
  // Bind type select change
  document.querySelectorAll('#param_tbody_config .type-select, #param_tbody_executor .type-select').forEach(el => {
    el.addEventListener('change', e => {
      const section = e.target.getAttribute('data-section');
      const idx = Number(e.target.getAttribute('data-idx'));
      const rows = section === 'executor' ? _state.paramRowsExecutor : _state.paramRowsConfig;
      rows[idx].type = e.target.value;
      renderParamTable();
    });
  });
  // Bind value input change
  document.querySelectorAll('#param_tbody_config .value-input, #param_tbody_executor .value-input').forEach(el => {
    el.addEventListener('input', e => {
      const section = e.target.getAttribute('data-section');
      const idx = Number(e.target.getAttribute('data-idx'));
      const rows = section === 'executor' ? _state.paramRowsExecutor : _state.paramRowsConfig;
      rows[idx].value = e.target.value;
    });
  });
}

function collectConfigFromTable() {
  const cfg = {};
  for (const r of _state.paramRowsConfig) cfg[r.key] = parseValue(r.type, r.value);
  for (const r of _state.paramRowsExecutor) cfg[r.key] = parseValue(r.type, r.value);
  return cfg;
}

function applyDefaultToCliFields(config) {
  const setIf = (k, v) => {
    if (v !== undefined && v !== null && document.getElementById(k)) document.getElementById(k).value = String(v);
  };
  setIf('max_epoch', config.max_epoch ?? 10);
  setIf('seed', config.seed);
  setIf('gpu_id', config.gpu_id);
  setIf('train_rate', config.train_rate);
  setIf('eval_rate', config.eval_rate);
  setIf('batch_size', config.batch_size);
  setIf('learning_rate', config.learning_rate);
  setIf('dataset_class', config.dataset_class);
  setIf('executor', config.executor);
  setIf('evaluator', config.evaluator);
  if (config.gpu !== undefined && document.getElementById('gpu')) {
    document.getElementById('gpu').value = String(config.gpu).toLowerCase();
  }
}

function collectCliOptions() {
  const out = {};
  for (const k of CLI_FIELDS) {
    const el = document.getElementById(k);
    if (!el) continue;
    const v = (el.value ?? '').trim();
    if (v !== '') out[k] = v;
  }
  return out;
}

function getTrainContextMeta() {
  const vid = document.getElementById('train_data_version')?.value || '';
  if (vid) {
    const v = (_state.dataVersions || []).find(x => x.version_id === vid);
    if (v && v.task && v.model && v.dataset) {
      return {task: String(v.task), model: String(v.model), dataset: String(v.dataset), version: v};
    }
  }
  const task = document.getElementById('data_task')?.value || '';
  const model = document.getElementById('data_model')?.value || '';
  const dataset = document.getElementById('data_dataset')?.value || '';
  if (task && model && dataset) return {task, model, dataset, version: null};
  return null;
}

/* ── GPU multi-select ── */
function onNumGpusChange() {
  const numGpus = parseInt(document.getElementById('num_gpus')?.value || '1', 10);
  const checkboxesDiv = document.getElementById('gpu_checkboxes');
  const display = document.getElementById('gpu_ids_display');
  if (!checkboxesDiv || !display) return;

  if (numGpus > 1) {
    checkboxesDiv.style.display = 'flex';
    // Auto-check first N GPUs
    const cbs = checkboxesDiv.querySelectorAll('input[type="checkbox"]');
    const currentChecked = new Set(getCheckedGpuIds());
    let need = numGpus - currentChecked.size;
    if (need > 0) {
      for (const cb of cbs) {
        if (need <= 0) break;
        if (!cb.checked) { cb.checked = true; need--; }
      }
    } else if (need < 0) {
      // Uncheck from the end
      for (let i = cbs.length - 1; i >= 0 && need < 0; i--) {
        if (cbs[i].checked) { cbs[i].checked = false; need++; }
      }
    }
  } else {
    checkboxesDiv.style.display = 'none';
  }
  updateGpuDisplay();
}

function onGpuCheckboxChange() {
  updateGpuDisplay();
}

function getCheckedGpuIds() {
  const checkboxes = document.querySelectorAll('#gpu_checkboxes input[type="checkbox"]:checked');
  return Array.from(checkboxes).map(cb => parseInt(cb.value, 10)).sort((a,b)=>a-b);
}

function updateGpuDisplay() {
  const display = document.getElementById('gpu_ids_display');
  if (!display) return;
  const ids = getCheckedGpuIds();
  display.textContent = ids.length > 0 ? `gpu_ids: [${ids.join(', ')}]` : 'gpu_ids: [0]';
}

function collectGpuOptions() {
  const numGpusEl = document.getElementById('num_gpus');
  const numGpus = parseInt(numGpusEl?.value || '1', 10);
  if (numGpus <= 1) return {num_gpus: '1', gpu_ids: ''};
  const ids = getCheckedGpuIds();
  return {num_gpus: String(numGpus), gpu_ids: ids.join(',')};
}

function applyTrainDataVersionSelection() {
  const vid = document.getElementById('train_data_version')?.value || '';
  const v = (_state.dataVersions || []).find(x => x.version_id === vid);
  const lockIds = ['seed', 'train_rate', 'eval_rate', 'dataset_class', 'config_file'];
  lockIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = !!v;
  });
}

async function loadDefaults() {
  applyTrainDataVersionSelection();
  const ctx = getTrainContextMeta();
  if (!ctx) return;
  try {
    const data = await apiPost('/api/default_config', {
      task: ctx.task, model: ctx.model, dataset: ctx.dataset,
    });
    const cfg = data.config || {};
    cfg.max_epoch = 10;
    const executorKeys = new Set(data.executor_keys || []);
    const allRows = Object.keys(cfg).filter(k => !HIDDEN_TRAIN_PARAM_KEYS.has(k)).sort().map(k => ({
      key: k,
      defaultValue: cfg[k],
      type: inferType(cfg[k]),
      value: toInputString(cfg[k]),
    }));
    _state.paramRowsExecutor = allRows.filter(row => executorKeys.has(row.key));
    _state.paramRowsConfig = allRows.filter(row => !executorKeys.has(row.key));
    applyDefaultToCliFields(cfg);
    renderParamTable();
  } catch (e) {
    console.error('loadDefaults failed:', e);
  }
}

async function populateTrainParams(model) {
  await loadDefaults();
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
    const data = await apiGet('/api/data/versions');
    const versions = Array.isArray(data.versions) ? data.versions : [];
    _state.dataVersions = versions;
    populateDataVersionSelects(versions);
    renderDataVersionTable(versions);
    populateDatasetFilter(versions);
  } catch (e) {
    console.error('loadDataVersions failed:', e);
  }
}

function renderDataVersionTable(versions) {
  const tbody = document.getElementById('data_version_tbody');
  const countEl = document.getElementById('dataVersionCount');
  if (countEl) countEl.textContent = `${versions.length} versions`;
  if (!tbody) return;
  const filterVal = (document.getElementById('data_versions_dataset_filter')?.value || '').trim();
  const filtered = filterVal ? versions.filter(v => (v.dataset || '') === filterVal) : versions;
  if (!filtered.some(v => v.version_id === _state.selectedDataVersionId)) {
    _state.selectedDataVersionId = filtered.length ? filtered[0].version_id : '';
  }
  tbody.innerHTML = filtered.map(v => {
    const status = (v.status || '').toLowerCase();
    const statusCls = status === 'ready' ? 'badge-finished' : status === 'processing' ? 'badge-processing' : 'badge-failed';
    const p = getVersionProfile(v);
    const splitText = `${fmtRate(p.trainRate)}/${fmtRate(p.evalRate)}/${fmtRate(p.testRate)}`;
    const selectedCls = v.version_id === _state.selectedDataVersionId ? 'is-selected' : '';
    return `<tr class="${selectedCls}" data-version-id="${escHtml(v.version_id || '')}">
      <td class="mono">${escHtml(v.version_id || '-')}</td>
      <td><span class="badge ${statusCls}">${v.status || '-'}</span></td>
      <td>${escHtml(v.task || '-')}</td>
      <td>${escHtml(v.model || '-')}</td>
      <td>${escHtml(v.dataset || '-')}</td>
      <td class="mono">${splitText}</td>
      <td>${escHtml(v.updated_at || '-')}</td>
    </tr>`;
  }).join('');
  tbody.querySelectorAll('tr[data-version-id]').forEach(row => {
    row.addEventListener('click', () => {
      _state.selectedDataVersionId = row.getAttribute('data-version-id') || '';
      renderDataVersionTable(_state.dataVersions || []);
    });
  });
}

function getVersionProfile(v) {
  const cli = v.cli_options || {};
  const cfg = v.config_payload || {};
  const tn = (n) => {
    const x = Number(n);
    return Number.isFinite(x) ? x : null;
  };
  const trainRate = tn(v.train_rate ?? cli.train_rate ?? cfg.train_rate);
  const evalRate = tn(v.eval_rate ?? cli.eval_rate ?? cfg.eval_rate);
  let testRate = tn(v.test_rate);
  if (testRate === null && trainRate !== null && evalRate !== null) testRate = 1 - trainRate - evalRate;
  return { trainRate, evalRate, testRate };
}

function fmtRate(v) { return Number.isFinite(v) ? Number(v).toFixed(3) : '-'; }

function populateDatasetFilter(versions) {
  const sel = document.getElementById('data_versions_dataset_filter');
  if (!sel) return;
  const prev = sel.value;
  const datasets = [...new Set(versions.map(v => v.dataset).filter(Boolean))];
  sel.innerHTML = '<option value="">全部数据集</option>' +
    datasets.map(d => `<option value="${escHtml(d)}">${escHtml(d)}</option>`).join('');
  if (prev && datasets.includes(prev)) sel.value = prev;
  sel.onchange = () => renderDataVersionTable(_state.dataVersions || []);
}

async function stopDataPrep() {
  try {
    await apiPost('/api/data/stop', {});
  } catch (e) {
    console.error('stopDataPrep failed:', e);
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ═══════════════════════ DATA PREP ═══════════════════════ */
async function startDataPrep() {
  const task = document.getElementById('data_task')?.value;
  const model = document.getElementById('data_model')?.value;
  const dataset = document.getElementById('data_dataset')?.value;
  const extraArgs = document.getElementById('data_extra_args')?.value || '';

  // 收集 CLI 字段
  const cliOptions = {};
  ['seed', 'batch_size', 'dataset_class'].forEach(k => {
    const el = document.getElementById('data_' + k);
    if (el && el.value.trim()) cliOptions[k] = el.value.trim();
  });

  // 验证并收集 split 比例
  const t = parseFloat(document.getElementById('data_train_rate')?.value);
  const e = parseFloat(document.getElementById('data_eval_rate')?.value);
  const s = parseFloat(document.getElementById('data_test_rate')?.value);
  if (!Number.isFinite(t) || !Number.isFinite(e) || !Number.isFinite(s)) {
    alert('请填写有效的划分比例（必须为数字）');
    return;
  }
  if (Math.abs(t + e + s - 1) > 1e-6) {
    alert(`train+eval+test 必须等于 1（当前 ${(t+e+s).toFixed(4)}）`);
    return;
  }
  cliOptions.train_rate = String(t);
  cliOptions.eval_rate = String(e);

  try {
    dataLogPrevLen = 0;
    await apiPost('/api/data/start', {
      task, model, dataset,
      extra_args: extraArgs,
      cli_options: cliOptions,
      config: {},
    });
    pollDataLogs();
  } catch (e) {
    alert(`数据处理启动失败: ${e.message}`);
  }
}

/* ═══════════════════════ DATA LOG POLLING ═══════════════════════ */
let dataLogPrevLen = 0;

async function pollDataLogs() {
  const el = document.getElementById('data_logs');
  if (!el) return;
  try {
    const data = await apiGet('/api/data/status');
    const tail = Array.isArray(data.logs_tail) ? data.logs_tail : [];
    // 只追加新行（增量更新）
    if (tail.length > dataLogPrevLen) {
      for (let i = dataLogPrevLen; i < tail.length; i++) {
        appendDataLogLine(el, tail[i], i + 1);
      }
      dataLogPrevLen = tail.length;
    }
    if (data.running) {
      setTimeout(pollDataLogs, 1000);
    }
  } catch (e) {
    setTimeout(pollDataLogs, 2000);
  }
}

function appendDataLogLine(container, rawLine, idx) {
  const text = String(rawLine ?? '');
  const levelMatch = text.match(/\b(INFO|WARNING|ERROR|DEBUG)\b/);
  const level = levelMatch ? levelMatch[1].toLowerCase() : 'info';
  const div = document.createElement('div');
  div.className = 'log-line';
  div.innerHTML =
    `<span class="log-index">${idx}</span>` +
    `<span class="log-level log-level-${level}">${level.toUpperCase()}</span>` +
    `<span class="log-message">${escapeHtml(text)}</span>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ═══════════════════════ LOG FILTERING ═══════════════════════ */
function trainLogLevel(line) {
  const m = (line.message || '').match(/\b(INFO|WARNING|ERROR|DEBUG)\b/);
  return m ? m[1].toLowerCase() : 'info';
}

function renderFilteredLogs(containerId, linesArray) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const levelFilter = (document.getElementById('log_level_filter')?.value || 'all').toLowerCase();
  const keyword = (document.getElementById('log_keyword_filter')?.value || '').trim().toLowerCase();
  const filtered = (linesArray || []).filter(line => {
    if (levelFilter !== 'all' && trainLogLevel(line) !== levelFilter) return false;
    if (keyword && !(line.message || '').toLowerCase().includes(keyword)) return false;
    return true;
  });
  // Rebuild with innerHTML for clean re-filtering
  const html = filtered.map((line, i) => {
    const lvl = trainLogLevel(line);
    return `<div class="log-line">
      <span class="log-index">${i + 1}</span>
      <span class="log-level log-level-${lvl}">${(line.level || 'INFO').toUpperCase()}</span>
      <span class="log-message">${escapeHtml(line.message || '')}</span>
    </div>`;
  }).join('') || '<div class="log-empty" style="color:var(--text-muted);padding:1em">无匹配日志</div>';
  container.innerHTML = html;
  container.scrollTop = container.scrollHeight;
}

function applyLogFilter() {
  renderFilteredLogs('logs', _trainLogLines);
  renderFilteredLogs('resume_logs', _resumeLogLines);
}

function clearLogBuffer() {
  _trainLogLines = [];
  _resumeLogLines = [];
}

/* ═══════════════════════ TRAINING ═══════════════════════ */
async function startTrain() {
  const dataVersionId = document.getElementById('train_data_version')?.value;
  if (!dataVersionId) {
    alert('请先在训练页选择 ready 的 data_version_id');
    return;
  }
  const v = (_state.dataVersions || []).find(x => x.version_id === dataVersionId);
  if (!v || !v.task || !v.model || !v.dataset) {
    alert('所选 data_version 缺少 task/model/dataset 信息');
    return;
  }
  applyTrainDataVersionSelection();
  let config;
  try {
    config = collectConfigFromTable();
  } catch (e) {
    alert(`参数解析失败: ${e.message}`);
    return;
  }
  const cliOptions = collectCliOptions();
  const savedModel = document.getElementById('saved_model')?.value || 'true';
  const train = document.getElementById('train')?.value || 'true';
  const extraArgs = document.getElementById('extra_args')?.value || '';
  const gpuOpts = collectGpuOptions();

  try {
    await apiPost('/api/start', {
      task: v.task, model: v.model, dataset: v.dataset,
      data_version_id: dataVersionId,
      saved_model: savedModel, train,
      extra_args: extraArgs,
      cli_options: cliOptions,
      config: config,
      num_gpus: gpuOpts.num_gpus,
      gpu_ids: gpuOpts.gpu_ids,
    });
    clearLogBuffer();
    trainLogIndex = 0;
    pollTrainLogs();
  } catch (e) {
    alert(`启动失败: ${e.message}`);
  }
}

async function stopTrain() {
  try { await apiPost('/api/stop', {}); } catch (e) { console.error(e); }
}

async function clearState() {
  try { await apiPost('/api/clear', {}); } catch (e) { console.error(e); }
  clearLogBuffer();
  trainLogIndex = 0;
  resumeLogIndex = 0;
  dataLogPrevLen = 0;
  // 清空终端显示
  ['logs', 'resume_logs', 'data_logs'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  });
}

async function pollTrainLogs() {
  try {
    const data = await apiGet(`/api/status?since=${trainLogIndex}`);
    if (data.lines && data.lines.length) {
      _trainLogLines.push(...data.lines);
      trainLogIndex = data.next_index || trainLogIndex;
      renderFilteredLogs('logs', _trainLogLines);
    }
    // 更新 status badge
    const badge = document.getElementById('status');
    if (badge) {
      badge.className = `badge badge-${data.status || 'idle'}`;
      badge.textContent = data.status || 'idle';
    }
    // 更新模型参数量图
    if (data.model_plot || data.model_plot_option_pie || data.model_plot_option_bar) {
      _state.modelPlot = data.model_plot || _state.modelPlot;
      _state.modelPlotOptionPie = data.model_plot_option_pie || _state.modelPlotOptionPie;
      _state.modelPlotOptionBar = data.model_plot_option_bar || _state.modelPlotOptionBar;
      drawModelParamChart(_state.modelPlot, _state.modelPlotOptionPie, _state.modelPlotOptionBar);
    }
    // 更新 Loss 曲线
    if (data.loss_plot || data.loss_plot_option) {
      _state.lossPlot = data.loss_plot || _state.lossPlot;
      drawLossChart(data.loss_plot_option || {}, data.loss_plot || {});
    }
    // 结果就绪时获取指标和预测
    if (data.result_ready) {
      try {
        const result = await apiGet('/api/result');
        renderMetrics(result);
        const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
        const ranges = result?.prediction_selector?.ranges || {};
        if (shape.length >= 4 || ranges.horizon) {
          applyPredictionSelectorMeta(result);
          refreshPredictionSeries();
        }
      } catch (_) {}
    }
    if (data.running) {
      setTimeout(pollTrainLogs, 1500);
    }
  } catch (e) {
    setTimeout(pollTrainLogs, 2000);
  }
}

/* ═══════════════════════ RESUME TRAINING ═══════════════════════ */
function formatResumeRunTag(run) {
  const latest = Number(run?.latest_epoch);
  const latestText = Number.isFinite(latest)
    ? `latest=${latest}`
    : 'latest=-';
  return `${run.run_id} | ${run.model || '-'} | ${run.dataset || '-'} | ${latestText}`;
}

function renderResumeHint(run) {
  const el = document.getElementById('resume_hint');
  if (!el) return;
  if (!run) {
    el.textContent = t('resume_hint_empty', '请选择一个可续训的运行目录');
    return;
  }
  const epochs = Array.isArray(run.epochs) ? run.epochs : [];
  const latest = Number(run.latest_epoch);
  const minEpoch = epochs.length ? epochs[0] : '-';
  const maxEpoch = Number.isFinite(latest) ? latest : '-';
  el.textContent = `run=${run.run_id}, model=${run.model || '-'}, dataset=${run.dataset || '-'}, checkpoints=${minEpoch}..${maxEpoch}`;
}

function renderResumeEpochOptions(run) {
  const epochEl = document.getElementById('resume_epoch');
  if (!epochEl) return;
  const epochsRaw = Array.isArray(run?.epochs) ? run.epochs : [];
  const epochs = epochsRaw
    .map(x => Number(x))
    .filter(x => Number.isFinite(x) && x >= 0)
    .map(x => Math.floor(x))
    .sort((a, b) => a - b);
  if (!epochs.length) {
    epochEl.innerHTML = '<option value=""></option>';
    epochEl.value = '';
    epochEl.disabled = true;
    return;
  }
  let selected = Math.floor(Number(epochEl.value));
  if (!epochs.includes(selected)) {
    selected = Math.floor(Number(run?.latest_epoch));
  }
  if (!epochs.includes(selected)) {
    selected = epochs[epochs.length - 1];
  }
  epochEl.innerHTML = epochs
    .map(epoch => `<option value="${epoch}" ${epoch === selected ? 'selected' : ''}>${epoch}</option>`)
    .join('');
  epochEl.value = String(selected);
  epochEl.disabled = false;
}

function onResumeRunChanged() {
  const runId = document.getElementById('resume_run_id')?.value || '';
  const run = (_resumeRuns || []).find(x => x.run_id === runId);
  populateDataVersionSelects(_state.dataVersions || []);
  renderResumeHint(run || null);
  renderResumeEpochOptions(run || null);
  if (!run) return;
  const resumeEpoch = Math.floor(Number(document.getElementById('resume_epoch').value));
  if (!Number.isFinite(resumeEpoch) || resumeEpoch < 0) return;
  const currentMax = Math.max(1, Math.floor(Number(document.getElementById('resume_max_epoch').value || (resumeEpoch + 10))));
  document.getElementById('resume_max_epoch').value = String(Math.max(currentMax, resumeEpoch + 1));
}

async function loadResumeRuns() {
  try {
    const data = await apiGet('/api/resume_runs');
    _resumeRuns = Array.isArray(data.runs) ? data.runs : [];
    renderResumeRunOptions();
    populateDataVersionSelects(_state.dataVersions || []);
  } catch (e) {
    console.error('loadResumeRuns failed:', e);
  }
}

function renderResumeRunOptions() {
  const el = document.getElementById('resume_run_id');
  if (!el) return;
  const selected = el.value;
  const list = _resumeRuns || [];
  el.innerHTML = '<option value=""></option>' +
    list.map(run => {
      const sel = selected && selected === run.run_id ? 'selected' : '';
      return `<option value="${escHtml(run.run_id)}" ${sel}>${escHtml(formatResumeRunTag(run))}</option>`;
    }).join('');
  if (!el.value && list.length) el.value = list[0].run_id;
  onResumeRunChanged();
}

async function startResumeTrain() {
  const runId = (document.getElementById('resume_run_id').value || '').trim();
  if (!runId) {
    alert(t('resume_select_first', '请先选择可续训模型'));
    return;
  }
  const run = (_resumeRuns || []).find(x => x.run_id === runId);
  if (!run) {
    alert(t('resume_run_not_found', '未找到对应运行目录'));
    return;
  }
  if (!run.task || !run.model || !run.dataset) {
    alert(t('resume_run_meta_missing', '所选运行目录缺少 task/model/dataset 信息，无法继续训练'));
    return;
  }
  const dataVersionId = String(document.getElementById('resume_data_version')?.value || '').trim();
  if (!dataVersionId) {
    alert('请先选择可用的 data_version_id');
    return;
  }
  const resumeEpoch = Math.floor(Number(document.getElementById('resume_epoch').value));
  if (!Number.isFinite(resumeEpoch) || resumeEpoch < 0) {
    alert(t('resume_invalid_epoch', '请选择有效的续训 epoch'));
    return;
  }
  const availableEpochs = (Array.isArray(run.epochs) ? run.epochs : []).map(x => Math.floor(Number(x)));
  if (!availableEpochs.includes(resumeEpoch)) {
    alert(t('resume_invalid_epoch', '请选择有效的续训 epoch'));
    return;
  }
  const targetMaxEpoch = Math.max(1, Math.floor(Number(document.getElementById('resume_max_epoch').value || (resumeEpoch + 1))));
  if (targetMaxEpoch <= resumeEpoch) {
    alert(t('resume_invalid_max_epoch', '目标 max_epoch 必须大于续训起始 epoch'));
    return;
  }
  document.getElementById('resume_epoch').value = String(resumeEpoch);
  document.getElementById('resume_max_epoch').value = String(targetMaxEpoch);
  try {
    await apiPost('/api/start_resume', {
      task: run.task,
      model: run.model,
      dataset: run.dataset,
      data_version_id: dataVersionId,
      saved_model: document.getElementById('resume_saved_model').value,
      train: true,
      extra_args: document.getElementById('resume_extra_args').value || '',
      cli_options: {exp_id: run.run_id},
      config: {epoch: resumeEpoch, max_epoch: targetMaxEpoch},
    });
    // Start polling resume logs
    pollResumeLogs();
  } catch (e) {
    alert(`继续训练启动失败: ${e.message}`);
  }
}

async function pollResumeLogs() {
  try {
    const data = await apiGet(`/api/status?since=${resumeLogIndex}`);
    if (data.lines && data.lines.length) {
      _resumeLogLines.push(...data.lines);
      resumeLogIndex = data.next_index || resumeLogIndex;
      renderFilteredLogs('resume_logs', _resumeLogLines);
    }
    const badge = document.getElementById('resume_status');
    if (badge) {
      badge.className = `badge badge-${data.status || 'idle'}`;
      badge.textContent = data.status || 'idle';
    }
    if (data.running) {
      setTimeout(pollResumeLogs, 1000);
    }
  } catch (e) {
    setTimeout(pollResumeLogs, 2000);
  }
}

/* ═══════════════════════ COMPARE ═══════════════════════ */
function renderCompareRunOptions() {
  const list = _compareRuns || [];
  const render = (id, selected) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '<option value=""></option>' +
      list.map(r => {
        const tag = `${r.run_id} | ${r.model || '-'} | ${r.dataset || '-'}`;
        const sel = selected && selected === r.run_id ? 'selected' : '';
        return `<option value="${escHtml(r.run_id)}" ${sel}>${escHtml(tag)}</option>`;
      }).join('');
  };
  const a = document.getElementById('compare_run_a').value;
  const b = document.getElementById('compare_run_b').value;
  const c = document.getElementById('compare_run_c').value;
  render('compare_run_a', a);
  render('compare_run_b', b);
  render('compare_run_c', c);
  if (!document.getElementById('compare_run_a').value && list.length) {
    document.getElementById('compare_run_a').value = list[0].run_id;
  }
  if (!document.getElementById('compare_run_b').value && list.length > 1) {
    document.getElementById('compare_run_b').value = list[1].run_id;
  }
}

async function loadCompareRuns() {
  try {
    const data = await apiGet('/api/runs');
    _compareRuns = Array.isArray(data.runs) ? data.runs : [];
    renderCompareRunOptions();
  } catch (e) {
    console.error('loadCompareRuns failed:', e);
  }
}

function renderCompareMetrics(items) {
  const table = document.getElementById('compare_metrics');
  if (!table) return;
  if (!items || !items.length) {
    table.innerHTML = `<tr><td>${escHtml(t('compare_no_data', '无可对比数据'))}</td></tr>`;
    return;
  }
  const metrics = new Set();
  items.forEach(it => Object.keys(it.metrics_summary || {}).forEach(k => metrics.add(k)));
  const metricList = Array.from(metrics);
  let html = `<tr><th>${escHtml(t('compare_run_header', 'run'))}</th>`;
  metricList.forEach(m => {
    html += `<th>${escHtml(m)}(${escHtml(t('compare_h1', 'h1'))})</th><th>${escHtml(m)}(${escHtml(t('compare_avg', 'avg'))})</th><th>${escHtml(m)}(${escHtml(t('compare_best', 'best'))})</th>`;
  });
  html += '</tr>';
  items.forEach(it => {
    html += `<tr><td class="mono">${escHtml(it.run_id)}</td>`;
    metricList.forEach(m => {
      const x = it.metrics_summary?.[m];
      if (!x) html += '<td>-</td><td>-</td><td>-</td>';
      else html += `<td>${Number(x.h1).toFixed(4)}</td><td>${Number(x.avg).toFixed(4)}</td><td>${Number(x.best).toFixed(4)}</td>`;
    });
    html += '</tr>';
  });
  table.innerHTML = html;
}

async function loadComparison() {
  const runIds = [
    document.getElementById('compare_run_a')?.value,
    document.getElementById('compare_run_b')?.value,
    document.getElementById('compare_run_c')?.value,
  ].filter(Boolean);
  if (!runIds.length) {
    document.getElementById('compare_metrics').innerHTML = `<tr><td>${escHtml(t('compare_select_model_first', '请先选择模型'))}</td></tr>`;
    return;
  }
  try {
    const data = await apiPost('/api/compare', {run_ids: runIds});
    renderCompareMetrics(data.items || []);
    renderEChart('chartCompare', data.chart_option || {});
  } catch (e) {
    alert(`对比失败: ${e.message}`);
  }
}

/* ═══════════════════════ HISTORY ═══════════════════════ */
function formatDuration(sec) {
  if (!Number.isFinite(Number(sec)) || Number(sec) < 0) return '-';
  const s = Math.floor(Number(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m ${r}s`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

function formatTime(ts) {
  if (!Number.isFinite(Number(ts))) return '-';
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return '-';
  return d.toLocaleString();
}

function renderHistoryTable(items) {
  const table = document.getElementById('history_table');
  if (!table) return;
  if (!Array.isArray(items) || !items.length) {
    table.innerHTML = `<tr><td>${escHtml(t('history_empty', '暂无历史记录'))}</td></tr>`;
    return;
  }
  let html = `<tr>
    <th>${escHtml(t('history_col_status', '状态'))}</th>
    <th>${escHtml(t('history_col_run', 'Run'))}</th>
    <th>${escHtml(t('history_col_model', '模型'))}</th>
    <th>${escHtml(t('history_col_dataset', '数据集'))}</th>
    <th>${escHtml(t('history_col_duration', '耗时'))}</th>
    <th>${escHtml(t('history_col_metrics', '主要指标'))}</th>
    <th>${escHtml(t('history_col_output', '输出目录'))}</th>
    <th>${escHtml(t('history_col_end_time', '结束时间'))}</th>
  </tr>`;
  for (const it of items) {
    const mm = it.major_metrics || {};
    const metricText = Object.keys(mm).map(k => `${k}=${Number(mm[k]).toFixed(4)}`).join(', ') || '-';
    html += `<tr>
      <td>${escHtml(t('history_col_' + (it.status || 'finished'), it.status || 'finished'))}</td>
      <td class="mono">${escHtml(it.run_id || '-')}</td>
      <td>${escHtml(it.model || '-')}</td>
      <td>${escHtml(it.dataset || '-')}</td>
      <td>${escHtml(formatDuration(it.duration_sec))}</td>
      <td>${escHtml(metricText)}</td>
      <td class="mono">${escHtml(it.output_dir || '-')}</td>
      <td>${escHtml(formatTime(it.ended_at))}</td>
    </tr>`;
  }
  table.innerHTML = html;
}

async function loadHistory() {
  const limitInput = document.getElementById('history_limit');
  const n = clampTopK(limitInput?.value, 20);
  if (limitInput) limitInput.value = String(n);
  try {
    const data = await apiGet(`/api/history?limit=${encodeURIComponent(n)}`);
    renderHistoryTable(Array.isArray(data.items) ? data.items : []);
  } catch (e) {
    console.error('loadHistory failed:', e);
  }
}

function clampTopK(v, d) {
  const n = Number(v);
  if (!Number.isFinite(n)) return d;
  return Math.max(1, Math.min(100, Math.floor(n)));
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
  const sourceId = String(_state.selectedDataVersionId || document.getElementById('train_data_version')?.value || '').trim();
  const trainSel = document.getElementById('train_data_version');
  if (!sourceId || !trainSel) {
    alert('请先选择有效的 data_version_id');
    return;
  }
  const version = (_state.dataVersions || []).find(v => v.version_id === sourceId);
  if (!version) {
    alert('请先选择有效的 data_version_id');
    return;
  }
  if (String(version.status || '').toLowerCase() !== 'ready') {
    alert('仅 ready 数据版本可同步到训练参数');
    return;
  }
  trainSel.value = sourceId;
  applyTrainDataVersionSelection();
  loadDefaults();
}

/* ═══════════════════════ ECHARTS ═══════════════════════ */
function renderEChart(containerId, option) {
  const dom = document.getElementById(containerId);
  if (!dom || typeof echarts === 'undefined') return;
  let chart = _charts[containerId];
  if (!chart) {
    chart = echarts.init(dom);
    _charts[containerId] = chart;
  }
  // Revive function strings from JSON
  const reviveFn = (node) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { node.forEach(reviveFn); return; }
    for (const [k, v] of Object.entries(node)) {
      if ((k === 'formatter' || k === 'valueFormatter') && typeof v === 'string') {
        const s = v.trim();
        if (s.startsWith('function(') || s.startsWith('(function(')) {
          try { node[k] = new Function(`return (${s})`)(); } catch (_) {}
        }
      } else if (v && typeof v === 'object') {
        reviveFn(v);
      }
    }
  };
  const opt = option || {};
  reviveFn(opt);
  try {
    chart.setOption(opt, true);
  } catch (err) {
    console.warn(`renderEChart failed: ${containerId}`, err);
  }
}

function updateLossSummary(plot) {
  const xs = plot?.epochs || [];
  const maxEpoch = plot?.max_epoch;
  const maxText = maxEpoch ? ` / ${maxEpoch}` : '';
  const el = document.getElementById('loss_summary');
  if (el) el.textContent = xs.length ? `已记录 ${xs.length} 个 epoch${maxText}` : '暂无 loss 数据';
}

function drawLossChart(option, plot) {
  updateLossSummary(plot);
  renderEChart('chartLoss', option || {});
}

function drawModelParamChart(plot, pieOption, barOption) {
  const total = plot?.total_params;
  const summaryEl = document.getElementById('model_param_summary');
  if (summaryEl) {
    summaryEl.textContent = total
      ? `总参数量: ${total.toLocaleString()} | 展示: Top 参数分布`
      : '暂无参数分布数据';
  }
  const useBar = _state.modelPlotType === 'bar';
  let option = useBar ? (barOption || {}) : (pieOption || {});
  if (!useBar) {
    const hasSeries = Array.isArray(option?.series) && option.series.length > 0;
    if (!hasSeries) {
      const labels = Array.isArray(plot?.labels) ? plot.labels : [];
      const counts = Array.isArray(plot?.counts) ? plot.counts : [];
      const pairs = labels.map((name, i) => ({name, value: Number(counts[i] || 0)}))
        .filter(x => Number.isFinite(x.value) && x.value > 0)
        .sort((a, b) => b.value - a.value);
      const k = 8;
      const top = pairs.slice(0, k);
      const rest = pairs.slice(k).reduce((s, x) => s + x.value, 0);
      if (rest > 0) top.push({name: 'Others', value: rest});
      const fallbackData = top.length ? top : [{name: '暂无数据', value: 1}];
      option = {
        tooltip: {trigger: 'item', formatter: '{b}: {c} ({d}%)'},
        legend: {type: 'scroll', orient: 'vertical', left: '68%', top: '12%'},
        series: [{
          type: 'pie',
          radius: ['35%', '65%'],
          center: ['40%', '55%'],
          data: fallbackData,
          label: {formatter: '{b}: {d}%'},
        }],
      };
    }
  }
  renderEChart('chartModelParams', option);
}

function clearResultPanels() {
  const metricsEl = document.getElementById('metrics');
  if (metricsEl) metricsEl.innerHTML = `<div class="small">${escHtml('等待训练全部完成后展示')}</div>`;
  const chart = _charts.chartPred;
  if (chart) chart.clear();
  resetPredictionSelectors();
}

function resetPredictionSelectors() {
  const setOpts = (id, count, selected) => {
    const el = document.getElementById(id);
    if (!el) return;
    const upper = Number.isFinite(Number(count)) ? Math.max(0, Math.floor(Number(count))) : 0;
    if (upper <= 0) {
      el.innerHTML = '<option value="1">1</option>';
      el.value = '1';
      el.disabled = true;
      return;
    }
    const safe = Math.min(Math.max(1, Number(selected) || 1), upper);
    const opts = [];
    for (let i = 1; i <= upper; i += 1) opts.push(`<option value="${i}" ${i === safe ? 'selected' : ''}>${i}</option>`);
    el.innerHTML = opts.join('');
    el.value = String(safe);
    el.disabled = false;
  };
  setOpts('pred_horizon', 0, 1);
  setOpts('pred_node', 0, 1);
  setOpts('pred_feature', 0, 1);
  _state.predictionRanges = {horizon: 0, node: 0, feature: 0};
  _state.predictionSelection = {horizon: 1, node: 1, feature: 1};
}

async function refreshPredictionSeries() {
  if ((_state.predictionRanges.horizon || 0) <= 0 ||
      (_state.predictionRanges.node || 0) <= 0 ||
      (_state.predictionRanges.feature || 0) <= 0) return;
  const horizon = Number(document.getElementById('pred_horizon')?.value || 1);
  const node = Number(document.getElementById('pred_node')?.value || 1);
  const feature = Number(document.getElementById('pred_feature')?.value || 1);
  _state.predictionSelection = {horizon, node, feature};
  try {
    const qs = new URLSearchParams({horizon: String(horizon), node: String(node), feature: String(feature)});
    const data = await apiGet(`/api/result_series?${qs.toString()}`);
    renderEChart('chartPred', data.chart_option || {});
  } catch (_) {}
}

function applyPredictionSelectorMeta(result) {
  const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
  const ranges = result?.prediction_selector?.ranges || {};
  const selection = result?.prediction_selector?.selection || {};
  const toInt = v => { const n = Number(v); return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0; };
  const horizonMax = toInt(ranges.horizon || shape[1] || 0);
  const nodeMax = toInt(ranges.node || shape[2] || 0);
  const featureMax = toInt(ranges.feature || shape[3] || 0);
  const setOpts = (id, max, sel) => {
    const el = document.getElementById(id);
    if (!el) return;
    const upper = Math.max(0, Math.floor(max));
    if (upper <= 0) { el.innerHTML = '<option value="1">1</option>'; el.value = '1'; el.disabled = true; return; }
    const safe = Math.min(Math.max(1, Number(sel) || 1), upper);
    const opts = [];
    for (let i = 1; i <= upper; i += 1) opts.push(`<option value="${i}" ${i === safe ? 'selected' : ''}>${i}</option>`);
    el.innerHTML = opts.join(''); el.value = String(safe); el.disabled = false;
  };
  setOpts('pred_horizon', horizonMax, selection.horizon || _state.predictionSelection.horizon);
  setOpts('pred_node', nodeMax, selection.node || _state.predictionSelection.node);
  setOpts('pred_feature', featureMax, selection.feature || _state.predictionSelection.feature);
  _state.predictionRanges = {horizon: horizonMax, node: nodeMax, feature: featureMax};
  _state.predictionSelection = {
    horizon: Number(document.getElementById('pred_horizon')?.value || 1),
    node: Number(document.getElementById('pred_node')?.value || 1),
    feature: Number(document.getElementById('pred_feature')?.value || 1),
  };
}

function renderMetrics(result) {
  const metricsEl = document.getElementById('metrics');
  if (metricsEl) {
    metricsEl.innerHTML = result.metrics_table_html || '<div class="small">无数据</div>';
  }
}

function initCharts() {
  // Predictor controls binding
  document.getElementById('pred_horizon')?.addEventListener('change', refreshPredictionSeries);
  document.getElementById('pred_node')?.addEventListener('change', refreshPredictionSeries);
  document.getElementById('pred_feature')?.addEventListener('change', refreshPredictionSeries);

  // Model plot type switch
  document.getElementById('model_plot_type')?.addEventListener('change', e => {
    _state.modelPlotType = e.target.value === 'bar' ? 'bar' : 'pie';
    drawModelParamChart(_state.modelPlot, _state.modelPlotOptionPie, _state.modelPlotOptionBar);
  });
}

/* ═══════════════════════ INIT ═══════════════════════ */
function init() {
  // Tab 切换绑定（含按需加载）
  document.querySelectorAll('.sidebar-nav-item').forEach(btn => {
    btn.addEventListener('click', async () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
      if (tab === 'resume') {
        if (!_resumeRuns.length) await loadResumeRuns();
      } else if (tab === 'compare') {
        if (!_compareRuns.length) await loadCompareRuns();
      } else if (tab === 'history') {
        await loadHistory();
      }
    });
  });

  // 主题切换
  document.getElementById('themeToggleSidebar')?.addEventListener('click', toggleTheme);
  setTheme(DEFAULT_THEME);

  // Tweaks
  setupTweaks();

  // Split 预设
  document.getElementById('data_split_preset')?.addEventListener('change', applySplitPreset);

  // 数据版本同步到训练
  document.getElementById('btnSyncToTrain')?.addEventListener('click', applySelectedDataVersionToTrain);

  // i18n 加载（自动调用 applyI18n）
  loadI18n(DEFAULT_LANG);

  // 语言切换
  const langSelect = document.getElementById('lang_select');
  if (langSelect) {
    langSelect.value = DEFAULT_LANG;
    langSelect.addEventListener('change', e => loadI18n(e.target.value));
  }

  // 参数表过滤
  document.getElementById('param_filter')?.addEventListener('input', renderParamTable);

  // 数据版本切换 → 加载默认参数
  document.getElementById('train_data_version')?.addEventListener('change', () => {
    applyTrainDataVersionSelection();
    loadDefaults();
  });

  // 继续训练 — run 切换
  document.getElementById('resume_run_id')?.addEventListener('change', onResumeRunChanged);

  // 日志过滤
  document.getElementById('log_level_filter')?.addEventListener('change', applyLogFilter);
  document.getElementById('log_keyword_filter')?.addEventListener('input', applyLogFilter);

  // 加载元数据
  loadMeta();
  loadDataVersions();

  // Charts（含绑定）
  initCharts();
}

document.addEventListener('DOMContentLoaded', init);
