const query = new URLSearchParams(window.location.search);
const DEFAULT_LANG = query.get('lang') || window.localStorage.getItem('train_web_lang') || 'zh-CN';
const DEFAULT_THEME = query.get('theme') || window.localStorage.getItem('train_web_theme') || 'light';
const DEFAULT_MODEL_PLOT_TOPK_PIE = Number(window.localStorage.getItem('train_web_model_plot_topk_pie') || 8);
const DEFAULT_MODEL_PLOT_TOPK_BAR = Number(window.localStorage.getItem('train_web_model_plot_topk_bar') || 12);
const DEFAULT_CHART_SCALE = Number(window.localStorage.getItem('train_web_chart_scale') || 100);
const state = {
    models: [],
    datasets: [],
    paramRowsConfig: [],
    paramRowsExecutor: [],
    i18n: {params: {}, ui: {}},
    lang: DEFAULT_LANG,
    theme: DEFAULT_THEME,
    logAutoScroll: true,
    modelPlotType: 'pie',
    modelPlotTopKPie: Number.isFinite(DEFAULT_MODEL_PLOT_TOPK_PIE) ? DEFAULT_MODEL_PLOT_TOPK_PIE : 8,
    modelPlotTopKBar: Number.isFinite(DEFAULT_MODEL_PLOT_TOPK_BAR) ? DEFAULT_MODEL_PLOT_TOPK_BAR : 12,
    chartScale: Number.isFinite(DEFAULT_CHART_SCALE) ? DEFAULT_CHART_SCALE : 100,
    activeRunKey: '',
    lossRenderedEpochCount: -1,
    resultRenderedRunKey: '',
    modelPlot: {},
    modelPlotOptionPie: {},
    modelPlotOptionBar: {},
    lossPlot: {},
    compareRuns: [],
    resumeRuns: [],
    dataVersions: [],
    dataStatus: {},
    dataPreview: {},
    dataVersionsDatasetFilter: '',
    latestLogs: [],
    logFilterLevel: 'all',
    logFilterKeyword: '',
    activeTab: 'train',
    predictionRanges: {horizon: 0, node: 0, feature: 0},
    predictionSelection: {horizon: 1, node: 1, feature: 1},
};
const byId = (id) => document.getElementById(id);
const esc = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const CLI_FIELDS = ["config_file", "exp_id", "seed", "gpu", "gpu_id", "train_rate", "eval_rate", "batch_size", "learning_rate", "max_epoch", "dataset_class", "executor", "evaluator"];
const HIDDEN_TRAIN_PARAM_KEYS = new Set(["config_file", "train_rate", "eval_rate", "dataset_class", "task", "model", "dataset", "seed"]);
const normalizeLang = (lang) => {
    const x = String(lang || '').trim();
    if (!x) return 'zh-CN';
    const lower = x.toLowerCase();
    if (lower === 'ja-jp') return 'ja-JP';
    if (lower === 'en-us') return 'en-US';
    if (lower === 'zh-cn') return 'zh-CN';
    return x;
};
const t = (key, fallback) => state.i18n?.ui?.[key] || fallback || key;
const tf = (key, vars, fallback) => {
    const text = t(key, fallback);
    return String(text).replace(/\{(\w+)\}/g, (_, k) => String(vars?.[k] ?? ''));
};

function getParamDisplayName(key) {
    return state.i18n?.params?.[key] || key;
}

function applyI18nLabels() {
    document.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute('data-i18n');
        if (!key) return;
        if (!el.dataset.i18nFallback) el.dataset.i18nFallback = el.innerText;
        el.innerText = t(key, el.dataset.i18nFallback);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (!key) return;
        if (!el.dataset.i18nPlaceholderFallback) el.dataset.i18nPlaceholderFallback = el.getAttribute('placeholder') || '';
        el.setAttribute('placeholder', t(key, el.dataset.i18nPlaceholderFallback));
    });
    document.querySelectorAll('[data-param-label]').forEach((el) => {
        const key = el.getAttribute('data-param-label');
        if (!key) return;
        el.innerText = getParamDisplayName(key);
    });
    byId('param_title_config').innerText = getParamDisplayName('config');
    byId('param_title_executor').innerText = getParamDisplayName('executor');
    const localizedName = getParamDisplayName('localized_name');
    const rawName = getParamDisplayName('raw_name');
    const typeName = getParamDisplayName('type');
    const valueName = getParamDisplayName('value');
    byId('th_config_name_localized').innerText = localizedName;
    byId('th_config_name_raw').innerText = rawName;
    byId('th_config_type').innerText = typeName;
    byId('th_config_value').innerText = valueName;
    byId('th_executor_name_localized').innerText = localizedName;
    byId('th_executor_name_raw').innerText = rawName;
    byId('th_executor_type').innerText = typeName;
    byId('th_executor_value').innerText = valueName;
    updateAutoScrollBtnText();
}

async function loadI18n(lang) {
    const normalized = normalizeLang(lang);
    state.lang = normalized;
    window.localStorage.setItem('train_web_lang', normalized);
    try {
        const r = await fetch(`/static/i18n/${encodeURIComponent(normalized)}.json`);
        if (r.ok) {
            state.i18n = await r.json();
            document.documentElement.lang = state.i18n?.lang || normalized;
        } else {
            state.i18n = {params: {}, ui: {}};
        }
    } catch (_) {
        state.i18n = {params: {}, ui: {}};
    }
    applyI18nLabels();
    // Parameter rows use localized labels from i18n map, so re-render after language switch.
    renderParamTable();
    drawModelParamChart(state.modelPlot, state.modelPlotOptionPie, state.modelPlotOptionBar);
    updateLossSummary(state.lossPlot || {});
    renderLogs(state.latestLogs || []);
    if (state.activeTab === 'history') await loadHistory();
    if (state.activeTab === 'resume') await loadResumeRuns();
    if (state.activeTab === 'data') await loadDataVersions();
    const select = byId('lang_select');
    if (select) select.value = normalized;
}

function applyTheme(theme) {
    const mode = theme === 'dark' ? 'dark' : 'light';
    state.theme = mode;
    window.localStorage.setItem('train_web_theme', mode);
    const href = mode === 'dark' ? '/static/train_web_flask_dark.css' : '/static/train_web_flask.css';
    byId('theme_stylesheet').setAttribute('href', href);
}

function clampTopK(v, d) {
    const n = Number(v);
    if (!Number.isFinite(n)) return d;
    return Math.max(1, Math.min(100, Math.floor(n)));
}

function persistModelPlotTopK() {
    window.localStorage.setItem('train_web_model_plot_topk_pie', String(state.modelPlotTopKPie));
    window.localStorage.setItem('train_web_model_plot_topk_bar', String(state.modelPlotTopKBar));
}

async function pushModelPlotTopK() {
    try {
        await fetch('/api/plot_settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                pie_topk: state.modelPlotTopKPie,
                bar_topk: state.modelPlotTopKBar,
            }),
        });
    } catch (_) {
        // Keep local setting even if server sync temporarily fails.
    }
}

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
        throw new Error(tf('error_bool_value', {raw}, `布尔值错误: ${raw}`));
    }
    if (t === 'int') {
        const n = parseInt(String(raw).trim(), 10);
        if (Number.isNaN(n)) throw new Error(tf('error_int_value', {raw}, `整数错误: ${raw}`));
        return n;
    }
    if (t === 'float') {
        const n = Number(String(raw).trim());
        if (Number.isNaN(n)) throw new Error(tf('error_float_value', {raw}, `浮点数错误: ${raw}`));
        return n;
    }
    if (t === 'json') {
        const txt = String(raw).trim();
        return txt ? JSON.parse(txt) : null;
    }
    return String(raw);
}

function renderTasks() {
    const tasks = [...new Set(state.models.map(x => x.task))].sort();
    const html = tasks.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
    const dataTaskEl = byId('data_task');
    if (!dataTaskEl) return;
    dataTaskEl.innerHTML = html;
    const stgcnMeta = state.models.find((x) => x.model === 'STGCN');
    if (stgcnMeta && tasks.includes(stgcnMeta.task)) {
        dataTaskEl.value = stgcnMeta.task;
    }
}

function renderModelsForTask() {
    renderDataModelsForTask();
}

function renderDataModelsForTask() {
    const task = byId('data_task')?.value || '';
    const models = [...new Set(state.models.filter(x => x.task === task).map(x => x.model))].sort();
    if (byId('data_model')) {
        byId('data_model').innerHTML = models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
        if (models.includes('STGCN')) byId('data_model').value = 'STGCN';
    }
}

function renderDatasets() {
    const html = state.datasets.map(d => `<option value="${esc(d)}">${esc(d)}</option>`).join('');
    if (byId('dataset')) byId('dataset').innerHTML = html;
    if (byId('data_dataset')) byId('data_dataset').innerHTML = html;
    if (state.datasets.includes('PEMSD4')) {
        if (byId('dataset')) byId('dataset').value = 'PEMSD4';
        if (byId('data_dataset')) byId('data_dataset').value = 'PEMSD4';
    }
    const filterEl = byId('data_versions_dataset_filter');
    if (filterEl) {
        const selected = String(state.dataVersionsDatasetFilter || '');
        const options = ['<option value="">全部数据集</option>'].concat(
            state.datasets.map((d) => `<option value="${esc(d)}" ${selected === d ? 'selected' : ''}>${esc(d)}</option>`)
        );
        filterEl.innerHTML = options.join('');
    }
}

function renderParamCell(section, idx) {
    if (idx === null) return '';
    const rows = section === 'executor' ? state.paramRowsExecutor : state.paramRowsConfig;
    const row = rows[idx];
    const valueEditor = row.type === 'json'
        ? `<textarea class="param-value mono value-input value-input-json" data-section="${section}" data-idx="${idx}" rows="4">${esc(row.value)}</textarea>`
        : `<input class="param-value mono value-input" data-section="${section}" data-idx="${idx}" value="${esc(row.value)}"/>`;
    return `
    <td class="param-localized">${esc(getParamDisplayName(row.key))}</td>
    <td class="mono param-key">${esc(row.key)}</td>
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
    const filter = byId('param_filter').value.trim().toLowerCase();
    const configIds = state.paramRowsConfig
        .map((r, i) => ({r, i}))
        .filter(x => !filter || x.r.key.toLowerCase().includes(filter))
        .map(x => x.i);
    const executorIds = state.paramRowsExecutor
        .map((r, i) => ({r, i}))
        .filter(x => !filter || x.r.key.toLowerCase().includes(filter))
        .map(x => x.i);
    byId('param_count').innerText = String(configIds.length + executorIds.length);
    byId('param_count_config').innerText = String(configIds.length);
    byId('param_count_executor').innerText = String(executorIds.length);
    const bodyConfig = byId('param_tbody_config');
    const bodyExecutor = byId('param_tbody_executor');
    bodyConfig.innerHTML = configIds.map((idx) => `<tr>${renderParamCell('config', idx)}</tr>`).join('')
        || `<tr><td colspan="4" class="small">${esc(t('table_no_params', '无参数'))}</td></tr>`;
    bodyExecutor.innerHTML = executorIds.map((idx) => `<tr>${renderParamCell('executor', idx)}</tr>`).join('')
        || `<tr><td colspan="4" class="small">${esc(t('table_no_params', '无参数'))}</td></tr>`;

    document.querySelectorAll('#param_tbody_config .type-select, #param_tbody_executor .type-select').forEach(el => {
        el.addEventListener('change', (e) => {
            const section = e.target.getAttribute('data-section');
            const idx = Number(e.target.getAttribute('data-idx'));
            const rows = section === 'executor' ? state.paramRowsExecutor : state.paramRowsConfig;
            rows[idx].type = e.target.value;
            renderParamTable();
        });
    });
    document.querySelectorAll('#param_tbody_config .value-input, #param_tbody_executor .value-input').forEach(el => {
        el.addEventListener('input', (e) => {
            const section = e.target.getAttribute('data-section');
            const idx = Number(e.target.getAttribute('data-idx'));
            const rows = section === 'executor' ? state.paramRowsExecutor : state.paramRowsConfig;
            rows[idx].value = e.target.value;
        });
    });
}

function resetParamTypes() {
    state.paramRowsConfig.forEach(r => r.type = inferType(r.defaultValue));
    state.paramRowsExecutor.forEach(r => r.type = inferType(r.defaultValue));
    renderParamTable();
}

function collectConfigFromTable() {
    const cfg = {};
    for (const r of state.paramRowsConfig) cfg[r.key] = parseValue(r.type, r.value);
    for (const r of state.paramRowsExecutor) cfg[r.key] = parseValue(r.type, r.value);
    return cfg;
}

function applyDefaultToCliFields(config) {
    const setIf = (k, v) => {
        if (v !== undefined && v !== null && byId(k)) byId(k).value = String(v);
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
    if (config.gpu !== undefined && byId('gpu')) byId('gpu').value = String(config.gpu).toLowerCase();
}

function collectCliOptions() {
    const out = {};
    for (const k of CLI_FIELDS) {
        const el = byId(k);
        if (!el) continue;
        const v = (el.value ?? '').trim();
        if (v !== '') out[k] = v;
    }
    return out;
}

function formatSplitRate(v) {
    return Number(v).toFixed(3);
}

function parseSplitRate(raw, name) {
    const text = String(raw ?? '').trim();
    if (!text) throw new Error(`${name} 不能为空`);
    const n = Number(text);
    if (!Number.isFinite(n)) throw new Error(`${name} 不是有效数字`);
    if (n < 0 || n > 1) throw new Error(`${name} 需在 [0, 1]`);
    return n;
}

function getDataSplitValues() {
    const train = parseSplitRate(byId('data_train_rate')?.value, 'train');
    const evalR = parseSplitRate(byId('data_eval_rate')?.value, 'eval');
    const test = parseSplitRate(byId('data_test_rate')?.value, 'test');
    const sum = train + evalR + test;
    if (Math.abs(sum - 1) > 1e-6) {
        throw new Error(`train+eval+test 必须等于 1（当前 ${sum.toFixed(6)}）`);
    }
    return {train, eval: evalR, test};
}

function setDataSplitValues(split) {
    if (byId('data_train_rate')) byId('data_train_rate').value = formatSplitRate(split.train);
    if (byId('data_eval_rate')) byId('data_eval_rate').value = formatSplitRate(split.eval);
    if (byId('data_test_rate')) byId('data_test_rate').value = formatSplitRate(split.test);
    updateDataSplitHint();
    syncDataSplitPreset();
}

function parsePresetValue(raw) {
    const parts = String(raw || '').split(',').map((x) => Number(String(x).trim()));
    if (parts.length !== 3 || parts.some((x) => !Number.isFinite(x))) return null;
    return {train: parts[0], eval: parts[1], test: parts[2]};
}

function applyDataSplitPreset() {
    const presetEl = byId('data_split_preset');
    if (!presetEl) return;
    const parsed = parsePresetValue(presetEl.value);
    if (!parsed) return;
    setDataSplitValues(parsed);
}

function syncDataSplitPreset() {
    const presetEl = byId('data_split_preset');
    if (!presetEl) return;
    let current;
    try {
        current = getDataSplitValues();
    } catch (_) {
        presetEl.value = '';
        return;
    }
    let matched = '';
    for (const opt of Array.from(presetEl.options)) {
        if (!opt.value) continue;
        const p = parsePresetValue(opt.value);
        if (!p) continue;
        if (
            Math.abs(p.train - current.train) < 1e-9 &&
            Math.abs(p.eval - current.eval) < 1e-9 &&
            Math.abs(p.test - current.test) < 1e-9
        ) {
            matched = opt.value;
            break;
        }
    }
    presetEl.value = matched;
}

function updateDataSplitHint() {
    const hintEl = byId('data_split_ratio_hint');
    if (!hintEl) return;
    try {
        const split = getDataSplitValues();
        hintEl.innerText = `train/eval/test = ${formatSplitRate(split.train)} / ${formatSplitRate(split.eval)} / ${formatSplitRate(split.test)}`;
    } catch (e) {
        hintEl.innerText = String(e.message || 'split 配置错误');
    }
}

function initDataSplitControls() {
    const trainEl = byId('data_train_rate');
    const evalEl = byId('data_eval_rate');
    const testEl = byId('data_test_rate');
    const presetEl = byId('data_split_preset');
    if (!trainEl || !evalEl || !testEl) return;
    try {
        setDataSplitValues(getDataSplitValues());
    } catch (_) {
        setDataSplitValues({train: 0.6, eval: 0.2, test: 0.2});
    }
    trainEl.addEventListener('input', () => {
        updateDataSplitHint();
        syncDataSplitPreset();
    });
    evalEl.addEventListener('input', () => {
        updateDataSplitHint();
        syncDataSplitPreset();
    });
    testEl.addEventListener('input', () => {
        updateDataSplitHint();
        syncDataSplitPreset();
    });
    if (presetEl) presetEl.addEventListener('change', applyDataSplitPreset);
}

function collectDataCliOptions() {
    const out = {};
    const mappings = {
        seed: 'data_seed',
        batch_size: 'data_batch_size',
        dataset_class: 'data_dataset_class',
    };
    for (const [k, id] of Object.entries(mappings)) {
        const el = byId(id);
        if (!el) continue;
        const v = String(el.value ?? '').trim();
        if (v !== '') out[k] = v;
    }
    const split = getDataSplitValues();
    out.train_rate = String(split.train);
    out.eval_rate = String(split.eval);
    return out;
}

function setDataStatus(stateName, extra) {
    const el = byId('data_status');
    if (!el) return;
    el.className = 'badge';
    if (stateName === 'running') el.classList.add('status-running');
    else if (stateName === 'finished') el.classList.add('status-finished');
    else el.classList.add('status-idle');
    el.innerText = extra ? `${stateName} (${extra})` : stateName;
}

function setResumeStatus(stateName, extra) {
    const el = byId('resume_status');
    if (!el) return;
    el.className = 'badge';
    if (stateName === 'running') el.classList.add('status-running');
    else if (stateName === 'finished') el.classList.add('status-finished');
    else el.classList.add('status-idle');
    const localizedState = t(`status_${stateName}`, stateName);
    el.innerText = extra ? `${localizedState} (${extra})` : localizedState;
}

function toNumOrNull(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

function toIntOrNull(v) {
    const n = Number(v);
    return Number.isFinite(n) ? Math.floor(n) : null;
}

function getDataVersionProfile(version) {
    const v = version || {};
    const cli = v.cli_options || {};
    const cfg = v.config_payload || {};
    const seed = toIntOrNull(v.seed ?? cli.seed ?? cfg.seed);
    const trainRate = toNumOrNull(v.train_rate ?? cli.train_rate ?? cfg.train_rate);
    const evalRate = toNumOrNull(v.eval_rate ?? cli.eval_rate ?? cfg.eval_rate);
    let testRate = toNumOrNull(v.test_rate);
    if (testRate === null && trainRate !== null && evalRate !== null) {
        testRate = 1 - trainRate - evalRate;
    }
    return {seed, trainRate, evalRate, testRate};
}

function formatRate(v) {
    if (!Number.isFinite(v)) return '-';
    return Number(v).toFixed(3);
}

function formatDataVersionTag(v) {
    const status = String(v?.status || '-');
    const task = String(v?.task || '-');
    const model = String(v?.model || '-');
    const dataset = String(v?.dataset || '-');
    const p = getDataVersionProfile(v);
    const seedText = p.seed === null ? '-' : String(p.seed);
    const splitText = `${formatRate(p.trainRate)}/${formatRate(p.evalRate)}/${formatRate(p.testRate)}`;
    return `${v.version_id} | ${status} | ${task}/${model}/${dataset} | seed=${seedText} | split(T/V/Te)=${splitText}`;
}

function renderDataVersionHint(version) {
    const el = byId('data_version_hint');
    if (!el) return;
    if (!version) {
        el.innerText = '未选择数据版本';
        const noteEl = byId('data_version_note');
        if (noteEl) noteEl.value = '';
        return;
    }
    const s = version.script_meta || {};
    const cache = s.cache_file_name || '-';
    const trainB = s.train_batches ?? '-';
    const validB = s.valid_batches ?? '-';
    const testB = s.test_batches ?? '-';
    const p = getDataVersionProfile(version);
    const seedText = p.seed === null ? '-' : String(p.seed);
    const splitText = `${formatRate(p.trainRate)}/${formatRate(p.evalRate)}/${formatRate(p.testRate)}`;
    const note = String(version.note || '').trim();
    const noteText = note ? `, note=${note}` : '';
    el.innerText = `status=${version.status || '-'}, dataset=${version.dataset || '-'}, seed=${seedText}, split(train/val/test)=${splitText}, cache=${cache}, batches(train/val/test)=${trainB}/${validB}/${testB}${noteText}`;
    const noteEl = byId('data_version_note');
    if (noteEl) noteEl.value = note;
}

function renderDataPreview(payload) {
    const tableEl = byId('data_preview_table');
    const metaEl = byId('data_preview_meta');
    if (!tableEl || !metaEl) return;
    const data = payload || {};
    const columns = Array.isArray(data.columns) ? data.columns : [];
    const rows = Array.isArray(data.rows) ? data.rows : [];
    if (!columns.length) {
        tableEl.innerHTML = `<tr><td class="small">${esc(t('table_no_data', '无数据'))}</td></tr>`;
        metaEl.innerText = '暂无预览数据';
        return;
    }
    const thead = `<thead><tr>${columns.map((col) => `<th>${esc(col)}</th>`).join('')}</tr></thead>`;
    const tbodyRows = rows.map((row) => {
        return `<tr>${columns.map((col) => `<td>${esc(row?.[col] ?? '')}</td>`).join('')}</tr>`;
    }).join('');
    const tbody = `<tbody>${tbodyRows || `<tr><td colspan="${columns.length}" class="small">${esc(t('table_no_data', '无数据'))}</td></tr>`}</tbody>`;
    tableEl.innerHTML = `${thead}${tbody}`;
    const hasMoreText = data.has_more ? '（仅展示前 N 行）' : '';
    metaEl.innerText = `dataset=${data.dataset || '-'}, file=${data.file_name || '-'}, cols=${columns.length}, rows=${rows.length}${hasMoreText}`;
}

async function loadDataPreview() {
    const dataset = String(byId('data_dataset')?.value || '').trim();
    const fileType = String(byId('data_preview_file_type')?.value || 'dyna').trim();
    const rowInput = Math.floor(Number(byId('data_preview_rows')?.value || 20));
    const rows = Number.isFinite(rowInput) ? Math.max(1, Math.min(100, rowInput)) : 20;
    const entityId = String(byId('data_preview_entity_id')?.value || '').trim();
    const timeStart = String(byId('data_preview_time_start')?.value || '').trim();
    const timeEnd = String(byId('data_preview_time_end')?.value || '').trim();
    const columns = String(byId('data_preview_columns')?.value || '').trim();
    if (byId('data_preview_rows')) byId('data_preview_rows').value = String(rows);
    if (!dataset) {
        renderDataPreview(null);
        return;
    }
    const qs = new URLSearchParams({dataset, file_type: fileType, rows: String(rows)});
    if (entityId) qs.set('entity_id', entityId);
    if (timeStart) qs.set('time_start', timeStart);
    if (timeEnd) qs.set('time_end', timeEnd);
    if (columns) qs.set('columns', columns);
    const r = await fetch(`/api/data/preview?${qs.toString()}`);
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || '加载数据预览失败');
        renderDataPreview(null);
        return;
    }
    state.dataPreview = data;
    renderDataPreview(data);
}

function applyConfigPayloadToParamRows(configPayload) {
    if (!configPayload || typeof configPayload !== 'object') return;
    const rowMap = new Map();
    [...state.paramRowsConfig, ...state.paramRowsExecutor].forEach((row) => rowMap.set(row.key, row));
    for (const [k, v] of Object.entries(configPayload)) {
        const row = rowMap.get(k);
        if (row) {
            row.value = toInputString(v);
            row.type = inferType(v);
        }
    }
}

async function applySelectedDataVersionToTrain() {
    const sourceId = String(byId('data_version_select')?.value || getSelectedTrainDataVersion()).trim();
    if (!sourceId) {
        alert('请先选择有效的 data_version_id');
        return;
    }
    if (byId('train_data_version')) byId('train_data_version').value = sourceId;
    const version = (state.dataVersions || []).find((x) => x.version_id === sourceId);
    if (!version) {
        alert('请先选择有效的 data_version_id');
        return;
    }
    if (String(version.status || '').toLowerCase() !== 'ready') {
        alert('仅 ready 数据版本可同步到训练参数');
        return;
    }
    await loadDefaults();
    const cli = (version.cli_options && typeof version.cli_options === 'object') ? version.cli_options : {};
    for (const key of CLI_FIELDS) {
        if (key in cli && byId(key)) byId(key).value = String(cli[key]);
    }
    const configPayload = (version.config_payload && typeof version.config_payload === 'object') ? version.config_payload : {};
    applyDefaultToCliFields(configPayload);
    applyConfigPayloadToParamRows(configPayload);
    renderParamTable();
    applyTrainDataVersionSelection();
}

function getSelectedTrainDataVersion() {
    return String(byId('train_data_version')?.value || '').trim();
}

function getSelectedTrainDataVersionMeta() {
    const vid = getSelectedTrainDataVersion();
    if (!vid) return null;
    return (state.dataVersions || []).find((x) => x.version_id === vid) || null;
}

function getTrainContextMeta() {
    const v = getSelectedTrainDataVersionMeta();
    if (v && v.task && v.model && v.dataset) {
        return {task: String(v.task), model: String(v.model), dataset: String(v.dataset), version: v};
    }
    const task = String(byId('data_task')?.value || '').trim();
    const model = String(byId('data_model')?.value || '').trim();
    const dataset = String(byId('data_dataset')?.value || '').trim();
    if (task && model && dataset) {
        return {task, model, dataset, version: null};
    }
    return null;
}

function applyTrainDataVersionSelection() {
    const vid = getSelectedTrainDataVersion();
    const v = (state.dataVersions || []).find((x) => x.version_id === vid);
    const hintEl = byId('train_data_version_hint');
    const contextEl = byId('train_data_context');
    const lockIds = ['seed', 'train_rate', 'eval_rate', 'dataset_class', 'config_file'];
    if (!v) {
        if (hintEl) hintEl.innerText = '';
        if (contextEl) contextEl.innerText = '-';
        lockIds.forEach((id) => {
            const el = byId(id);
            if (el) el.disabled = false;
        });
        return;
    }
    if (hintEl) {
        hintEl.innerText = '已锁定数据相关参数（seed/train_rate/eval_rate/dataset_class/config_file 等）为所选 data_version 配置，训练页修改不会触发重新生成。';
    }
    if (contextEl) {
        const p = getDataVersionProfile(v);
        const seedText = p.seed === null ? '-' : String(p.seed);
        const splitText = `${formatRate(p.trainRate)}/${formatRate(p.evalRate)}/${formatRate(p.testRate)}`;
        contextEl.innerText = `${v.task || '-'} / ${v.model || '-'} / ${v.dataset || '-'} | seed=${seedText} | split(train/val/test)=${splitText}`;
    }
    lockIds.forEach((id) => {
        const el = byId(id);
        if (el) el.disabled = true;
    });
}

function renderDataVersionOptions() {
    const versions = Array.isArray(state.dataVersions) ? state.dataVersions : [];
    const datasetFilter = String(state.dataVersionsDatasetFilter || '').trim();
    const versionsForDataTab = datasetFilter
        ? versions.filter((x) => String(x.dataset || '').trim() === datasetFilter)
        : versions;
    const ready = versions.filter((x) => String(x.status || '').toLowerCase() === 'ready');
    const render = (id, list, selected) => {
        const el = byId(id);
        if (!el) return;
        const opts = ['<option value=""></option>'].concat(
            list.map((v) => `<option value="${esc(v.version_id)}" ${selected === v.version_id ? 'selected' : ''}>${esc(formatDataVersionTag(v))}</option>`)
        );
        el.innerHTML = opts.join('');
    };
    render('data_version_select', versionsForDataTab, byId('data_version_select')?.value || '');
    render('train_data_version', ready, byId('train_data_version')?.value || '');
    const runId = byId('resume_run_id')?.value || '';
    const run = (state.resumeRuns || []).find((x) => x.run_id === runId);
    const resumeReady = run ? ready.filter((v) => v.task === run.task && v.model === run.model && v.dataset === run.dataset) : ready;
    render('resume_data_version', resumeReady, byId('resume_data_version')?.value || '');
    if (!byId('train_data_version')?.value && ready.length) byId('train_data_version').value = ready[0].version_id;
    if (!byId('resume_data_version')?.value && resumeReady.length) byId('resume_data_version').value = resumeReady[0].version_id;
    if (!byId('data_version_select')?.value && versionsForDataTab.length) byId('data_version_select').value = versionsForDataTab[0].version_id;
    const selected = (versionsForDataTab || []).find((x) => x.version_id === byId('data_version_select')?.value);
    renderDataVersionHint(selected || null);
    applyTrainDataVersionSelection();
}

async function loadDataVersions() {
    const r = await fetch('/api/data/versions');
    const data = await r.json();
    state.dataVersions = Array.isArray(data.versions) ? data.versions : [];
    renderDataVersionOptions();
}

async function saveDataVersionNote() {
    const versionId = String(byId('data_version_select')?.value || '').trim();
    if (!versionId) {
        alert('请先选择数据版本');
        return;
    }
    const note = String(byId('data_version_note')?.value || '').trim();
    const r = await fetch('/api/data/version/note', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({version_id: versionId, note}),
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || '保存备注失败');
        return;
    }
    await loadDataVersions();
}

async function renameDataVersion() {
    const versionId = String(byId('data_version_select')?.value || '').trim();
    if (!versionId) {
        alert('请先选择数据版本');
        return;
    }
    const next = window.prompt('输入新的版本ID', versionId);
    if (next === null) return;
    const newVersionId = String(next).trim();
    if (!newVersionId || newVersionId === versionId) return;
    const r = await fetch('/api/data/version/rename', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({version_id: versionId, new_version_id: newVersionId}),
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || '重命名失败');
        return;
    }
    await loadDataVersions();
}

async function deleteDataVersion() {
    const versionId = String(byId('data_version_select')?.value || '').trim();
    if (!versionId) {
        alert('请先选择数据版本');
        return;
    }
    const ok = window.confirm(`确认删除数据版本 ${versionId} ?`);
    if (!ok) return;
    const r = await fetch('/api/data/version/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({version_id: versionId}),
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || '删除失败');
        return;
    }
    await loadDataVersions();
}

async function refreshDataStatus() {
    const r = await fetch('/api/data/status');
    const s = await r.json();
    state.dataStatus = s || {};
    const st = s.running ? 'running' : (s.return_code === 0 ? 'finished' : 'idle');
    setDataStatus(st, s.error || '');
    renderLogsInto(s.logs_tail || [], 'data_logs');
}

async function startDataPrep() {
    let dataCliOptions;
    try {
        dataCliOptions = collectDataCliOptions();
    } catch (e) {
        alert(`划分比例错误: ${e.message || e}`);
        return;
    }
    const body = {
        task: byId('data_task').value,
        model: byId('data_model').value,
        dataset: byId('data_dataset').value,
        extra_args: byId('data_extra_args').value || '',
        cli_options: dataCliOptions,
        config: {},
    };
    const r = await fetch('/api/data/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || '数据处理启动失败');
        return;
    }
    await loadDataVersions();
}

async function stopDataPrep() {
    const r = await fetch('/api/data/stop', {method: 'POST'});
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || '停止数据处理失败');
    }
}

async function loadDefaults() {
    applyTrainDataVersionSelection();
    const ctx = getTrainContextMeta();
    if (!ctx) return;
    const body = {task: ctx.task, model: ctx.model, dataset: ctx.dataset};
    const r = await fetch('/api/default_config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || t('error_load_defaults_failed', '加载默认参数失败'));
        return;
    }
    const cfg = data.config || {};
    cfg.max_epoch = 10;
    const executorKeys = new Set(data.executor_keys || []);
    const allRows = Object.keys(cfg).filter((k) => !HIDDEN_TRAIN_PARAM_KEYS.has(k)).sort().map(k => ({
        key: k,
        defaultValue: cfg[k],
        type: inferType(cfg[k]),
        value: toInputString(cfg[k])
    }));
    state.paramRowsExecutor = allRows.filter((row) => executorKeys.has(row.key));
    state.paramRowsConfig = allRows.filter((row) => !executorKeys.has(row.key));
    applyDefaultToCliFields(cfg);
    renderParamTable();
}

async function startTrain() {
    const dataVersionId = getSelectedTrainDataVersion();
    if (!dataVersionId) {
        alert('请先在训练页选择 ready 的 data_version_id');
        return;
    }
    const version = getSelectedTrainDataVersionMeta();
    if (!version || !version.task || !version.model || !version.dataset) {
        alert('所选 data_version 缺少 task/model/dataset 信息');
        return;
    }
    applyTrainDataVersionSelection();
    let config;
    try {
        config = collectConfigFromTable();
    } catch (e) {
        alert(`${t('error_param_parse_failed', '参数解析失败')}: ${e.message}`);
        return;
    }
    const cliOptions = collectCliOptions();

    const body = {
        task: version.task,
        model: version.model,
        dataset: version.dataset,
        data_version_id: dataVersionId,
        saved_model: byId('saved_model').value,
        train: byId('train').value,
        extra_args: byId('extra_args').value || '',
        cli_options: cliOptions,
        config: config,
    };
    const r = await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    });
    const data = await r.json();
    if (!r.ok) alert(data.error || t('error_start_failed', '启动失败'));
}

function formatResumeRunTag(run) {
    const latest = Number(run?.latest_epoch);
    const latestText = Number.isFinite(latest)
        ? tf('resume_latest_epoch', {epoch: latest}, `latest=${latest}`)
        : t('resume_latest_epoch_na', 'latest=-');
    return `${run.run_id} | ${run.model || '-'} | ${run.dataset || '-'} | ${latestText}`;
}

function renderResumeHint(run) {
    const el = byId('resume_hint');
    if (!el) return;
    if (!run) {
        el.innerText = t('resume_hint_empty', '请选择一个可续训的运行目录');
        return;
    }
    const epochs = Array.isArray(run.epochs) ? run.epochs : [];
    const latest = Number(run.latest_epoch);
    const minEpoch = epochs.length ? epochs[0] : '-';
    const maxEpoch = Number.isFinite(latest) ? latest : '-';
    el.innerText = tf(
        'resume_hint_selected',
        {run: run.run_id, model: run.model || '-', dataset: run.dataset || '-', min: minEpoch, max: maxEpoch},
        `run=${run.run_id}, model=${run.model || '-'}, dataset=${run.dataset || '-'}, checkpoints=${minEpoch}..${maxEpoch}`
    );
}

function renderResumeRunOptions() {
    const el = byId('resume_run_id');
    if (!el) return;
    const selected = el.value;
    const list = state.resumeRuns || [];
    const options = ['<option value=""></option>'].concat(
        list.map((run) => {
            const sel = selected && selected === run.run_id ? 'selected' : '';
            return `<option value="${esc(run.run_id)}" ${sel}>${esc(formatResumeRunTag(run))}</option>`;
        })
    );
    el.innerHTML = options.join('');
    if (!el.value && list.length) {
        el.value = list[0].run_id;
    }
    onResumeRunChanged();
}

function renderResumeEpochOptions(run) {
    const epochEl = byId('resume_epoch');
    if (!epochEl) return;
    const epochsRaw = Array.isArray(run?.epochs) ? run.epochs : [];
    const epochs = epochsRaw
        .map((x) => Number(x))
        .filter((x) => Number.isFinite(x) && x >= 0)
        .map((x) => Math.floor(x))
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
        .map((epoch) => `<option value="${epoch}" ${epoch === selected ? 'selected' : ''}>${epoch}</option>`)
        .join('');
    epochEl.value = String(selected);
    epochEl.disabled = false;
}

function onResumeRunChanged() {
    const runId = byId('resume_run_id')?.value || '';
    const run = (state.resumeRuns || []).find((x) => x.run_id === runId);
    renderResumeHint(run || null);
    renderResumeEpochOptions(run || null);
    if (!run) return;
    const resumeEpoch = Math.floor(Number(byId('resume_epoch').value));
    if (!Number.isFinite(resumeEpoch) || resumeEpoch < 0) return;
    const currentMax = Math.max(1, Math.floor(Number(byId('resume_max_epoch').value || (resumeEpoch + 10))));
    byId('resume_max_epoch').value = String(Math.max(currentMax, resumeEpoch + 1));
    renderDataVersionOptions();
}

async function loadResumeRuns() {
    const r = await fetch('/api/resume_runs');
    const data = await r.json();
    state.resumeRuns = Array.isArray(data.runs) ? data.runs : [];
    renderResumeRunOptions();
}

async function startResumeTrain() {
    const runId = (byId('resume_run_id').value || '').trim();
    if (!runId) {
        alert(t('resume_select_first', '请先选择可续训模型'));
        return;
    }
    const run = (state.resumeRuns || []).find((x) => x.run_id === runId);
    if (!run) {
        alert(t('resume_run_not_found', '未找到对应运行目录'));
        return;
    }
    if (!run.task || !run.model || !run.dataset) {
        alert(t('resume_run_meta_missing', '所选运行目录缺少 task/model/dataset 信息，无法继续训练'));
        return;
    }
    const dataVersionId = String(byId('resume_data_version')?.value || '').trim();
    if (!dataVersionId) {
        alert('请先选择可用的 data_version_id');
        return;
    }
    const resumeEpoch = Math.floor(Number(byId('resume_epoch').value));
    if (!Number.isFinite(resumeEpoch) || resumeEpoch < 0) {
        alert(t('resume_invalid_epoch', '请选择有效的续训 epoch'));
        return;
    }
    const availableEpochs = (Array.isArray(run.epochs) ? run.epochs : []).map((x) => Math.floor(Number(x)));
    if (!availableEpochs.includes(resumeEpoch)) {
        alert(t('resume_invalid_epoch', '请选择有效的续训 epoch'));
        return;
    }
    const targetMaxEpoch = Math.max(1, Math.floor(Number(byId('resume_max_epoch').value || (resumeEpoch + 1))));
    if (targetMaxEpoch <= resumeEpoch) {
        alert(t('resume_invalid_max_epoch', '目标 max_epoch 必须大于续训起始 epoch'));
        return;
    }
    byId('resume_epoch').value = String(resumeEpoch);
    byId('resume_max_epoch').value = String(targetMaxEpoch);
    const expId = String(run.run_id || '').split('__')[0];
    const body = {
        task: run.task,
        model: run.model,
        dataset: run.dataset,
        data_version_id: dataVersionId,
        saved_model: byId('resume_saved_model').value,
        train: true,
        extra_args: byId('resume_extra_args').value || '',
        cli_options: {exp_id: expId},
        config: {
            epoch: resumeEpoch,
            max_epoch: targetMaxEpoch,
        },
    };
    const r = await fetch('/api/start_resume', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || t('resume_start_failed', '继续训练启动失败'));
        return;
    }
}

async function stopTrain() {
    const r = await fetch('/api/stop', {method: 'POST'});
    const data = await r.json();
    if (!r.ok) alert(data.error || t('error_stop_failed', '停止失败'));
}

async function clearState() {
    const r = await fetch('/api/clear', {method: 'POST'});
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || t('error_clear_failed', '清空状态失败'));
        return;
    }
    state.activeRunKey = '';
    state.lossRenderedEpochCount = -1;
    state.resultRenderedRunKey = '';
    await refreshStatus();
}

const charts = {
    model_param_chart: null,
    loss_chart: null,
    chart: null,
    compare_chart: null,
};

function resizeAllCharts() {
    Object.values(charts).forEach((chart) => {
        if (chart) chart.resize();
    });
}

function renderEChart(containerId, option) {
    const dom = byId(containerId);
    if (!dom || typeof echarts === 'undefined') return;
    let chart = charts[containerId];
    if (!chart) {
        chart = echarts.init(dom);
        charts[containerId] = chart;
    }
    const reviveFn = (node) => {
        if (!node || typeof node !== 'object') return;
        if (Array.isArray(node)) {
            node.forEach((item) => reviveFn(item));
            return;
        }
        for (const [k, v] of Object.entries(node)) {
            if ((k === 'formatter' || k === 'valueFormatter') && typeof v === 'string') {
                const s = v.trim();
                if (s.startsWith('function(') || s.startsWith('(function(')) {
                    try {
                        node[k] = new Function(`return (${s});`)();
                    } catch (_) {
                        // Keep original string if conversion fails.
                    }
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
        console.warn(`Failed to render chart: ${containerId}`, err);
    }
}

const PANE_RATIO_CONTROLS = {
    pane_top: {inputId: 'layout_ratio_top', textId: 'layout_ratio_top_text'},
    pane_bottom: {inputId: 'layout_ratio_bottom', textId: 'layout_ratio_bottom_text'},
    pane_compare: {inputId: 'layout_ratio_compare', textId: 'layout_ratio_compare_text'},
};

const CHART_BASE_HEIGHTS = {
    chart: 330,
    compare_chart: 360,
    model_param_chart: 300,
    loss_chart: 390,
};

const TILE_OFFSET_LIMIT = 1400;

function clampPaneRatioPercent(v, fallback = 50) {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(20, Math.min(80, Math.floor(n)));
}

function applyPaneGridTemplate(pane, leftPercent) {
    if (!pane) return;
    if (window.matchMedia('(max-width: 1050px)').matches) {
        pane.style.gridTemplateColumns = '1fr';
        return;
    }
    pane.style.gridTemplateColumns = `${leftPercent}% 8px ${100 - leftPercent}%`;
}

function setPaneRatioByPercent(paneId, leftPercent, persist = true, resize = true) {
    const pane = byId(paneId);
    if (!pane) return;
    const left = clampPaneRatioPercent(leftPercent);
    applyPaneGridTemplate(pane, left);
    if (persist) {
        window.localStorage.setItem(`pane_ratio_${paneId}`, String(left / 100));
    }
    const cfg = PANE_RATIO_CONTROLS[paneId];
    if (cfg) {
        const input = byId(cfg.inputId);
        const text = byId(cfg.textId);
        if (input) input.value = String(left);
        if (text) text.innerText = `${left}% / ${100 - left}%`;
    }
    if (resize) resizeAllCharts();
}

function applyChartScale(scalePercent, persist = true) {
    const scale = Math.max(70, Math.min(170, Math.floor(Number(scalePercent) || 100)));
    state.chartScale = scale;
    const root = document.documentElement;
    root.style.setProperty('--chart-height-pred', `${Math.round(CHART_BASE_HEIGHTS.chart * scale / 100)}px`);
    root.style.setProperty('--chart-height-compare', `${Math.round(CHART_BASE_HEIGHTS.compare_chart * scale / 100)}px`);
    root.style.setProperty('--chart-height-model', `${Math.round(CHART_BASE_HEIGHTS.model_param_chart * scale / 100)}px`);
    root.style.setProperty('--chart-height-loss', `${Math.round(CHART_BASE_HEIGHTS.loss_chart * scale / 100)}px`);
    const slider = byId('layout_chart_scale');
    const text = byId('layout_chart_scale_text');
    if (slider) slider.value = String(scale);
    if (text) text.innerText = `${scale}%`;
    if (persist) window.localStorage.setItem('train_web_chart_scale', String(scale));
    resizeAllCharts();
}

function resetLayoutControls() {
    setPaneRatioByPercent('pane_top', 50, true, false);
    setPaneRatioByPercent('pane_bottom', 50, true, false);
    setPaneRatioByPercent('pane_compare', 50, true, false);
    applyChartScale(100, true);
}

function initLayoutControls() {
    Object.entries(PANE_RATIO_CONTROLS).forEach(([paneId, cfg]) => {
        const input = byId(cfg.inputId);
        if (!input) return;
        const raw = Number(window.localStorage.getItem(`pane_ratio_${paneId}`));
        const left = Number.isFinite(raw) ? clampPaneRatioPercent(raw * 100) : 50;
        setPaneRatioByPercent(paneId, left, false, false);
        input.addEventListener('input', (e) => setPaneRatioByPercent(paneId, e.target.value, true));
    });
    applyChartScale(state.chartScale, false);
    const scaleInput = byId('layout_chart_scale');
    if (scaleInput) {
        scaleInput.addEventListener('input', (e) => applyChartScale(e.target.value, true));
    }
}

function refreshPaneLayoutForViewport() {
    Object.keys(PANE_RATIO_CONTROLS).forEach((paneId) => {
        const raw = Number(window.localStorage.getItem(`pane_ratio_${paneId}`));
        const left = Number.isFinite(raw) ? clampPaneRatioPercent(raw * 100) : 50;
        setPaneRatioByPercent(paneId, left, false, false);
    });
}

function setLayoutPopoverOpen(open) {
    const panel = byId('layout_popup');
    const btn = byId('layout_fab');
    if (!panel || !btn) return;
    const isOpen = !!open;
    panel.classList.toggle('open', isOpen);
    panel.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
    btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}

function initLayoutPopover() {
    const panel = byId('layout_popup');
    const btn = byId('layout_fab');
    if (!panel || !btn) return;
    btn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        setLayoutPopoverOpen(!panel.classList.contains('open'));
    });
    panel.addEventListener('pointerdown', (ev) => ev.stopPropagation());
    window.addEventListener('pointerdown', (ev) => {
        if (!panel.classList.contains('open')) return;
        if (panel.contains(ev.target) || btn.contains(ev.target)) return;
        setLayoutPopoverOpen(false);
    });
    window.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape') setLayoutPopoverOpen(false);
    });
}

function clampTileOffset(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 0;
    return Math.max(-TILE_OFFSET_LIMIT, Math.min(TILE_OFFSET_LIMIT, Math.round(n)));
}

function getTileStorageKey(tileKey) {
    return `train_web_tile_offset_${tileKey}`;
}

function readTileOffset(tileKey) {
    const raw = window.localStorage.getItem(getTileStorageKey(tileKey));
    if (!raw) return {x: 0, y: 0};
    const [xRaw, yRaw] = raw.split(',');
    return {x: clampTileOffset(xRaw), y: clampTileOffset(yRaw)};
}

function applyTileOffset(card, x, y, persist = true) {
    if (!card) return;
    const tileKey = card.dataset.tileKey;
    if (!tileKey) return;
    const safeX = clampTileOffset(x);
    const safeY = clampTileOffset(y);
    card.style.left = `${safeX}px`;
    card.style.top = `${safeY}px`;
    card.dataset.tileOffsetX = String(safeX);
    card.dataset.tileOffsetY = String(safeY);
    if (persist) {
        window.localStorage.setItem(getTileStorageKey(tileKey), `${safeX},${safeY}`);
    }
}

function resetTilePositions() {
    document.querySelectorAll('.resizable-card').forEach((card, idx) => {
        const tileKey = card.dataset.tileKey || card.id || `resizable_tile_${idx + 1}`;
        card.dataset.tileKey = tileKey;
        applyTileOffset(card, 0, 0, true);
    });
    resizeAllCharts();
}

function initMovableTiles() {
    document.querySelectorAll('.resizable-card').forEach((card, idx) => {
        const tileKey = card.id || `resizable_tile_${idx + 1}`;
        card.dataset.tileKey = tileKey;
        card.classList.add('movable-tile');
        const saved = readTileOffset(tileKey);
        applyTileOffset(card, saved.x, saved.y, false);

        const toolbar = card.querySelector('.toolbar');
        if (!toolbar) return;
        toolbar.addEventListener('pointerdown', (ev) => {
            if (ev.button !== 0) return;
            if (ev.target.closest('.toolbar-controls')) return;
            if (ev.target.closest('button,input,select,textarea,a,label')) return;
            ev.preventDefault();

            const startX = ev.clientX;
            const startY = ev.clientY;
            const baseX = Number(card.dataset.tileOffsetX || 0);
            const baseY = Number(card.dataset.tileOffsetY || 0);
            card.classList.add('tile-moving');
            document.body.style.userSelect = 'none';

            const onPointerMove = (moveEv) => {
                const nextX = baseX + (moveEv.clientX - startX);
                const nextY = baseY + (moveEv.clientY - startY);
                applyTileOffset(card, nextX, nextY, false);
            };

            const onPointerUp = () => {
                document.body.style.userSelect = '';
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', onPointerUp);
                card.classList.remove('tile-moving');
                applyTileOffset(card, card.dataset.tileOffsetX, card.dataset.tileOffsetY, true);
            };

            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
        });
    });
}

function initPaneResizers() {
    document.querySelectorAll('.pane').forEach((pane) => {
        const left = pane.querySelector('.pane-left');
        const right = pane.querySelector('.pane-right');
        const handle = pane.querySelector('.pane-resizer');
        if (!left || !right || !handle) return;

        const storageKey = pane.id ? `pane_ratio_${pane.id}` : '';
        if (storageKey) {
            const ratioRaw = window.localStorage.getItem(storageKey);
            const ratio = Number(ratioRaw);
            if (Number.isFinite(ratio) && ratio > 0.15 && ratio < 0.85) {
                if (pane.id) setPaneRatioByPercent(pane.id, ratio * 100, false, false);
            }
        }

        handle.addEventListener('pointerdown', (ev) => {
            if (window.matchMedia('(max-width: 1050px)').matches) return;
            ev.preventDefault();
            handle.setPointerCapture(ev.pointerId);
            document.body.style.userSelect = 'none';
            const paneRect = pane.getBoundingClientRect();
            const startX = ev.clientX;
            const startLeft = left.getBoundingClientRect().width;
            const minWidth = 280;

            const onPointerMove = (moveEv) => {
                const delta = moveEv.clientX - startX;
                const total = paneRect.width - 8;
                let nextLeft = startLeft + delta;
                nextLeft = Math.max(minWidth, Math.min(total - minWidth, nextLeft));
                const ratio = nextLeft / Math.max(1, total);
                applyPaneGridTemplate(pane, ratio * 100);
                resizeAllCharts();
            };

            const onPointerUp = () => {
                document.body.style.userSelect = '';
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', onPointerUp);
                const totalWidth = left.getBoundingClientRect().width + right.getBoundingClientRect().width;
                const ratio = left.getBoundingClientRect().width / Math.max(1, totalWidth);
                if (storageKey && Number.isFinite(ratio)) {
                    window.localStorage.setItem(storageKey, String(ratio));
                    if (pane.id) setPaneRatioByPercent(pane.id, ratio * 100, false);
                }
            };

            window.addEventListener('pointermove', onPointerMove);
            window.addEventListener('pointerup', onPointerUp);
        });
    });
}

function renderMetrics(result) {
    const summary = result.metrics_summary || {};
    const rows = result.metrics_rows || [];
    const columns = Array.isArray(result.metrics_columns) ? result.metrics_columns : [];
    const summaryOrder = columns.filter((c) => c !== 'horizon' && summary[c] !== undefined);
    const fallbackSummaryOrder = Object.keys(summary).filter((k) => !summaryOrder.includes(k));
    const summaryKeys = summaryOrder.concat(fallbackSummaryOrder);
    byId('summary').innerText = summaryKeys
        .map(k => `${k}: h1=${summary[k].h1.toFixed(4)}, avg=${summary[k].avg.toFixed(4)}, best=${summary[k].best.toFixed(4)}`)
        .join(' | ');
    byId('metrics').innerHTML = result.metrics_table_html || (rows.length ? '' : `<div class="small">${esc(t('table_no_data', '无数据'))}</div>`);
}

function drawChart(option) {
    renderEChart('chart', option || {});
}

function getPredictionSelectors() {
    return {
        horizon: byId('pred_horizon'),
        node: byId('pred_node'),
        feature: byId('pred_feature'),
    };
}

function setPredictionSelectOptions(el, count, selected) {
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
    for (let i = 1; i <= upper; i += 1) {
        opts.push(`<option value="${i}" ${i === safe ? 'selected' : ''}>${i}</option>`);
    }
    el.innerHTML = opts.join('');
    el.value = String(safe);
    el.disabled = false;
}

function resetPredictionSelectors() {
    const selectors = getPredictionSelectors();
    setPredictionSelectOptions(selectors.horizon, 0, 1);
    setPredictionSelectOptions(selectors.node, 0, 1);
    setPredictionSelectOptions(selectors.feature, 0, 1);
    state.predictionRanges = {horizon: 0, node: 0, feature: 0};
    state.predictionSelection = {horizon: 1, node: 1, feature: 1};
}

function applyPredictionSelectorMeta(result) {
    const selectors = getPredictionSelectors();
    const shape = Array.isArray(result?.shapes?.prediction) ? result.shapes.prediction : [];
    const ranges = result?.prediction_selector?.ranges || {};
    const selection = result?.prediction_selector?.selection || {};
    const toPositiveInt = (v) => {
        const n = Number(v);
        return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
    };
    const horizonMax = toPositiveInt(ranges.horizon || shape[1] || 0);
    const nodeMax = toPositiveInt(ranges.node || shape[2] || 0);
    const featureMax = toPositiveInt(ranges.feature || shape[3] || 0);
    setPredictionSelectOptions(selectors.horizon, horizonMax, selection.horizon || state.predictionSelection.horizon);
    setPredictionSelectOptions(selectors.node, nodeMax, selection.node || state.predictionSelection.node);
    setPredictionSelectOptions(selectors.feature, featureMax, selection.feature || state.predictionSelection.feature);
    state.predictionRanges = {horizon: horizonMax, node: nodeMax, feature: featureMax};
    state.predictionSelection = {
        horizon: Number(selectors.horizon?.value || 1),
        node: Number(selectors.node?.value || 1),
        feature: Number(selectors.feature?.value || 1),
    };
}

async function refreshPredictionSeries() {
    if ((state.predictionRanges.horizon || 0) <= 0 || (state.predictionRanges.node || 0) <= 0 || (state.predictionRanges.feature || 0) <= 0) {
        return;
    }
    const selectors = getPredictionSelectors();
    const horizon = Number(selectors.horizon?.value || 1);
    const node = Number(selectors.node?.value || 1);
    const feature = Number(selectors.feature?.value || 1);
    state.predictionSelection = {horizon, node, feature};
    const qs = new URLSearchParams({
        horizon: String(horizon),
        node: String(node),
        feature: String(feature),
    });
    try {
        const rr = await fetch(`/api/result_series?${qs.toString()}`);
        if (!rr.ok) return;
        const data = await rr.json();
        drawChart(data.chart_option || {});
    } catch (_) {
        // Ignore transient network errors and keep current chart.
    }
}

function drawModelParamChart(plot, pieOption, barOption) {
    const total = plot?.total_params;
    byId('model_param_summary').innerText = total
        ? tf('summary_total_params', {total: total.toLocaleString()}, `总参数量: ${total.toLocaleString()} | 展示: Top 参数分布`)
        : t('summary_no_param_data', '暂无参数分布数据');
    const useBar = state.modelPlotType === 'bar';
    let option = useBar ? (barOption || {}) : (pieOption || {});
    if (!useBar) {
        const hasSeries = Array.isArray(option?.series) && option.series.length > 0;
        if (!hasSeries) {
            const labels = Array.isArray(plot?.labels) ? plot.labels : [];
            const counts = Array.isArray(plot?.counts) ? plot.counts : [];
            const pairs = labels.map((name, i) => ({name, value: Number(counts[i] || 0)}))
                .filter((x) => Number.isFinite(x.value) && x.value > 0)
                .sort((a, b) => b.value - a.value);
            const k = Math.max(1, state.modelPlotTopKPie || 8);
            const top = pairs.slice(0, k);
            const rest = pairs.slice(k).reduce((s, x) => s + x.value, 0);
            if (rest > 0) top.push({name: t('chart_others', 'Others'), value: rest});
            const fallbackData = top.length ? top : [{name: t('table_no_data', '暂无数据'), value: 1}];
            option = {
                tooltip: {trigger: 'item', formatter: '{b}: {c} ({d}%)'},
                legend: {type: 'scroll', orient: 'vertical', left: '68%', top: '12%'},
                series: [{
                    type: 'pie',
                    radius: ['35%', '65%'],
                    center: ['40%', '55%'],
                    data: fallbackData,
                    label: {formatter: '{b}: {d}%'}
                }]
            };
        }
    }
    renderEChart('model_param_chart', option);
}

function updateLossSummary(plot) {
    const xs = plot?.epochs || [];
    const maxEpoch = plot?.max_epoch;
    const maxText = maxEpoch ? ` / ${maxEpoch}` : '';
    byId('loss_summary').innerText = xs.length
        ? tf('summary_loss_recorded', {count: xs.length, max: maxText}, `已记录 ${xs.length} 个 epoch${maxText}`)
        : t('summary_loss_empty', '暂无 loss 数据');
}

function drawLossChart(option, plot) {
    updateLossSummary(plot);
    renderEChart('loss_chart', option || {});
}

function clearResultPanels() {
    byId('summary').innerText = t('summary_wait_train_done', '等待训练全部完成后展示');
    byId('metrics').innerHTML = `<div class="small">${esc(t('summary_wait_train_done', '等待训练全部完成后展示'))}</div>`;
    const chart = charts.chart;
    if (chart) chart.clear();
    resetPredictionSelectors();
}

function switchTab(tabName) {
    state.activeTab = ['data', 'train', 'resume', 'compare', 'history'].includes(tabName) ? tabName : 'train';
    byId('tab_btn_data').classList.toggle('active', state.activeTab === 'data');
    byId('tab_btn_train').classList.toggle('active', state.activeTab === 'train');
    byId('tab_btn_resume').classList.toggle('active', state.activeTab === 'resume');
    byId('tab_btn_compare').classList.toggle('active', state.activeTab === 'compare');
    byId('tab_btn_history').classList.toggle('active', state.activeTab === 'history');
    byId('tab_data').classList.toggle('active', state.activeTab === 'data');
    byId('tab_train').classList.toggle('active', state.activeTab === 'train');
    byId('tab_resume').classList.toggle('active', state.activeTab === 'resume');
    byId('tab_compare').classList.toggle('active', state.activeTab === 'compare');
    byId('tab_history').classList.toggle('active', state.activeTab === 'history');
    setTimeout(resizeAllCharts, 60);
}

function renderCompareRunOptions() {
    const list = state.compareRuns || [];
    const render = (id, selected) => {
        const el = byId(id);
        const opts = ['<option value=""></option>'].concat(
            list.map((r) => {
                const tag = `${r.run_id} | ${r.model || '-'} | ${r.dataset || '-'}`;
                const sel = selected && selected === r.run_id ? 'selected' : '';
                return `<option value="${esc(r.run_id)}" ${sel}>${esc(tag)}</option>`;
            })
        );
        el.innerHTML = opts.join('');
    };
    const a = byId('compare_run_a').value;
    const b = byId('compare_run_b').value;
    const c = byId('compare_run_c').value;
    render('compare_run_a', a);
    render('compare_run_b', b);
    render('compare_run_c', c);
    if (!byId('compare_run_a').value && list.length) byId('compare_run_a').value = list[0].run_id;
    if (!byId('compare_run_b').value && list.length > 1) byId('compare_run_b').value = list[1].run_id;
}

async function loadCompareRuns() {
    const r = await fetch('/api/runs');
    const data = await r.json();
    state.compareRuns = Array.isArray(data.runs) ? data.runs : [];
    renderCompareRunOptions();
}

function renderCompareMetrics(items) {
    const table = byId('compare_metrics');
    if (!items || !items.length) {
        table.innerHTML = `<tr><td>${esc(t('compare_no_data', '无可对比数据'))}</td></tr>`;
        return;
    }
    const metrics = new Set();
    items.forEach((it) => Object.keys(it.metrics_summary || {}).forEach((k) => metrics.add(k)));
    const metricList = Array.from(metrics);
    let html = `<tr><th>${esc(t('compare_run_header', 'run'))}</th>`;
    metricList.forEach((m) => {
        html += `<th>${esc(m)}(${esc(t('compare_h1', 'h1'))})</th><th>${esc(m)}(${esc(t('compare_avg', 'avg'))})</th><th>${esc(m)}(${esc(t('compare_best', 'best'))})</th>`;
    });
    html += '</tr>';
    items.forEach((it) => {
        html += `<tr><td class="mono">${esc(it.run_id)}</td>`;
        metricList.forEach((m) => {
            const x = it.metrics_summary?.[m];
            if (!x) html += '<td>-</td><td>-</td><td>-</td>';
            else html += `<td>${Number(x.h1).toFixed(4)}</td><td>${Number(x.avg).toFixed(4)}</td><td>${Number(x.best).toFixed(4)}</td>`;
        });
        html += '</tr>';
    });
    table.innerHTML = html;
}

async function loadComparison() {
    const runIds = [byId('compare_run_a').value, byId('compare_run_b').value, byId('compare_run_c').value]
        .filter((x) => !!x);
    if (!runIds.length) {
        byId('compare_metrics').innerHTML = `<tr><td>${esc(t('compare_select_model_first', '请先选择模型'))}</td></tr>`;
        return;
    }
    const r = await fetch('/api/compare', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({run_ids: runIds}),
    });
    const data = await r.json();
    if (!r.ok) {
        alert(data.error || t('compare_failed', '对比失败'));
        return;
    }
    renderCompareMetrics(data.items || []);
    renderEChart('compare_chart', data.chart_option || {});
}

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
    const table = byId('history_table');
    if (!Array.isArray(items) || !items.length) {
        table.innerHTML = `<tr><td>${esc(t('history_empty', '暂无历史记录'))}</td></tr>`;
        return;
    }
    let html = `<tr>
        <th>${esc(t('history_col_status', '状态'))}</th>
        <th>${esc(t('history_col_run', 'Run'))}</th>
        <th>${esc(t('history_col_model', '模型'))}</th>
        <th>${esc(t('history_col_dataset', '数据集'))}</th>
        <th>${esc(t('history_col_duration', '耗时'))}</th>
        <th>${esc(t('history_col_metrics', '主要指标'))}</th>
        <th>${esc(t('history_col_output', '输出目录'))}</th>
        <th>${esc(t('history_col_end_time', '结束时间'))}</th>
    </tr>`;
    for (const it of items) {
        const mm = it.major_metrics || {};
        const metricText = Object.keys(mm).map((k) => `${k}=${Number(mm[k]).toFixed(4)}`).join(', ') || '-';
        html += `<tr>
            <td>${esc(t(`status_${it.status || 'finished'}`, it.status || 'finished'))}</td>
            <td class="mono">${esc(it.run_id || '-')}</td>
            <td>${esc(it.model || '-')}</td>
            <td>${esc(it.dataset || '-')}</td>
            <td>${esc(formatDuration(it.duration_sec))}</td>
            <td>${esc(metricText)}</td>
            <td class="mono history-path">${esc(it.output_dir || '-')}</td>
            <td>${esc(formatTime(it.ended_at))}</td>
        </tr>`;
    }
    table.innerHTML = html;
}

async function loadHistory() {
    const n = clampTopK(byId('history_limit').value, 20);
    byId('history_limit').value = String(n);
    const r = await fetch(`/api/history?limit=${encodeURIComponent(n)}`);
    const data = await r.json();
    renderHistoryTable(Array.isArray(data.items) ? data.items : []);
}

function updateAutoScrollBtnText() {
    const btn = byId('log_autoscroll_btn');
    if (!btn) return;
    btn.innerText = tf(
        'log_autoscroll',
        {state: state.logAutoScroll ? t('state_on', '开') : t('state_off', '关')},
        `自动滚动: ${state.logAutoScroll ? '开' : '关'}`
    );
}

function renderLogsInto(lines, containerId) {
    const logEl = byId(containerId);
    if (!logEl) return;
    const keyword = (state.logFilterKeyword || '').trim().toLowerCase();
    const levelFilter = (state.logFilterLevel || 'all').toLowerCase();
    const shouldStickBottom = state.logAutoScroll && (logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 28);
    const filtered = (lines || []).map((rawLine) => {
        const line = String(rawLine ?? '');
        const levelMatch = line.match(/\b(INFO|WARNING|ERROR|DEBUG)\b/);
        const level = levelMatch ? levelMatch[1] : '';
        const cls = level ? level.toLowerCase() : (line.trim().startsWith('$') ? 'cmd' : 'plain');
        return {line, cls, level: level ? level.toLowerCase() : cls};
    }).filter((x) => {
        if (levelFilter !== 'all' && x.level !== levelFilter) return false;
        if (keyword && !x.line.toLowerCase().includes(keyword)) return false;
        return true;
    });
    const html = filtered.map((x, idx) => `
                <div class="log-line">
                    <span class="log-index">${idx + 1}</span>
                    <span class="log-level log-level-${x.cls}">${x.cls === 'cmd' ? t('log_cmd', 'CMD') : (x.level === 'plain' ? t('log_log', 'LOG') : x.level.toUpperCase())}</span>
                    <span class="log-message">${esc(x.line)}</span>
                </div>
            `).join('');
    logEl.innerHTML = html || `<div class="log-empty">${esc(t('log_empty', '暂无日志'))}</div>`;
    if (shouldStickBottom) logEl.scrollTop = logEl.scrollHeight;
}

function renderLogs(lines) {
    renderLogsInto(lines, 'logs');
    renderLogsInto(lines, 'resume_logs');
}

function setStatus(stateName, extra) {
    const el = byId('status');
    el.className = 'badge';
    if (stateName === 'running') el.classList.add('status-running');
    else if (stateName === 'finished') el.classList.add('status-finished');
    else el.classList.add('status-idle');
    const localizedState = t(`status_${stateName}`, stateName);
    el.innerText = extra ? `${localizedState} (${extra})` : localizedState;
}

async function refreshStatus() {
    const r = await fetch('/api/status');
    const s = await r.json();
    const st = s.running ? 'running' : (s.return_code === 0 ? 'finished' : 'idle');
    const runKey = `${s.started_at || ''}__${s.exp_id || ''}__${s.ended_at || ''}__${s.return_code || ''}`;
    if (runKey !== state.activeRunKey) {
        state.activeRunKey = runKey;
        state.lossRenderedEpochCount = -1;
        state.resultRenderedRunKey = '';
    }
    setStatus(st, s.error ? tf('status_error', {error: s.error}, `error: ${s.error}`) : '');
    setResumeStatus(st, s.error ? tf('status_error', {error: s.error}, `error: ${s.error}`) : '');
    state.latestLogs = s.logs_tail || [];
    renderLogs(state.latestLogs);
    state.modelPlot = s.model_plot || {};
    state.lossPlot = s.loss_plot || {};
    state.modelPlotOptionPie = s.model_plot_option_pie || {};
    state.modelPlotOptionBar = s.model_plot_option_bar || {};
    drawModelParamChart(state.modelPlot, state.modelPlotOptionPie, state.modelPlotOptionBar);
    const epochsCount = Array.isArray(s.loss_plot?.epochs) ? s.loss_plot.epochs.length : 0;
    if (epochsCount !== state.lossRenderedEpochCount) {
        drawLossChart(s.loss_plot_option || {}, s.loss_plot || {});
        state.lossRenderedEpochCount = epochsCount;
    }
    if (!s.result_ready) clearResultPanels();
    if (s.result_ready && state.resultRenderedRunKey !== runKey) {
        const rr = await fetch('/api/result');
        if (rr.ok) {
            const result = await rr.json();
            renderMetrics(result);
            applyPredictionSelectorMeta(result);
            await refreshPredictionSeries();
            state.resultRenderedRunKey = runKey;
        }
    }
    await refreshDataStatus();
}

async function init() {
    byId('lang_select').value = normalizeLang(state.lang);
    byId('theme_select').value = state.theme;
    applyTheme(state.theme);
    byId('lang_select').addEventListener('change', async (e) => {
        const lang = e.target.value || 'zh-CN';
        await loadI18n(lang);
    });
    byId('theme_select').addEventListener('change', (e) => {
        applyTheme(e.target.value || 'light');
        setTimeout(resizeAllCharts, 60);
    });
    byId('tab_btn_data').addEventListener('click', async () => {
        switchTab('data');
        await loadDataVersions();
        await loadDataPreview();
    });
    byId('tab_btn_train').addEventListener('click', () => switchTab('train'));
    byId('tab_btn_resume').addEventListener('click', async () => {
        switchTab('resume');
        if (!state.resumeRuns.length) await loadResumeRuns();
    });
    byId('tab_btn_compare').addEventListener('click', async () => {
        switchTab('compare');
        if (!state.compareRuns.length) await loadCompareRuns();
    });
    byId('tab_btn_history').addEventListener('click', async () => {
        switchTab('history');
        await loadHistory();
    });
    await loadI18n(state.lang);
    const r = await fetch('/api/meta');
    const meta = await r.json();
    state.models = meta.models || [];
    state.datasets = meta.datasets || [];
    state.dataVersions = Array.isArray(meta.data_versions) ? meta.data_versions : [];
    renderTasks();
    renderDataModelsForTask();
    renderDatasets();
    initDataSplitControls();
    renderDataVersionOptions();
    byId('data_task').addEventListener('change', async () => {
        renderDataModelsForTask();
        await loadDefaults();
    });
    byId('data_model').addEventListener('change', loadDefaults);
    byId('data_dataset').addEventListener('change', async () => {
        await loadDataPreview();
        await loadDefaults();
    });
    byId('data_preview_file_type').addEventListener('change', loadDataPreview);
    byId('data_versions_dataset_filter').addEventListener('change', async (e) => {
        state.dataVersionsDatasetFilter = String(e.target.value || '');
        await loadDataVersions();
    });
    byId('data_preview_entity_id').addEventListener('change', loadDataPreview);
    byId('data_preview_time_start').addEventListener('change', loadDataPreview);
    byId('data_preview_time_end').addEventListener('change', loadDataPreview);
    byId('data_preview_columns').addEventListener('change', loadDataPreview);
    byId('train_data_version').addEventListener('change', async () => {
        applyTrainDataVersionSelection();
        await loadDefaults();
    });
    byId('data_version_select').addEventListener('change', () => {
        const selected = (state.dataVersions || []).find((x) => x.version_id === byId('data_version_select').value);
        renderDataVersionHint(selected || null);
    });
    await loadDefaults();
    await loadDataVersions();
    await loadDataPreview();
    applyTrainDataVersionSelection();
    await loadDefaults();
    byId('param_filter').addEventListener('input', renderParamTable);
    byId('model_plot_type').addEventListener('change', (e) => {
        state.modelPlotType = e.target.value === 'bar' ? 'bar' : 'pie';
        drawModelParamChart(state.modelPlot, state.modelPlotOptionPie, state.modelPlotOptionBar);
    });
    state.modelPlotTopKPie = clampTopK(state.modelPlotTopKPie, 8);
    state.modelPlotTopKBar = clampTopK(state.modelPlotTopKBar, 12);
    byId('model_plot_topk_pie').value = String(state.modelPlotTopKPie);
    byId('model_plot_topk_bar').value = String(state.modelPlotTopKBar);
    persistModelPlotTopK();
    await pushModelPlotTopK();
    byId('model_plot_topk_pie').addEventListener('change', async (e) => {
        const n = clampTopK(e.target.value, 8);
        e.target.value = String(n);
        state.modelPlotTopKPie = n;
        persistModelPlotTopK();
        await pushModelPlotTopK();
        await refreshStatus();
    });
    byId('model_plot_topk_bar').addEventListener('change', async (e) => {
        const n = clampTopK(e.target.value, 12);
        e.target.value = String(n);
        state.modelPlotTopKBar = n;
        persistModelPlotTopK();
        await pushModelPlotTopK();
        await refreshStatus();
    });
    updateAutoScrollBtnText();
    byId('log_level_filter').addEventListener('change', (e) => {
        state.logFilterLevel = String(e.target.value || 'all').toLowerCase();
        renderLogs(state.latestLogs);
    });
    byId('log_keyword_filter').addEventListener('input', (e) => {
        state.logFilterKeyword = String(e.target.value || '');
        renderLogs(state.latestLogs);
    });
    byId('log_autoscroll_btn').addEventListener('click', () => {
        state.logAutoScroll = !state.logAutoScroll;
        updateAutoScrollBtnText();
    });
    byId('pred_horizon').addEventListener('change', refreshPredictionSeries);
    byId('pred_node').addEventListener('change', refreshPredictionSeries);
    byId('pred_feature').addEventListener('change', refreshPredictionSeries);
    byId('resume_run_id').addEventListener('change', onResumeRunChanged);
    resetPredictionSelectors();
    initPaneResizers();
    initLayoutControls();
    initLayoutPopover();
    initMovableTiles();
    switchTab('train');
    await refreshStatus();
    setInterval(refreshStatus, 2000);
}

window.addEventListener('resize', () => {
    refreshPaneLayoutForViewport();
    resizeAllCharts();
});
init();
