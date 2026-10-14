const input = document.querySelector('#text-input');
const count = document.querySelector('#char-count');
const button = document.querySelector('#analyze');
const result = document.querySelector('#result');
const status = document.querySelector('#load-status');
const themeToggle = document.querySelector('#theme-toggle');
const modelSelect = document.querySelector('#model-select');
const retryModel = document.querySelector('#retry-model');
const descriptions = {
  char2: '字与相邻双字片段，配合否定、转折线索。原模型，默认选中。',
  jieba: '用 jieba 精确模式切词，再提取单词与相邻词对；保留否定、转折线索。',
  char4: '提取长度为 1、2、3、4 的字符片段，保留否定、转折线索。'
};
let selectedModel = 'char2';
let ready = false;
let requestId = 0;
let face = null;

/* ---------------------------------------------------------------
   两种运行方式，自动切换：
   1) http(s)://（GitHub Pages、本地服务器）→ 用 Web Worker，界面不卡
   2) file://（直接双击打开）→ 浏览器禁止创建 Worker，
      改用 <script> 加载 model.js（不受 CORS 限制）在主线程推理
   --------------------------------------------------------------- */
const isFile = location.protocol === 'file:';
let worker = null;
let model = null;
let tokenize = null;
let inlineBase;
let inlineDictionary;
const scriptCache = new Map();

function updateCount() {
  count.textContent = `${Array.from(input.value).length} / 500 字`;
}

function setStatus(message, kind = '') {
  status.textContent = message;
  status.className = `status ${kind}`.trim();
}

function loadScript(src) {
  if (scriptCache.has(src)) return scriptCache.get(src);
  const promise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => { scriptCache.delete(src); s.remove(); reject(new Error('无法加载 ' + src)); };
    document.head.appendChild(s);
  });
  scriptCache.set(src, promise);
  return promise;
}

function showFatal(message) {
  ready = false;
  button.disabled = true;
  retryModel.hidden = false;
  setStatus('加载失败', 'error');
  result.textContent = message;
}

// —— 模式 1：Worker ——
function bootWorker() {
  worker = new Worker('./predict-worker.js');
  worker.onmessage = event => {
    const message = event.data;
    if (message.id !== requestId || message.modelId !== selectedModel) return;
    if (message.type === 'ready') {
      ready = true;
      button.disabled = false;
      setStatus('模型已就绪', 'ready');
    } else if (message.type === 'result' && message.id === requestId) {
      button.disabled = false;
      showPrediction(message.prediction, message.score, message.tokens);
    } else if (message.type === 'error') {
      showFatal(`无法分析：${message.error}`);
    }
  };
  worker.onerror = () => {
    showFatal('浏览器未能运行模型，请重新加载或刷新页面。');
  };
}

// —— 模式 2：主线程（file:// 用）——
async function bootInline(id, modelId) {
  try {
    setStatus('正在加载模型…（本地文件模式）');
    if (!inlineBase) inlineBase = Promise.all([loadScript('./inference.js'), loadScript('./jieba-tokenizer.js')])
      .catch(error => { inlineBase = null; throw error; });
    await inlineBase;
    const entry = MODEL_CATALOG.models[modelId];
    await loadScript('./' + entry.url.replace(/\.json$/, '.js'));
    let selectedTokenizer = null;
    if (modelId === 'jieba') {
      if (!inlineDictionary) inlineDictionary = loadScript('./models/jieba-dictionary.js')
        .then(() => JiebaTokenizer.create(window.JIEBA_DICTIONARY))
        .catch(error => { inlineDictionary = null; throw error; });
      selectedTokenizer = await inlineDictionary;
    }
    if (id !== requestId || modelId !== selectedModel) return;
    if (!window.SENTIMENT_MODELS?.[modelId] || typeof SentimentInference === 'undefined') {
      throw new Error('模型数据缺失');
    }
    model = window.SENTIMENT_MODELS[modelId];
    tokenize = selectedTokenizer;
    ready = true;
    button.disabled = false;
    setStatus('模型已就绪（本地文件模式）', 'ready');
  } catch (err) {
    if (id !== requestId) return;
    showFatal('本地文件模式加载失败：' + err.message +
      '（请确认 model.js 与 inference.js 已放在同一文件夹，或改用本地服务器打开）');
  }
}

function selectModel() {
  selectedModel = modelSelect.value;
  const id = ++requestId;
  ready = false;
  button.disabled = true;
  retryModel.hidden = true;
  document.querySelector('#model-description').textContent = descriptions[selectedModel];
  resetResult('模型切换中，文本已保留。加载完成后点击分析。');
  setStatus('正在加载所选模型…');
  if (!isFile) {
    try {
      if (!worker) bootWorker();
      worker.postMessage({ id, type: 'load', modelId: selectedModel });
      return;
    } catch (_) { if (worker) worker.terminate(); worker = null; }
  }
  bootInline(id, selectedModel);
}

// 清空结果区，并把表情恢复为待分析状态（表情元素保存在 face.el，不会被丢弃）
function resetResult(message) {
  result.replaceChildren();
  if (message) result.textContent = message;
  if (face) face.reset();
}

function showPrediction(prediction, score, tokens) {
  result.replaceChildren();
  result.className = '';
  const card = document.createElement('div');
  card.className = `result-card ${prediction}`;
  const words = document.createElement('div');
  const kicker = document.createElement('div');
  kicker.className = 'result-kicker';
  kicker.textContent = MODEL_CATALOG.models[selectedModel].label;
  const label = document.createElement('div');
  label.className = 'result-label';
  label.textContent = prediction === 'positive' ? '偏正向 ↗' : '偏负向 ↘';
  const detail = document.createElement('div');
  detail.className = 'result-detail';
  detail.textContent = '这是模型根据训练文本作出的二分类判断，不代表作者的真实情绪。';
  const value = document.createElement('div');
  value.className = 'score';
  value.textContent = `决策分数 ${score.toFixed(3)}`;
  words.append(kicker, label, detail);
  card.append(words);
  result.append(card);
  if (tokens) {
    const tokenLine = document.createElement('p');
    tokenLine.className = 'token-preview';
    tokenLine.textContent = 'jieba 切分：' + tokens.join(' / ');
    result.append(tokenLine);
  }
  if (Math.abs(score) <= 0.5) {
    const caution = document.createElement('p');
    caution.className = 'uncertain-note';
    caution.textContent = '分数接近 0，判断把握较低；模型仍只输出正、负两类。';
    result.append(caution);
  }

  // 表情仅用于呈现模型分数；低分数不代表第三个“中性”类别。
  // 优先用照片版（SentimentFacePhoto）；没有素材时回退到矢量版（SentimentFace）
  const FaceLib = (typeof SentimentFacePhoto !== 'undefined') ? SentimentFacePhoto
    : (typeof SentimentFace !== 'undefined' ? SentimentFace : null);
  if (FaceLib) {
    const faceBox = document.createElement('div');
    faceBox.className = 'result-face';
    result.append(faceBox);
    if (!face) face = FaceLib.create(faceBox);
    else faceBox.append(face.el);
    // 瞬间切换：直接落到目标表情，不做淡入动画
    face.update(score, prediction);
  } else card.append(value);
}

function predict(text, id) {
  if (worker) {
    worker.postMessage({ id: id, type: 'predict', text, modelId: selectedModel });
    return;
  }
  // 主线程模式：先让界面把“正在分析…”画出来，再跑推理
  requestAnimationFrame(() => setTimeout(() => {
    try {
      if (id !== requestId) return;
      const r = SentimentInference.predict(text, model, tokenize);
      if (id === requestId) { button.disabled = false; showPrediction(r.prediction, r.score, r.tokens); }
    } catch (err) {
      button.disabled = false;
      resetResult('无法分析：' + (err.message || err));
    }
  }, 0));
}

input.addEventListener('input', updateCount);
button.addEventListener('click', () => {
  const text = input.value.trim();
  if (!text || Array.from(text).length > 500) {
    resetResult('请输入 1—500 字的中文短评。');
    return;
  }
  if (!ready) return;
  button.disabled = true;
  resetResult('正在分析…');
  predict(text, ++requestId);
});

// —— 按协议选择运行方式 ——
modelSelect.addEventListener('change', selectModel);
retryModel.addEventListener('click', () => { if (worker) worker.terminate(); worker = null; selectModel(); });
selectModel();

document.querySelectorAll('[data-example]').forEach(example => {
  example.addEventListener('click', () => {
    input.value = example.dataset.example;
    updateCount();
    input.focus();
  });
});

updateCount();

function syncThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  themeToggle.setAttribute('aria-pressed', String(dark));
  themeToggle.setAttribute('aria-label', dark ? '切换浅色模式' : '切换深色模式');
  document.querySelector('meta[name="theme-color"]').content = dark ? '#09090b' : '#f5f5f7';
}
themeToggle.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('sentiment-theme', next); } catch (_) { /* 私密模式仍可切换 */ }
  syncThemeButton();
});
syncThemeButton();
