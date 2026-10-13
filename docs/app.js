const input = document.querySelector('#text-input');
const count = document.querySelector('#char-count');
const button = document.querySelector('#analyze');
const result = document.querySelector('#result');
const status = document.querySelector('#load-status');
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

function updateCount() {
  count.textContent = `${Array.from(input.value).length} / 500 字`;
}

function setStatus(message, kind = '') {
  status.textContent = message;
  status.className = `status ${kind}`.trim();
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('无法加载 ' + src));
    document.head.appendChild(s);
  });
}

function showFatal(message) {
  setStatus('加载失败', 'error');
  result.textContent = message;
}

// —— 模式 1：Worker ——
function bootWorker() {
  worker = new Worker('./predict-worker.js');
  worker.onmessage = event => {
    const message = event.data;
    if (message.type === 'ready') {
      ready = true;
      button.disabled = false;
      setStatus('模型已就绪', 'ready');
    } else if (message.type === 'result' && message.id === requestId) {
      button.disabled = false;
      showPrediction(message.prediction, message.score);
    } else if (message.type === 'error') {
      button.disabled = !ready;
      setStatus(ready ? '模型已就绪' : '加载失败', ready ? 'ready' : 'error');
      resetResult(`无法分析：${message.error}`);
    }
  };
  worker.onerror = () => {
    setStatus('加载失败', 'error');
    resetResult('浏览器未能运行模型，请刷新页面重试。');
  };
  worker.postMessage({ id: 0, type: 'load' });
}

// —— 模式 2：主线程（file:// 用）——
async function bootInline() {
  try {
    setStatus('正在加载模型…（本地文件模式）');
    await loadScript('./inference.js');
    await loadScript('./model.js');
    if (!window.SENTIMENT_MODEL || typeof SentimentInference === 'undefined') {
      throw new Error('模型数据缺失');
    }
    model = window.SENTIMENT_MODEL;
    window.SENTIMENT_MODEL = null;      // 交给 model 变量持有，便于回收
    ready = true;
    button.disabled = false;
    setStatus('模型已就绪（本地文件模式）', 'ready');
  } catch (err) {
    showFatal('本地文件模式加载失败：' + err.message +
      '（请确认 model.js 与 inference.js 已放在同一文件夹，或改用本地服务器打开）');
  }
}

// 清空结果区，并把表情恢复为待分析状态（表情元素保存在 face.el，不会被丢弃）
function resetResult(message) {
  result.replaceChildren();
  if (message) result.textContent = message;
  if (face) face.reset();
}

function showPrediction(prediction, score) {
  result.replaceChildren();
  const card = document.createElement('div');
  card.className = `result-card ${prediction}`;
  const words = document.createElement('div');
  const label = document.createElement('div');
  label.className = 'result-label';
  label.textContent = prediction === 'positive' ? '偏正向 ↗' : '偏负向 ↘';
  const detail = document.createElement('div');
  detail.className = 'result-detail';
  detail.textContent = '这是模型根据训练文本作出的二分类判断，不代表作者的真实情绪。';
  const value = document.createElement('div');
  value.className = 'score';
  value.textContent = `决策分数 ${score.toFixed(3)}`;
  words.append(label, detail);
  card.append(words, value);
  result.append(card);

  // 表情可视化：score > 0.5 笑脸图、−0.5~0.5 中性图、< −0.5 哭脸图
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
  }
}

function predict(text, id) {
  if (worker) {
    worker.postMessage({ id: id, type: 'predict', text });
    return;
  }
  // 主线程模式：先让界面把“正在分析…”画出来，再跑推理
  requestAnimationFrame(() => setTimeout(() => {
    try {
      const r = SentimentInference.predict(text, model);
      if (id === requestId) { button.disabled = false; showPrediction(r.prediction, r.score); }
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
if (isFile) {
  bootInline();
} else {
  try {
    bootWorker();
  } catch (err) {
    bootInline();          // 极端情况下退回主线程模式
  }
}

document.querySelectorAll('[data-example]').forEach(example => {
  example.addEventListener('click', () => {
    input.value = example.dataset.example;
    updateCount();
    input.focus();
  });
});

updateCount();
