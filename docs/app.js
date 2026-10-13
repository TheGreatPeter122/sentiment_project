const input = document.querySelector('#text-input');
const count = document.querySelector('#char-count');
const button = document.querySelector('#analyze');
const result = document.querySelector('#result');
const status = document.querySelector('#load-status');
const worker = new Worker('./predict-worker.js');
let ready = false;
let requestId = 0;

function updateCount() {
  count.textContent = `${Array.from(input.value).length} / 500 字`;
}

function setStatus(message, kind = '') {
  status.textContent = message;
  status.className = `status ${kind}`.trim();
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
}

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
    result.textContent = `无法分析：${message.error}`;
  }
};

worker.onerror = () => {
  setStatus('加载失败', 'error');
  result.textContent = '浏览器未能运行模型，请刷新页面重试。';
};

input.addEventListener('input', updateCount);
button.addEventListener('click', () => {
  const text = input.value.trim();
  if (!text || Array.from(text).length > 500) {
    result.textContent = '请输入 1—500 字的中文短评。';
    return;
  }
  if (!ready) return;
  button.disabled = true;
  result.textContent = '正在分析…';
  worker.postMessage({ id: ++requestId, type: 'predict', text });
});

document.querySelectorAll('[data-example]').forEach(example => {
  example.addEventListener('click', () => {
    input.value = example.dataset.example;
    updateCount();
    input.focus();
  });
});

updateCount();
worker.postMessage({ id: 0, type: 'load' });
