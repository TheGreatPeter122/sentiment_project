importScripts('./inference.js', './jieba-tokenizer.js', './model-catalog.js');
const cache = new Map();
let tokenizerPromise;

async function fetchJson(entry) {
  const response = await fetch('./' + entry.url);
  if (!response.ok) throw new Error('下载失败：HTTP ' + response.status);
  const bytes = await response.arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
    .map(b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== entry.sha256) throw new Error('模型文件版本不一致，请刷新页面重试。');
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function getModel(modelId) {
  const entry = MODEL_CATALOG.models[modelId];
  if (!entry) throw new Error('未知模型。');
  if (!cache.has(modelId)) {
    cache.set(modelId, fetchJson(entry).catch(error => { cache.delete(modelId); throw error; }));
  }
  const model = await cache.get(modelId);
  let tokenize;
  if (modelId === 'jieba') {
    if (!tokenizerPromise) tokenizerPromise = fetchJson(MODEL_CATALOG.jieba_dictionary)
      .then(data => JiebaTokenizer.create(data)).catch(error => { tokenizerPromise = null; throw error; });
    tokenize = await tokenizerPromise;
  }
  return { model, tokenize };
}

self.onmessage = async event => {
  const { id, type, text, modelId = 'char2' } = event.data;
  try {
    const { model, tokenize } = await getModel(modelId);
    if (type === 'load') {
      self.postMessage({ id, modelId, type: 'ready', modelSha256: model.source_model_sha256 });
    } else if (type === 'predict') {
      self.postMessage({ id, modelId, type: 'result', ...SentimentInference.predict(text, model, tokenize) });
    }
  } catch (error) {
    self.postMessage({ id, modelId, type: 'error', error: error.message || String(error) });
  }
};
