importScripts('./inference.js');

let model;

self.onmessage = async event => {
  const { id, type, text } = event.data;
  try {
    if (type === 'load') {
      const response = await fetch('./model.json');
      if (!response.ok) throw new Error('模型文件下载失败：HTTP ' + response.status);
      model = await response.json();
      self.postMessage({ id, type: 'ready', modelSha256: model.source_model_sha256 });
    } else if (type === 'predict') {
      if (!model) throw new Error('模型尚未加载完成。');
      self.postMessage({ id, type: 'result', ...self.SentimentInference.predict(text, model) });
    }
  } catch (error) {
    self.postMessage({ id, type: 'error', error: error.message || String(error) });
  }
};
