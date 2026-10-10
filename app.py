"""本地交互演示：.venv/Scripts/python.exe app.py"""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
artifact = joblib.load(ROOT / 'models/baseline.joblib')

PAGE = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>中文短文本情感识别</title><style>
body{font:17px/1.65 system-ui,"Microsoft YaHei",sans-serif;max-width:780px;margin:7vh auto;padding:0 24px;color:#17212b;background:#f8fafb}
h1{font-size:31px;margin-bottom:8px}p{color:#475569}textarea{box-sizing:border-box;width:100%;height:135px;padding:16px;border:1px solid #b9c4d0;border-radius:10px;font:inherit;background:white}
button{border:0;border-radius:8px;background:#155e75;color:white;padding:10px 24px;font:inherit;cursor:pointer;margin-top:12px}button:hover{background:#0e475a}
#result{margin-top:24px;padding:18px 20px;background:white;border-left:5px solid #155e75;border-radius:5px;min-height:36px}
.hint{font-size:14px}.examples button{background:#e2edf0;color:#163946;margin:4px 8px 4px 0;padding:5px 10px;font-size:14px}
</style></head><body><h1>中文短文本情感识别</h1>
<p>输入一句中文，查看模型判断。它只学习了“正向”和“负向”两类，对中性、反讽和混合情感可能判断不准。</p>
<textarea id="input" maxlength="500" placeholder="例如：虽然过程很辛苦，但结果让我很满意。"></textarea><br>
<button id="submit">分析文本</button>
<div class="examples"><p class="hint">试试这些句子：</p>
<button data-text="今天终于完成了任务，特别开心。">开心</button>
<button data-text="今天不开心。">否定</button>
<button data-text="等了三个小时，服务可真好。">反讽</button>
<button data-text="今天是星期六。">中性</button></div>
<div id="result" role="status">等待输入</div>
<p class="hint">研究演示。模型在微博语料上训练，结果只是文本倾向判断，不代表对作者真实情绪的测量。</p>
<script>
const input=document.querySelector('#input'),result=document.querySelector('#result');
document.querySelectorAll('[data-text]').forEach(b=>b.onclick=()=>{input.value=b.dataset.text;input.focus()});
document.querySelector('#submit').onclick=async()=>{
  const text=input.value.trim();if(!text){result.textContent='请先输入文字。';return;}
  result.textContent='分析中…';
  try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');
  result.textContent=`判断：${d.label}。模型分数：${d.score}。${d.note}`;
  }catch(e){result.textContent='无法分析：'+e.message}
};
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        body = PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404)
            return
        try:
            n = int(self.headers.get('Content-Length','0'))
            if not 0 < n < 10000:
                raise ValueError('输入长度不合适')
            data = json.loads(self.rfile.read(n))
            text = data.get('text','').strip()
            if not text or len(text) > 500:
                raise ValueError('请输入1—500字的内容')
            clean = re.sub(r'\[[^\[\]]{1,15}\]', ' ', text) if artifact['variant']=='no_emoticons' else text
            x = artifact['vectorizer'].transform([clean])
            score = float(artifact['classifier'].decision_function(x)[0])
            result = {'label':'偏正向' if score >= 0 else '偏负向',
                      'score':round(score,3),
                      'note':'分数表示相对判断方向，不是正确概率。'}
            code = 200
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            result, code = {'error':str(e)}, 400
        body = json.dumps(result,ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    import os
    import sys
    import threading
    import webbrowser

    port = int(os.environ.get('SENTIMENT_PORT', '8765'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}'
    print(f'打开 {url} ；按 Ctrl+C 结束', flush=True)
    if '--open' in sys.argv[1:]:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()
