"""本地交互演示：.venv/Scripts/python.exe app.py"""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from daily_review_model import load_model, predict

load_model()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / 'docs'), **kwargs)

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404)
            return
        try:
            n = int(self.headers.get('Content-Length','0'))
            if not 0 < n < 10000:
                raise ValueError('输入长度不合适')
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict):
                raise ValueError('请求需要包含 text 字段')
            prediction = predict([data.get('text', '')], data.get('model', 'char2'))
            row = prediction['results'][0]
            result = {'label':'偏正向' if row['prediction'] == 'positive' else '偏负向',
                      'score':round(row['score'], 3), 'note':prediction['note'],
                      'model':prediction['model'], 'model_sha256':prediction['model_sha256']}
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
