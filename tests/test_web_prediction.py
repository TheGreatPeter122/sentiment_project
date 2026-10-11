import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import app
from scripts.predict_daily_reviews import predict


class WebPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def post(self, data):
        request = Request(self.url + '/predict', json.dumps(data).encode('utf-8'),
                          {'Content-Type': 'application/json'})
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_web_matches_selected_model_for_daily_comments(self):
        texts = ['我一点也不喜欢原神', '今天不开心。', '外卖送得很快，味道也不错',
                 '包装很好，但是东西太难用了']
        expected = predict(texts)
        for row in expected['results']:
            with self.subTest(text=row['text']):
                result = self.post({'text': row['text']})
                self.assertEqual(result['label'],
                                 '偏正向' if row['prediction'] == 'positive' else '偏负向')
                self.assertEqual(result['score'], round(row['score'], 3))
                self.assertEqual(result['model'], 'structure_v2')
                self.assertEqual(result['model_sha256'], expected['model_sha256'])

    def test_page_explains_current_training_sources(self):
        with urlopen(self.url, timeout=10) as response:
            page = response.read().decode('utf-8')
        for phrase in ('微博、购物和外卖', '结构 v2', '概率'):
            self.assertIn(phrase, page)

    def test_invalid_payloads_return_json_errors(self):
        for data in ([], {'text': None}, {'text': 123}, {'text': ' '}, {'text': '好' * 501}):
            with self.subTest(data=data):
                with self.assertRaises(HTTPError) as raised:
                    self.post(data)
                self.assertEqual(raised.exception.code, 400)
                self.assertIn('error', json.loads(raised.exception.read()))


if __name__ == '__main__':
    unittest.main()
