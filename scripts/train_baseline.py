"""训练可在普通笔记本运行的基线模型；只在训练集拟合特征。"""
import json
import re
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]

def read_split(name):
    path = ROOT / 'data/splits' / f'{name}.jsonl'
    rows = [json.loads(line) for line in path.open(encoding='utf-8')]
    return [r['review'] for r in rows], [int(r['label']) for r in rows]

def remove_emoticons(text):
    return re.sub(r'\[[^\[\]]{1,15}\]', ' ', text)

def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'negative': {'precision': round(p[0], 4), 'recall': round(r[0], 4), 'f1': round(f1[0], 4)},
            'positive': {'precision': round(p[1], 4), 'recall': round(r[1], 4), 'f1': round(f1[1], 4)},
            'confusion': {'true_negative': int(tn), 'false_positive': int(fp),
                          'false_negative': int(fn), 'true_positive': int(tp)}}

def main():
    start = time.perf_counter()
    splits = {name: read_split(name) for name in ('train', 'validation', 'test')}
    val_results = []
    trained = {}
    for variant in ('no_emoticons', 'with_emoticons'):
        process = remove_emoticons if variant == 'no_emoticons' else lambda text: text
        x_train, y_train = splits['train']
        x_val, y_val = splits['validation']
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), min_df=3,
                                     max_features=120000,
                                     dtype=__import__('numpy').float32, sublinear_tf=True)
        t0 = time.perf_counter()
        a = vectorizer.fit_transform([process(t) for t in x_train])
        b = vectorizer.transform([process(t) for t in x_val])
        for name, model in (
            ('naive_bayes', MultinomialNB(alpha=1.0)),
            ('logistic_sgd', SGDClassifier(loss='log_loss', alpha=1e-5, random_state=20261010,
                                           max_iter=50, tol=1e-3))):
            model.fit(a, y_train)
            result = metrics(y_val, model.predict(b))
            entry = {'variant': variant, 'model': name, 'validation': result,
                     'elapsed_seconds_including_vectorization': round(time.perf_counter()-t0, 2)}
            val_results.append(entry)
            trained[(variant, name)] = (vectorizer, model)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
    # 面向现场自由输入，优先选择无表情版本，在其两种模型中以验证集 F1 选最优。
    candidates = [r for r in val_results if r['variant'] == 'no_emoticons']
    selected = max(candidates, key=lambda r: (r['validation']['positive']['f1'],
                                               r['validation']['negative']['f1']))
    vec, clf = trained[(selected['variant'], selected['model'])]
    x_test, y_test = splits['test']
    test_metrics = metrics(y_test, clf.predict(vec.transform([remove_emoticons(t) for t in x_test])))
    model_dir = ROOT / 'models'
    model_dir.mkdir(exist_ok=True)
    joblib.dump({'vectorizer': vec, 'classifier': clf, 'variant': selected['variant'],
                 'model': selected['model'], 'version': 1}, model_dir / 'baseline.joblib', compress=3)
    report = {'model_kind': 'scikit-learn baseline (not PySpark)',
              'training_rows': len(splits['train'][0]),
              'validation_rows': len(splits['validation'][0]),
              'test_rows': len(x_test), 'validation_comparisons': val_results,
              'selection_rule': '无表情方案中验证集正向F1最高；同分看负向F1',
              'selected': {'variant': selected['variant'], 'model': selected['model']},
              'final_test': test_metrics,
              'total_elapsed_seconds': round(time.perf_counter()-start, 2),
              'limitations': ['测试集只用于最终评价，诊断样例另测。',
                              '此实验尚未使用PySpark，不能据此声称完成Spark技术部分。',
                              '预测分数未经概率校准，不作为可靠概率。']}
    (ROOT/'reports/baseline_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'selected':report['selected'],'final_test':test_metrics,
                      'total_elapsed_seconds':report['total_elapsed_seconds']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
