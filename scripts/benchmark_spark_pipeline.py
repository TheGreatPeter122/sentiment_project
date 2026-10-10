"""Measure the same Spark cleaning/grouping pipeline at three input sizes."""
import hashlib
import json
import statistics
import time
from pathlib import Path

from pyspark.sql import functions as F

from spark_all_sources import BASE_SCHEMA, ROOT, spark_session, transformed

SIZES = (10000, 50000, 184923)


def stage_inputs():
    rows = []
    for split in ('train', 'validation', 'test'):
        with (ROOT / 'data/splits' / f'{split}.jsonl').open(encoding='utf-8') as file:
            for line in file:
                row = json.loads(line)
                rows.append({'label': row['label'], 'review': row['review'],
                             'source': 'weibo_senti_100k', 'split': split,
                             'categories': []})
    with (ROOT / 'data/processed/external_clean.jsonl').open(encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            rows.append({'label': row['label'], 'review': row['review'],
                         'source': row['sources'][0] if len(row['sources']) == 1
                         else 'multiple_external_sources', 'split': 'external',
                         'categories': row['categories']})
    if len(rows) != SIZES[-1]:
        raise ValueError(f'Unexpected combined row count: {len(rows)}')
    rows.sort(key=lambda row: hashlib.sha256(
        (row['source'] + '\0' + row['review']).encode('utf-8')).digest())
    out = ROOT / 'runtime/spark_benchmark'
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for size in SIZES:
        path = out / f'{size}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                for row in rows[:size]), encoding='utf-8')
        paths[size] = path
    return paths


def main():
    paths = stage_inputs()  # Staging is intentionally outside the timed Spark tasks.
    spark = spark_session('SentimentScaleBenchmark')
    try:
        # Start the JVM before timing; each run includes JSON read and Spark computation.
        spark.range(1).count()
        results = []
        for size, path in paths.items():
            runs = []
            for _ in range(3):
                started = time.perf_counter()
                source = spark.read.schema(BASE_SCHEMA).json(str(path))
                counts = (transformed(source).groupBy('source', 'label').agg(
                    F.count('*').alias('rows'),
                    F.avg('clean_length').alias('average_clean_length'),
                    F.sum(F.col('has_emoticon').cast('long')).alias('with_emoticon'),
                ).collect())
                elapsed = round(time.perf_counter() - started, 3)
                if sum(int(row['rows']) for row in counts) != size:
                    raise ValueError(f'Count mismatch for {size} rows')
                runs.append(elapsed)
            results.append({'rows': size, 'seconds': runs,
                            'median_seconds': round(statistics.median(runs), 3),
                            'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        report = {
            'engine': 'PySpark DataFrame SQL', 'master': 'local[4]',
            'spark_version': spark.version, 'sizes': results,
            'timed_work': 'read JSONL, regex cleaning, text length and group by source/label',
            'excluded_from_timing': 'Python staging of deterministic input files and Spark JVM startup',
            'repeats_per_size': 3,
            'limitations': ['One Windows PC, not a distributed cluster.',
                            'Local file cache, JVM warm-up and system load can affect timing.',
                            'These timings alone do not establish Spark speedup over Python.'],
        }
        (ROOT / 'reports/spark_scale_benchmark.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
