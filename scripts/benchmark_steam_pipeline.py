"""Compare identical Steam JSONL cleaning and aggregation in Python and local Spark."""
import hashlib
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/steam_reviews_bulk.jsonl'
AUDIT = ROOT / 'reports/steam_reviews_bulk.json'
OUT = ROOT / 'reports/steam_spark_comparison.json'
STAGED = ROOT / 'runtime/steam_benchmark'
SIZES = (10000, 50000, 102902)
REPEATS = 3
SPACE = re.compile(r'\s+')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_inputs():
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if digest(SOURCE) != audit['output_sha256'] or audit['counts']['accepted'] != SIZES[-1]:
        raise ValueError('Steam source differs from collection audit')
    lines = []
    seen = set()
    with SOURCE.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            review_id = str(row['review_id'])
            if review_id in seen or not row['text'] or row['weak_label'] not in ('negative', 'positive'):
                raise ValueError('Invalid or duplicate Steam input')
            seen.add(review_id)
            lines.append((hashlib.sha256(('steam-benchmark-v1:' + review_id).encode()).digest(), line))
    if len(lines) != SIZES[-1]:
        raise ValueError('Unexpected Steam row count')
    lines.sort(key=lambda pair: pair[0])
    STAGED.mkdir(parents=True, exist_ok=True)
    paths = {}
    for size in SIZES:
        path = STAGED / f'{size}.jsonl'
        path.write_text(''.join(line for _, line in lines[:size]), encoding='utf-8')
        paths[size] = path
    return paths, audit['output_sha256']


def length_bucket(length):
    if length <= 40:
        return '001-040'
    if length <= 100:
        return '041-100'
    return '101+'


def python_counts(path):
    counts = defaultdict(lambda: [0, 0])
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            clean = SPACE.sub(' ', row['text']).strip()
            key = (int(row['app_id']), row['weak_label'], row['created_at_utc'][:7],
                   length_bucket(len(clean)))
            counts[key][0] += 1
            counts[key][1] += len(clean)
    return dict(counts)


def spark_counts(spark, path):
    from pyspark.sql import functions as F, types as T

    schema = T.StructType([
        T.StructField('app_id', T.LongType()),
        T.StructField('weak_label', T.StringType()),
        T.StructField('text', T.StringType()),
        T.StructField('created_at_utc', T.StringType()),
    ])
    source = spark.read.schema(schema).json(str(path))
    frame = (source.withColumn('clean', F.trim(F.regexp_replace('text', r'\s+', ' ')))
             .withColumn('chars', F.length('clean'))
             .withColumn('month', F.substring('created_at_utc', 1, 7))
             .withColumn('bucket', F.when(F.col('chars') <= 40, '001-040')
                         .when(F.col('chars') <= 100, '041-100').otherwise('101+')))
    collected = (frame.groupBy('app_id', 'weak_label', 'month', 'bucket')
                 .agg(F.count('*').alias('rows'), F.sum('chars').alias('characters')).collect())
    return {(int(row['app_id']), row['weak_label'], row['month'], row['bucket']):
            [int(row['rows']), int(row['characters'])] for row in collected}


def summarize(groups):
    by_game_label = defaultdict(int)
    by_month = defaultdict(int)
    by_length = defaultdict(int)
    total_chars = 0
    for (app_id, label, month, bucket), (count, chars) in groups.items():
        by_game_label[f'{app_id}/{label}'] += count
        by_month[month] += count
        by_length[bucket] += count
        total_chars += chars
    rows = sum(by_game_label.values())
    return {'rows': rows, 'characters_after_whitespace_normalization': total_chars,
            'average_characters': round(total_chars / rows, 2),
            'by_game_label': dict(sorted(by_game_label.items())),
            'by_month': dict(sorted(by_month.items())),
            'by_length_bucket': dict(sorted(by_length.items()))}


def main():
    paths, source_hash = stage_inputs()  # Deterministic staging is outside both timers.
    python_results = {}
    for size, path in paths.items():
        runs = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            groups = python_counts(path)
            runs.append(time.perf_counter() - started)
        python_results[size] = {'groups': groups, 'seconds': runs}

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master('local[4]').appName('SteamPipelineComparison')
             .config('spark.ui.enabled', 'false')
             .config('spark.driver.bindAddress', '127.0.0.1')
             .config('spark.sql.shuffle.partitions', '8').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        spark.range(1).count()  # JVM startup excluded; the first real input can still be colder.
        results = []
        for size, path in paths.items():
            spark_runs = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                groups = spark_counts(spark, path)
                spark_runs.append(time.perf_counter() - started)
                if groups != python_results[size]['groups']:
                    differing = sum(groups.get(key) != python_results[size]['groups'].get(key)
                                    for key in groups.keys() | python_results[size]['groups'].keys())
                    raise ValueError(f'Python/Spark aggregate mismatch: {size} rows, {differing} groups')
            results.append({'input_rows': size, 'input_sha256': digest(path),
                            'matching_aggregate_groups': len(groups),
                            'python_seconds': [round(t, 3) for t in python_results[size]['seconds']],
                            'spark_seconds': [round(t, 3) for t in spark_runs],
                            'python_median_seconds': round(statistics.median(python_results[size]['seconds']), 3),
                            'spark_median_seconds': round(statistics.median(spark_runs), 3),
                            'aggregates_identical': True})
        report = {'source_path': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': source_hash,
                  'engine': {'python': 'CPython standard library, single process',
                             'spark': 'PySpark DataFrame local[4]', 'spark_version': spark.version},
                  'task': 'Read identical JSONL; normalize whitespace; count characters; group by game, weak label, UTC month and length bucket; compute group count and character total',
                  'staging': 'SHA-256(seed + review_id) sorted nested prefixes; generated outside timing',
                  'repeats_per_size': REPEATS, 'sizes': results,
                  'full_data_summary': summarize(python_results[SIZES[-1]]['groups']),
                  'timing_excludes': ['input staging', 'Spark JVM startup'],
                  'limitations': ['One Windows computer running Spark local[4], not a multi-machine cluster',
                                  'Python and Spark runs are sequential; operating-system file cache and runtime warm-up affect timing',
                                  'This benchmark measures this specific task and input size, not general framework superiority',
                                  'Steam recommendation choices are weak labels and sampled proportions are not natural prevalence']}
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'sizes': results, 'full_data_summary': report['full_data_summary']},
                         ensure_ascii=False, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
