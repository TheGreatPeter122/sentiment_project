"""Use Spark local mode to analyze every cleaned source; never train a model."""
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

ROOT = Path(__file__).resolve().parents[1]
BASE_SCHEMA = T.StructType([
    T.StructField('label', T.StringType()),
    T.StructField('review', T.StringType()),
    T.StructField('source', T.StringType()),
    T.StructField('split', T.StringType()),
    T.StructField('categories', T.ArrayType(T.StringType())),
])


def spark_session(name):
    spark = (SparkSession.builder.master('local[4]').appName(name)
             .config('spark.ui.enabled', 'false')
             .config('spark.driver.bindAddress', '127.0.0.1')
             .config('spark.sql.shuffle.partitions', '8').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    return spark


def transformed(frame):
    """Apply the same Spark transformations in analysis and scale benchmark."""
    return (frame
            .withColumn('has_emoticon', F.col('review').rlike(r'\[[^\[\]]{1,15}\]'))
            .withColumn('clean', F.regexp_replace('review', r'\[[^\[\]]{1,15}\]', ' '))
            .withColumn('clean', F.regexp_replace('clean', r'https?://\S+|www\.\S+', ' '))
            .withColumn('clean', F.regexp_replace('clean', r'@[^\s:：/@]+', ' '))
            .withColumn('clean', F.regexp_replace('clean', r'\s+', ' '))
            .withColumn('clean_length', F.length(F.trim('clean'))))


def read_all(spark):
    parts = []
    for split in ('train', 'validation', 'test'):
        part = (spark.read.json(str(ROOT / 'data/splits' / f'{split}.jsonl'))
                .select('label', 'review')
                .withColumn('source', F.lit('weibo_senti_100k'))
                .withColumn('split', F.lit(split))
                .withColumn('categories', F.array().cast(T.ArrayType(T.StringType()))))
        parts.append(part)
    external = (spark.read.json(str(ROOT / 'data/processed/external_clean.jsonl'))
                .select('label', 'review', 'sources', 'categories')
                .withColumn('source', F.when(F.size('sources') > 1, 'multiple_external_sources')
                            .otherwise(F.element_at('sources', 1)))
                .withColumn('split', F.lit('external'))
                .drop('sources'))
    parts.append(external)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.unionByName(part)
    return combined


def main():
    started = time.perf_counter()
    spark = spark_session('SentimentAllSources')
    try:
        frame = transformed(read_all(spark)).cache()
        by_source = frame.groupBy('source').agg(
            F.count('*').alias('rows'),
            F.round(F.avg('clean_length'), 2).alias('average_clean_length'),
            F.min('clean_length').alias('min_clean_length'),
            F.max('clean_length').alias('max_clean_length'),
            F.sum(F.col('has_emoticon').cast('long')).alias('with_emoticon'),
        ).collect()
        by_source_label = frame.groupBy('source', 'label').count().collect()
        by_split = frame.groupBy('split').count().collect()
        categories = (frame.where(F.col('split') == 'external')
                      .select('label', F.explode('categories').alias('category'))
                      .groupBy('category', 'label').count().collect())
        total = sum(int(row['rows']) for row in by_source)
        assert total == 110649 + 74274, total
        report = {
            'engine': 'PySpark DataFrame SQL', 'master': 'local[4]',
            'spark_version': spark.version, 'rows': total,
            'by_source': {row['source']: {
                'rows': int(row['rows']),
                'average_clean_length': float(row['average_clean_length']),
                'min_clean_length': int(row['min_clean_length']),
                'max_clean_length': int(row['max_clean_length']),
                'with_emoticon': int(row['with_emoticon']),
            } for row in by_source},
            'by_source_label': {f"{row['source']}:{row['label']}": int(row['count'])
                                for row in by_source_label},
            'by_split': {row['split']: int(row['count']) for row in by_split},
            'external_by_category_label': {
                f"{row['category']}:{row['label']}": int(row['count']) for row in categories},
            'operations': ['read JSONL', 'union sources', 'Spark regex cleaning',
                           'text length', 'group by source, label, split and category'],
            'elapsed_seconds_including_spark_startup': round(time.perf_counter() - started, 2),
            'note': 'One Windows PC, Spark local[4]. No cluster or model training. '
                    'Source counts assign cross-source duplicates to multiple_external_sources.',
        }
        out = ROOT / 'reports/spark_all_sources.json'
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
