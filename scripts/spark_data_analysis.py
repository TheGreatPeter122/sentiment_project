"""使用真实 PySpark DataFrame 对完整训练数据做清洗与统计。"""
import json
import time
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
started = time.perf_counter()
spark = (SparkSession.builder.master('local[4]').appName('SentimentDataAnalysis')
         .config('spark.ui.enabled','false')
         .config('spark.driver.bindAddress','127.0.0.1')
         .config('spark.sql.shuffle.partitions','8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
try:
    raw = spark.read.json(str(ROOT/'data/splits/train.jsonl'))
    frame = raw.select(F.col('label').alias('label'), F.col('review').alias('review'))
    frame = frame.withColumn('has_emoticon', F.col('review').rlike(r'\[[^\[\]]{1,15}\]'))
    frame = frame.withColumn('clean', F.regexp_replace('review',r'\[[^\[\]]{1,15}\]',' '))
    frame = frame.withColumn('clean', F.regexp_replace('clean',r'https?://\S+|www\.\S+',' '))
    frame = frame.withColumn('clean', F.regexp_replace('clean',r'@[^\s:：/@]+',' '))
    frame = frame.withColumn('clean_length',F.length(F.trim('clean')))
    summary = frame.groupBy('label').agg(
        F.count('*').alias('count'), F.round(F.avg('clean_length'),2).alias('average_clean_length'),
        F.sum(F.col('has_emoticon').cast('long')).alias('with_emoticon')).collect()
    report = {'engine':'PySpark DataFrame SQL','spark_version':spark.version,
              'rows':sum(int(r['count']) for r in summary),
              'by_label':{r['label']:{'count':int(r['count']),
                                   'average_clean_length':float(r['average_clean_length']),
                                   'with_emoticon':int(r['with_emoticon'])} for r in summary},
              'operations':['Spark读取JSONL','Spark正则清洗表情、链接和@提及','Spark计算文本长度',
                            'Spark按标签分组统计'],
              'elapsed_seconds':round(time.perf_counter()-started,2),
              'note':'在一台Windows电脑以local[4]运行；模型训练与现场推理使用另行记录的scikit-learn基线。'}
    assert report['rows']==88814
    (ROOT/'reports/spark_data_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
finally:
    spark.stop()
