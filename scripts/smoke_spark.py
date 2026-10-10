import json
import platform
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.ml.classification import NaiveBayes

root = Path(__file__).resolve().parents[1]
spark = (SparkSession.builder.master('local[2]').appName('sentiment-environment-check')
         .config('spark.ui.enabled','false').config('spark.driver.bindAddress','127.0.0.1')
         .config('spark.sql.shuffle.partitions','2').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
try:
    data = spark.read.json(str(root/'data/splits/train.jsonl'))
    count = data.count()
    assert count > 80000
    # 使用 Spark SQL 构建样例，验证实际 MLlib 训练链路。
    tiny = spark.sql("""SELECT * FROM VALUES
        (1.0D, '开心 满意'), (1.0D, '喜欢 开心'),
        (0.0D, '失望 难过'), (0.0D, '讨厌 失望') AS t(label,text)""")
    features = HashingTF(inputCol='words',outputCol='features',numFeatures=128).transform(
        Tokenizer(inputCol='text',outputCol='words').transform(tiny))
    model = NaiveBayes().fit(features)
    result = model.transform(features).select('prediction').collect()
    assert len(result)==4
    report = {'python':sys.version,'platform':platform.platform(),'spark':spark.version,
              'java':spark.sparkContext._jvm.java.lang.System.getProperty('java.version'),
              'training_rows_read':count,'spark_sql_and_toy_training':'passed',
              'note':'仅环境冒烟测试，不是正式情感模型或效果指标。'}
    (root/'reports/environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
finally:
    spark.stop()
