# 中文短文本情感分析

一个大数据导论课程项目：汇集公开中文语料和自行采集的 Steam 评测，用 Python 清洗、PySpark 统计与规模对照，再训练情感分类模型，在网页中输入新文本进行预测。

## 先使用

[在线分析器](https://thegreatpeter122.github.io/sentiment_project/)由 GitHub Pages 发布 `main` 分支的 `docs/`。本地修改只有提交、推送并完成 Pages 发布后才会出现在这个地址。

本地无需 VS Code：双击 `docs/index.html`，选择模型，等待就绪后输入短评。完整 `docs/` 文件夹必须一起保留。也可以双击 `start_demo.bat`，或运行 `python -m http.server 8766 --directory docs` 后打开 `http://127.0.0.1:8766/`。

网页可切换：**字符 1—2 gram（默认）、jieba 分词、字符 1—4 gram**。三个模型都使用微博、购物、外卖、Steam 四来源数据，没有独立的 Steam 产品线。输入在浏览器本地计算。

## 按目的阅读

| 目的 | 文档 |
|---|---|
| 找到所有说明及历史记录 | [文档导航](guides/README.md) |
| 打开网页、切换模型、排查问题 | [使用说明](guides/usage.md) |
| 理解采集、清洗、数据规模、Spark | [数据处理方法](guides/data_pipeline.md) |
| 理解分词、算法、实测结果 | [三模型对比](guides/models.md) |
| 复现训练、导出网页、维护发布 | [维护与复现](guides/maintenance.md) |
| 梳理一路的实验和问题 | [项目过程](reports/development_log.md) |

## 当前规模与边界

Steam 原始采集 216,355 条，清洗后 102,902 条，其中 81,983 条进入训练。四来源合并训练集共 230,199 条。这几个数字分别属于原始采集、清洗语料和训练子集。

Spark 在本机 `local[4]` 做后续处理与统计；情感分类由 scikit-learn 训练。Steam 推荐状态是弱标签，100 条抽样仅完成 AI 辅助初审。测试集曾用于历史实验，本轮结果属于诊断对比。分数不是概率，不能用不同模型分数大小判断谁更准确。

## 目录

- `scripts/`：采集、清洗、划分、训练、导出及检查脚本。
- `models/`：Python 模型与版本哈希。`daily_review.joblib` 是默认模型，另有 `jieba.joblib`、`char4.joblib`。
- `docs/`：GitHub Pages 网页及浏览器模型参数，不是说明文档目录。
- `guides/`：当前使用和维护说明。
- `reports/`：可追溯实验记录、固定协议及机器可读结果。旧报告有历史标识。
- `data/`：来源信息、诊断例句和本地数据。原始评测、清洗文本、检查点不随 Git 发布。
- PPT 位于仓库外的上级 `PPT资源/` 文件夹。

更多社群研究语料见 [语料包说明](data/community_corpora/README.md)，这些语料尚未并入情感训练。
