# 中文短文本情感识别

## 社群表达与句子关系语料

新增公开研究语料包包含 COLD、HED-COLD、ToxiCN、OCNLI，共 115,977 条来源记录和 537 个候选词条。
下载与分类目录、来源许可、校验值及使用方法见 [语料包说明](data/community_corpora/README.md)。
运行 `python scripts/restore_community_corpora.py` 离线恢复原始数据和可发布的分类视图，
再运行 `python scripts/prepare_community_corpora.py` 重建本地完整分类与标注模板。
情感标签未自动补标，现有模型不因下载语料而改变。

目标：基于 PySpark 的十万级中文短文本情感识别与误判分析。现场输入新文本，展示正负向预测；另用独立测试集评价效果。

项目从数据清洗到字符扩展、中文分词实验中遇到的问题和实际结论，集中记录在 [项目过程记录](reports/development_log.md)。

## 公开项目反馈采集试点

`scripts/crawl_public_issues.py` 使用 [GitHub 公开 Issues API](https://docs.github.com/en/rest/issues/issues) 采集 Element Plus 的最近公开问题标题，作为数据获取环节演示。默认最多请求 5 页、每页 100 条，顺序请求并间隔至少 1 秒；排除 Pull Request、明显非中文或过短标题，遮盖标题里的邮箱、手机号和 `@` 账号。不获取或保存作者资料。运行：

```powershell
.\.venv\Scripts\python.exe scripts\crawl_public_issues.py
```

2026-10-13 的试点共读取 500 条 API 记录，其中 384 条是 Pull Request；116 条 Issue 中按语言、长度和重复标题过滤后保留 65 条中文标题。数量与文件哈希见 `reports/crawler_pilot.json`。文本和标注表仅保存在 Git 忽略的 `data/processed/github_issue_feedback.jsonl` 和 `data/processed/github_issue_labels.csv`。后者已做标题级初标（58 条 `negative`、7 条 `unclear`，均待人工复核），没有可靠正例；“报告问题”不自动等于负面情感。**这些反馈没有用于训练、测试或网页预测。**实时 API 内容会变化，重复运行可能得到不同样本；已填写人工标注的 CSV 不会被覆盖。使用时遵守 [GitHub API 速率限制](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)和相关内容使用规则。

## Steam 中文评测采集试点

B 站[用户协议](https://www.bilibili.com/blackboard/protocal/licence.html)限制未经书面许可的自动脚本采集，因此本项目改用 [Steam 官方公开评测接口](https://partner.steamgames.com/doc/webapi/IUserReviewsService)做真实中文评论试点。源码保留在 `scripts/crawl_steam_reviews.py`，默认采集《赛博朋克 2077》最近的简体中文“推荐”和“不推荐”评测，各最多两页、每页 100 条。运行：

```powershell
.\.venv\Scripts\python.exe scripts\crawl_steam_reviews.py
```

2026-10-13 试点读取 399 条评测，过滤过长、过短及中文不足的文本后保留 193 条（推荐 62、不推荐 131），仅在本机 `data/processed/steam_review_pilot.jsonl` 保存评测 ID、游戏 ID、文本、时间和弱标签，不保存作者或 SteamID；统计及文件哈希见 `reports/steam_review_pilot.json`。基础规则遮盖链接、邮箱、手机号和 `@` 名称，但不能保证识别所有个人信息。**“推荐/不推荐”是整篇评测的用户选择，不等于每个句子的准确情感标签；短评筛选也改变了正负比例。**目前未合并到训练、验证、测试集或网页模型，后续应先抽样复核标签和来源偏差。

### 扩大到多游戏评论

`scripts/crawl_steam_reviews_bulk.py` 复用同一官方接口，覆盖《赛博朋克 2077》《黑神话：悟空》《博德之门 3》《泰拉瑞亚》《星露谷物语》《Dota 2》《Counter-Strike 2》《PUBG》《怪物猎人：世界》《GTA V 增强版》十款游戏，按每款游戏的推荐／不推荐分别请求。默认上限每类 10 页，适合小规模试跑；完整十万级采集使用下列命令。逐页检查点保存在本机 `data/processed/steam_bulk_pages/`；连接中断后以相同参数重跑，已保存的页面不会重抓。临时网络错误最多重试两次，遇到接口限流则停止。

```powershell
.\.venv\Scripts\python.exe scripts\crawl_steam_reviews_bulk.py --pages-per-label 120 --delay 1
```

2026-10-13 的初版采集得到 5,267 条；本轮沿用检查点补采至 **2,170 页、216,355 条原始评测**。按 8—300 字、至少 4 个汉字等规则筛选，再按评测 ID 和正文做全局去重、排除同文冲突标签，最终得到 **102,902 条**（推荐 39,694、不推荐 63,208）。部分类别不到 120 页即返回空页，因此实际页数小于理论上限。完整的逐游戏数量、处理阶段计数、查询参数和输出 SHA256 见 `reports/steam_reviews_bulk.json`；清洗文本（约 24 MB）仅在 Git 忽略的 `data/processed/steam_reviews_bulk.jsonl`。文本中邮箱、手机号、链接和 `@` 名称做基础遮盖，仍应人工检查敏感信息。**推荐选择只是弱标签，且按类别主动采样、只保留短评会改变原始比例。**

### Steam 弱标签模型训练和留出测试

按[固定协议](reports/steam_training_protocol.md)将这 102,902 条评论按游戏和标签分层，分为训练 82,313、验证 10,282、测试 10,307 条；ID 与正文无跨集重复。使用字符 1—2 gram TF-IDF 和 SGD 逻辑回归，在验证前固定超参数，不依据测试集调参。独立留出测试的准确率 **90.57%**、Macro-F1 **89.97%**，完整混淆矩阵及逐游戏指标见[结果文件](reports/steam_training_results.json)，划分哈希见[分割记录](reports/steam_split_report.json)。这评价的是“推荐／不推荐”弱标签预测，**不等于人工情感标注准确率**。模型在 `models/steam_reviews.joblib`；当前网页仍使用原日常评论模型，避免把游戏评论模型误作为通用情感模型。原始和分割后的评论仍仅在本地，不随 Git 分发。

本机复现（需先有同一 SHA-256 的清洗数据）：

```powershell
.\.venv\Scripts\python.exe scripts\train_steam_reviews.py
```

脚本会拒绝覆盖已有模型和正式结果；复现应在新副本中运行，防止无意间重做测试。

前一轮[否定、转折特征实验与数据处理记录](reports/text_structure_process.md)按阶段记录输入校验、清洗、候选提取、向量化、训练及误判对照。使用原有 88,814 条训练记录和 10,908 条验证记录，三组验证准确率为 77.6311%、77.5211%、77.6494%；整体尚无明确优势，该轮没有替换网页模型。复现：`python scripts/train_structure_experiment.py`；测试：`python -m unittest discover -s tests -v`。

后续完成[日常评论多来源优化](reports/daily_review_process.md)：使用微博、购物和外卖共 148,216 条训练记录，五组方案各训练三个固定种子，按三来源验证 Macro-F1 等权平均选择“直接混合＋结构 v2”。在锁定模型后的 7,429 条外部保留测试中，准确率为 **91.68%**，同轮微博单来源对照为 **80.05%**；微博验证表现略有退步，部分否定／转折诊断句仍然失败。报告记录了清洗、划分、来源权重、词内误命中处理、计算耗时和三张对照图，人工情感标注暂缓，社群黑话不在该实验范围。

网页已同步使用这轮选定的“直接混合＋结构 v2”（固定种子 `20261010`）。演示模型保存为 `models/daily_review.joblib`，版本与 SHA256 记录在 `models/daily_review.json`；与锁定的实验模型逐字节一致。网页与命令行共用预测入口：`python scripts/predict_daily_reviews.py "今天不开心。" "我一点也不喜欢原神"`。实验模型及逐条原文仍保存在本地 Git 忽略目录；后续训练不会自动改变演示版本。训练依赖使用 `requirements-demo.txt`，生成报告图表另安装 `requirements-report.txt`。新外部测试已评价，后续调参不能反复用它宣称独立提升。

## 给同学使用：双击启动（Windows）

把整个 `sentiment_project` 文件夹发给同学，或让同学下载仓库的 ZIP 并解压。电脑安装 **Python 3.12** 后，双击 [start_demo.bat](start_demo.bat)；首次启动需要联网安装演示依赖，之后会自动打开浏览器。输入一句中文短文本即可看到情感判断。启动窗口需要保持打开；关闭窗口就会停止服务。无需 VS Code、GitHub 账号、原始数据或 Spark。

每台电脑各自运行，页面地址为 `http://127.0.0.1:8765`，只能在运行程序的那台电脑上访问。分发项目时保留 `models/daily_review.joblib`、`models/daily_review.json` 及项目中的 Python 源文件。若端口被占用，先关闭之前启动的演示窗口。更新模型后需要重新启动服务，再刷新页面。

## 命令行运行

本地演示使用固定的 `models/daily_review.joblib`，无需原始语料或实验目录。使用 Python 3.12，在本项目目录运行：

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-demo.txt
./.venv/Scripts/python.exe app.py
```

浏览器打开 `http://127.0.0.1:8765`。运行时不需要 GitHub，也不需要下载原始数据。模型文件仅加载此仓库提交的版本；不要加载来源不明的 `joblib` 文件。

若要复现实验，先按下文提供的固定数据版本下载 `data/raw/weibo_senti_100k.csv`，再依次运行 `scripts/audit_data.py`、`scripts/prepare_splits.py`、`scripts/train_baseline.py` 和 `scripts/diagnose.py`。Spark数据分析还需完整Java 21，并按 `start_spark.ps1` 的路径约定准备 `runtime/jdk-21.0.12.1+1`。脚本在Windows上使用临时盘符绕开中文路径启动问题。

## 中文分词实验分支

`word_token_branch` 用 `jieba` 把中文切成词，比较“词及相邻词”与“字符 1—2 字 + 词及相邻词”两种特征。安装实验依赖后运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt
.\.venv\Scripts\python.exe scripts\train_word_baseline.py
```

结果写入 `reports/word_baseline_results.json`，模型分别保存为 `models/word_baseline.joblib` 和 `models/hybrid_baseline.joblib`。实验不会覆盖初版 `models/baseline.joblib`，也不使用测试集选模型。原字符模型验证集准确率 77.66%，纯分词模型 76.33%，组合模型 77.86%；组合方案只提升 0.20 个百分点，固定诊断句正确数仍是 11/12，因此该轮没有替换网页默认模型。分词和词组可能捕捉否定表达，但对分词错误、口语与新词仍敏感。

## 外部评论数据：最初清洗记录与后续训练

增加 [购物评论 10 类](https://github.com/SophonPlus/ChineseNlpCorpus/tree/a0b781d72158bde242c0626c9e354d2b328865ff/datasets/online_shopping_10_cats)和[外卖评论](https://github.com/SophonPlus/ChineseNlpCorpus/tree/a0b781d72158bde242c0626c9e354d2b328865ff/datasets/waimai_10k)。原始 62,774 + 11,987 条；经过标签／空文本检查、空白规范化、冲突与重复文本处理，并排除与现有微博训练／验证／测试集正文指纹重合的记录后，得到 **74,274 条**。清洗结果在本机 `data/processed/external_clean.jsonl`，数量、标签、类别和文件哈希见 `reports/external_data_audit.json`。最初清洗阶段没有用于训练；后续日常评论实验隔离 20 条跨来源共有文本，另外固定划分为训练 59,402、验证 7,423、保留测试 7,429 条。原微博划分保持不变，详见上面的日常评论多来源优化记录。

在项目目录复现下载和清洗：

```powershell
.\.venv\Scripts\python.exe scripts\fetch_external_data.py
.\.venv\Scripts\python.exe scripts\clean_external_data.py
```

原始及清洗后的文本都在 Git 忽略目录，仓库只提交下载／清洗脚本与统计报告。数据来源仓库未明确说明原始标注流程和再分发许可；使用时应注明来源与这一限制。

## Spark 多来源统计与规模实验

在原有微博划分和外部评论清洗文件都已准备好的电脑上，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_spark.ps1 -Script scripts/spark_all_sources.py
powershell -ExecutionPolicy Bypass -File .\start_spark.ps1 -Script scripts/benchmark_spark_pipeline.py
```

第一项使用单机 Spark `local[4]` 处理 **184,923 条**清洗记录，按来源、标签、类别和数据划分统计，结果见 `reports/spark_all_sources.json`。其中微博 110,649 条，外部评论 74,274 条；20 条外部文本同时来自购物和外卖源，在来源表中单列为 `multiple_external_sources`，总数没有重复计算。微博全部含方括号表情，购物评论仅 15 条含此类表情，外卖评论为 0 条。

第二项在 1 万、5 万、全量三个规模上执行同一 Spark JSONL 读取、正则清洗、长度计算和分组统计任务，各重复 3 次。实测中位数分别为 **0.523、0.555、0.858 秒**，见 `reports/spark_scale_benchmark.json`。计时不含输入文件准备和 Spark JVM 启动；这是本机热缓存下的处理耗时，不代表集群加速，也未与 Python 做速度优劣比较。两项都不训练或更换情感模型。

## 第一阶段：数据审核

- 数据说明：https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb
- 下载镜像：https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k
- 固定下载版本：3359c282d381367f470235cb071d122e0f1a4a5a
- 文件 SHA256：2d0a49e27de5295a84893e07816e131c0d116ecbbcbb624418d4e10bebf97a1b
- 下载日期：2026-10-10。
- 原始数据保留在 data/raw；清洗副本在 data/processed。
- 复现检查：在项目目录运行 `python scripts/audit_data.py`，结果写入 reports/data_audit.json。
- 清洗规则：去除文本 BOM、统一空白、删除无效数据；同一规范化文本标签冲突时整组排除；其余同文本去重。保留表情，供后续对照实验。

数据整理者声明原始作者和来源不详，标注方法与再分发许可尚未确认。当前用于本地课程实验，不将原始数据打包发布。结论只能代表该数据上的实验效果。需人工抽查标签及近似重复，不能将其视为高质量人工金标准。

## 环境检查

已检测：Windows，20个逻辑处理器，约31.7 GiB内存；可用的应用内置 Python 3.12.14 和 pandas。未发现 PATH 中可用的 Java，内置 Python 中尚无 pyspark、scikit-learn、streamlit。还未安装项目依赖，也未训练模型。

## 后续顺序

1. 检查近似重复，制作人工抽查清单；固定训练、验证、测试划分。
2. 建立项目独立环境，选择兼容 Java/Python/Spark 版本并验证本地运行。
3. 训练朴素贝叶斯和逻辑回归；只在训练集拟合词表及权重。
4. 在验证集选方案，最终测试集留作一次正式评价。
5. 对比有无表情特征；增加否定、转折和反讽的独立诊断样例。
6. 建立交互页面，再整理真实实验图表、报告和答辩材料。

模型分数不直接解释为正确概率；不确定不是第三个中性类别。数据规模实验使用同一测试集，仅改变训练子集。

## 第二阶段进展

已执行 `scripts/prepare_splits.py`：按正文指纹将格式/表情变体分组，固定分配至训练、验证、测试集合。排除正文指纹不足4字符的6,269条，最终使用110,649条：训练88,814、验证10,908、测试10,927。各集合的正文指纹无交集；重复执行得到相同文件哈希。详细规则及局限见 `reports/stage2_notes.md`，数量与哈希见 `reports/split_report.json`。

PyCharm自带的Java缺少Spark需要的模块，现已在项目`runtime/`准备完整Temurin 21。项目 `.venv` 使用 Python 3.12；目标依赖锁定于 requirements.txt。`scripts/smoke_spark.py` 已读取88,814条训练记录，并运行4条自建样例的小型MLlib模型，证明Spark可启动和计算。该检查不是正式情感模型训练。

Windows的工作目录含中文，Spark批处理启动会遇到路径编码问题。使用`powershell -ExecutionPolicy Bypass -File ./start_spark.ps1`创建临时短路径并运行成功的数据分析脚本。换电脑后需安装兼容Python和Java并重建`.venv`，不能直接复制虚拟环境；当前Python基础运行时来自本机应用内置环境。

## 第三阶段：可运行基线

已训练并保存去表情的字符1—2元词组TF-IDF + SGD逻辑回归模型。脚本 `scripts/train_baseline.py` 比较两种算法与“保留/去掉表情”，在验证集确定现场模型，再对独立测试集做一次最终评价。实际数值见 `reports/baseline_results.json`。此基线用 scikit-learn 实现，不把它称作Spark模型。

去表情现场模型在独立测试集准确率约77.29%。保留表情时，验证集约98.99%；这一差距显示数据标签与表情强相关，不代表模型已理解各种中文表达。人工诊断显示“今天不开心”被误判成正向；见 `reports/diagnostic_results.json`。诊断句只帮助解释错误，不计入正式准确率。

本地演示：在项目目录运行 `.venv/Scripts/python.exe app.py`，浏览器打开 `http://127.0.0.1:8765`。演示界面限制1—500字、只绑定本机。模型内部的决策分数没有校准为概率。

PySpark 4.0.1官方安装包已通过PyPI公布的SHA256校验。`scripts/spark_data_analysis.py`已使用Spark DataFrame读取、正则清洗、计算长度和按标签分组统计全部训练数据，结果见`reports/spark_data_analysis.json`。训练及现场推理模型由scikit-learn实现。尝试在此Windows环境中以Spark MLlib训练完整模型时，Hadoop因缺少winutils.exe无法写入训练临时文件，故不将它写成已完成的Spark训练。

训练数据每条都有方括号表情。保留表情的模型验证集准确率98.99%，去掉表情的模型验证集77.66%、独立测试集77.29%。本项目将此作为“表情依赖”问题重点分析。数值以`reports/baseline_results.json`为准。
