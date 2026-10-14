> 文档状态：历史阶段记录，数字和“当前模型”描述仅适用于该阶段。最新说明见 [文档导航](../guides/README.md)。

# 公开语料下载与分类说明

整理日期：2026-10-11（Asia/Singapore）。文件均保存在本项目本地。

## 已下载的资源

| 数据源 | 本次实际整理量 | 内容 | 上游许可声明 |
| --- | ---: | --- | --- |
| [ToxiCN](https://github.com/DUT-lujunyu/ToxiCN) | 12,011 条 | 社群评论、一般冒犯、显式/隐式仇恨、转述 | CC BY-NC-ND 4.0；科研、非商业 |
| ToxiCN lexicon | 537 个词条 | 5 类词表，原值为模型 token ID，不是释义 | 同 ToxiCN |
| [HED-COLD](https://github.com/sjie320/HED_COLDataset) | 10,000 条 | 谐音和句式改写，保留对应 COLD 原文 | README 声明 MIT |
| [COLD](https://github.com/thu-coai/COLDataset) | 37,480 条 | 冒犯/非冒犯，测试集另含攻击个人、攻击群体、反偏见等细分类 | Apache-2.0 |
| [OCNLI](https://github.com/CLUEbenchmark/OCNLI) | 56,486 个句对 | 蕴含、矛盾、中立；部分无共识、测试无标签 | CC BY-NC 2.0 |

记录合计 115,977 条。该数是来源记录数，不代表互不重复的独立文本数；HED-COLD 来源于 COLD。
原始工作目录和本地派生文件位于 Git 已忽略的 data/raw、data/processed 目录。
可发布归档位于 `data/community_corpora/archives`，来源、许可与成员校验值见该目录的 README、NOTICE 和 manifest.json。
ToxiCN 仅发布未改动的原始文件；其分类和组合文件在本地重建。不要将所有上游数据视为具有相同发布许可。
上游完整仓库、README 和可获得的许可证均保留在原始下载目录；没有运行上游程序。

## 目录

相对于项目根目录 `E:\HJR\Work\Project\SentimentProject`：

```text
data/raw/community_corpora/
  toxicn/                 原始仓库，含 CSV、官方 train/test JSON、词典
  hed_cold/               原始数据与原文对应表
  cold/                   原始 train/dev/test CSV
  ocnli/                  原始句对与各规模训练子集

data/processed/community_corpora/
  by_source/              按来源和官方文件划分，标准 JSONL
    toxicn/{train,test}.jsonl
    hed_cold/{train,dev,test}.jsonl
    cold/{train,dev,test}.jsonl
    ocnli/{train,dev,test}.jsonl
  by_category/            按原始标注或候选文本特征分类
    toxicn/               non_toxic、general_offensive、hate_explicit、hate_implicit、hate_reporting
    hed_cold/             offensive_variant、non_offensive_variant
    cold/                 offensive、non_offensive、attack_individual、attack_group、anti_bias、non_offensive_other
    ocnli/                entailment、neutral、contradiction、no_consensus、unlabeled_test
                          negation_candidates、double_negation_candidates、contrast_candidates
  lexicon/toxicn_terms.jsonl
  annotation/
    slang_dictionary_to_annotate.csv       537 个待补释义的词条
    community_examples_to_annotate.csv     540 条供人工标注的开发样例
  combined_offensive/
    train.jsonl                           37,010 条
    dev.jsonl                              8,113 条
    test.jsonl                            14,304 条
    needs_label_review.jsonl                  64 条，25 组同文异标签样本及其同源关联记录

scripts/restore_community_corpora.py         离线校验和恢复发布包
scripts/prepare_community_corpora.py         整理脚本，使用 Python 标准库
reports/community_corpora_report.json       来源版本、SHA256、数量和检查结果
```

by_source 和 by_category 是同一批数据的不同视图，不能相加计算总量，也不要全部拼接训练。
OCNLI 的三个候选文件可互相重叠；它们按标记词筛选，不能视为已标注正确语义的专门否定数据集。

## 你负责的社群表达

1. 首先填写 `annotation/slang_dictionary_to_annotate.csv`。
   `meaning` 填写释义，`community` 填适用社群，`usage_context` 填上下文，
   `normal_usage_counterexample` 填容易误伤的正常用法。
2. 使用 `by_category/toxicn/hate_implicit.jsonl`（1,995 条）观察隐含表达，
   使用 `general_offensive.jsonl`（816 条）观察一般攻击性表达。
3. 使用 `by_category/hed_cold` 查看变体；每条都带 `original_text` 和 COLD 来源记录。
4. 在 540 条开发样例标注表中补充情感标签、词语释义、必要上下文和两人的标注。
   这些样例从组合训练集合分层抽取并按同源组去重，不能冒充独立测试集。
5. 这些数据覆盖的是研究数据集中的表达，不保证包含你提到的近期圈内黑话。
   自己收集的新用例另建文件，记录来源、日期、社群和含义确认情况。

## 同学负责的句子关系

OCNLI 主训练集实际为 50,486 个句对；dev 3,000，test 3,000。
train.3k、train.10k、train.30k 是嵌套子集，已下载保留，但未重复并入整理结果。
99 个无共识标签 `-` 和 3,000 个无标签测试句对已分开。

可以先看 `by_category/ocnli/negation_candidates.jsonl`（25,747 条）、
`double_negation_candidates.jsonl`（170 条）和 `contrast_candidates.jsonl`（4,065 条）。
这些筛选数只统计具有有效 NLI 标签的样本。命中“不”“没”等不代表确实构成需要处理的否定关系，需人工复核。
NLI 标签是句对关系，不是正负情感，不能直接填入现有情感模型的 label 列。

## 标签与划分规则

- 所有 `sentiment_label` 均为空，未自动生成情感金标准。
- `offensive_label=1` 表示冒犯，`0` 表示非冒犯；你们现有基线的 label=1 表示正向，两者不可混用。
- ToxiCN 官方 train/test JSON 与 CSV 的 12,011 条记录已完整对应。
- COLD 源行编号存在重复，标准记录 ID 使用文件划分和行序号，原始编号另行保存。
- COLD dev.csv 中有 3,215 行的内部 split 字段写作 train。整理以所在文件为 official_split，
  同时在 source_declared_split 保留原字段，不自行把它们放回训练集。
- HED-COLD 10,000 条记录均已用主题、原编号、原文和标签定位对应 COLD 记录。
  其中 2,021 条 HED 测试记录的原文在 COLD 训练文件；完整交叉表见 JSON 报告。
- 原始和分类视图保留各来源自己的划分。需要混合 COLD、HED-COLD、ToxiCN 时，使用 combined_offensive。
- 组合划分连接已知原文/变体和规范化后相同文本；整个组采用最严格的已有划分：test > dev > train。
  因此官方测试样本不会被移入组合训练集；已有训练样本可能因同源测试关联被移到测试集合。
  总计 5,216 条记录的组合划分与其来源文件划分不同。
- 发现 25 组规范化同文异冒犯标签，关联的 64 条记录单独待核对，未放入组合训练/验证/测试文件。
- 组合三个集合的已知同源组交集为 0。该检查不能覆盖所有同义改写或语义近重复。
- 组合划分是本地派生实验划分，不能宣称为任一数据集的官方基准成绩。

## 来源与可复现性

完整下载版本、每个读取数据文件的 SHA256 和字节数见 `community_corpora_report.json`。
恢复脚本会拒绝覆盖不同内容的已有文件；整理脚本会保留已有人工标注 CSV。

```powershell
cd E:\HJR\Work\Project\SentimentProject
.\.venv\Scripts\python.exe scripts\restore_community_corpora.py
.\.venv\Scripts\python.exe scripts\prepare_community_corpora.py
```

此次下载和分类没有重训练模型，演示程序仍使用原 baseline.joblib。
