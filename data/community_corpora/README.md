# 社群表达与句子关系语料包

用途：非商业课程研究。115,977 条来源记录和 537 个候选词条，不代表同等数量的独立文本或人工情感标签。

本目录将原始数据和允许发布的分类视图按来源压缩保存，总计约 40.2 MB。
下载仓库后，无须联网或安装第三方库即可恢复和重建；建议使用项目约定的 Python 3.12。

```powershell
python scripts/restore_community_corpora.py
python scripts/prepare_community_corpora.py
```

第一条命令检查归档及成员 SHA256，并解压到 Git 忽略的 `data/raw/community_corpora` 和
`data/processed/community_corpora`。已存在且内容不同的文件会被保留并报错。
第二条命令从原始数据重建所有分类、组合划分和标注模板；已有人工标注 CSV 会保留。

仅检查发布包而不解压：

```powershell
python scripts/restore_community_corpora.py --verify-only
```

| 文件 | 内容 | 记录数 |
| --- | --- | ---: |
| archives/toxicn-raw.zip | 未改动的 ToxiCN CSV、官方划分、5 个原始词表和 README | 12,011；537 个词条 |
| archives/cold-raw.zip | COLD 原始 CSV、README、Apache 许可证 | 37,480 |
| archives/hed_cold-raw.zip | HED-COLD CSV、原文对应表、README | 10,000 |
| archives/ocnli-raw.zip | OCNLI 最大训练集、3 个嵌套子集、验证/无标签测试文件、README | 56,486，未重复计入子集 |
| archives/cold-classified.zip | COLD 标准 JSONL、来源划分和类别视图 | 同一批 37,480 条的多个视图 |
| archives/hed_cold-classified.zip | 标准 JSONL、攻击/非攻击变体及其原文关联 | 同一批 10,000 条的多个视图 |
| archives/ocnli-classified.zip | 标准句对、关系类别和否定/转折候选 | 同一批 56,486 条的多个视图 |

ToxiCN 的许可包含 NoDerivatives 条件。本仓库只发布字节未改动的上游原始文件；
其分类 JSONL、含其文本的组合划分和标注模板由脚本在本地生成，不包含在发布归档中。
ToxiCN 的词表值是模型 token ID，没有提供词条释义，释义需要人工填写。

文件版本、SHA256 和归档成员清单见 [manifest.json](manifest.json)。
分类数量、数据质量观察、原文/变体交叉划分见 [整理报告](../../reports/community_corpora_report.json)。
两人的工作入口及标签规则见 [使用指南](../../reports/community_corpora_guide.md)。
来源、作者和使用条件见 [NOTICE.md](NOTICE.md)。

所有 `sentiment_label` 都留空；冒犯标签和自然语言推理标签不可直接替代情感标签。
没有训练新模型，也不将同源的 COLD/HED-COLD 样本当作互不相关的独立数据。
