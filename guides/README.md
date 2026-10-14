# 文档导航

当前说明按“使用 → 数据 → 模型 → 维护”阅读。`docs/` 专门用于网站发布，说明集中在本目录。旧报告保留原路径和实测数字，通过导航与阶段标识区分历史状态（已锁定的协议保持原字节），避免与当前三模型功能混淆。

## 当前指南

- [使用网页](usage.md)
- [数据获取、清洗和 Spark 对照](data_pipeline.md)
- [三模型切分与对比](models.md)
- [复现与发布](maintenance.md)
- [项目入口](../README.md)
- [研究语料说明](../data/community_corpora/README.md)和[来源许可](../data/community_corpora/NOTICE.md)：未并入训练。
- [网页第三方组件](../docs/THIRD_PARTY_NOTICES.md)

## 阶段报告与证据索引

以下是全部报告说明；阶段性结论请结合日期和当时的数据规模阅读，不直接代替当前结果。

- [reports/community_corpora_guide.md](../reports/community_corpora_guide.md)
- [reports/daily_review_process.md](../reports/daily_review_process.md)
- [reports/daily_review_protocol.md](../reports/daily_review_protocol.md)
- [reports/daily_review_runs/20261011_133237/PROCESS.md](../reports/daily_review_runs/20261011_133237/PROCESS.md)
- [reports/development_log.md](../reports/development_log.md)
- [reports/README_before_model_selector.md](../reports/README_before_model_selector.md)
- [reports/results_summary.md](../reports/results_summary.md)
- [reports/selectable_models_protocol.md](../reports/selectable_models_protocol.md)
- [reports/stage2_notes.md](../reports/stage2_notes.md)
- [reports/steam_audit_review.md](../reports/steam_audit_review.md)
- [reports/steam_spark_process.md](../reports/steam_spark_process.md)
- [reports/steam_training_protocol.md](../reports/steam_training_protocol.md)
- [reports/text_structure_process.md](../reports/text_structure_process.md)
- [reports/text_structure_protocol.md](../reports/text_structure_protocol.md)
- [reports/unified_model_process.md](../reports/unified_model_process.md)
- [reports/unified_model_protocol.md](../reports/unified_model_protocol.md)

本轮完整指标与一致性结果分别在 `reports/selectable_models_results.json`、`reports/selectable_models_verification.json`。原始报告 JSON 不改写，保留证据。
