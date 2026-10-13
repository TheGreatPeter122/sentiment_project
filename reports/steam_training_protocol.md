# Steam 短评训练与正式测试协议

- 输入固定为 `data/processed/steam_reviews_bulk.jsonl`，以采集报告中的 SHA-256 校验；标签是 Steam 的“推荐/不推荐”，属于弱标签。
- 按 `app_id × weak_label` 分层，用 `steam-review-20261013-v1 + review_id` 的 SHA-256 排序，分为 80% 训练、10% 验证、约 10% 测试。保存分割文件和摘要，复跑不得改动已有分割。
- 预先固定模型：字符 1–2 gram TF-IDF（`min_df=3`, `max_features=120000`）和 `SGDClassifier(loss=log_loss, alpha=1e-5, max_iter=50, tol=1e-3, random_state=20261013)`。只在训练集拟合；验证集只作诊断，不据此调整模型。保存模型后，测试集只评估一次。
- 指标包括准确率、Macro-F1、两类 F1、混淆矩阵及各游戏指标。测试数据与训练/验证集的评论 ID 和文本均不得重叠。
- 这些数字衡量对未见过的 Steam 评论“推荐/不推荐”记录的预测能力，不等于人工标注的通用中文情感准确率；抽样和短文本过滤也不代表 Steam 的自然评论分布。
