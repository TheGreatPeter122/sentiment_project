# 100 条 Steam 短评抽样复核（AI 辅助初审）

从已经固定的 Steam **测试集**中，按 10 款游戏 × 推荐／不推荐两种弱标签各抽 5 条，共 100 条。`scripts/sample_steam_review_audit.py` 用评论 ID 的固定哈希抽样并核对测试集哈希；阅读文本时隐藏弱标签和模型结果。先按正文主旨标为“正向、负向、混合、不明确”，再由 `scripts/report_steam_review_audit.py` 解盲统计。逐条正文、初审标签、原弱标签、模型预测及人工复核空列只保存在本机 Git 忽略的 `data/processed/steam_audit_review.csv`，不上传评论原文。

| 初审类别 | 条数 |
|---|---:|
| 负向 | 48 |
| 正向 | 35 |
| 正负混合 | 10 |
| 不明确 | 7 |

83 条明确正／负向的评论中，Steam 推荐弱标签与 AI 初审一致 **77/83（92.77%）**；已训练模型与 AI 初审一致 **75/83（90.36%）**。6 条弱标签与正文主旨不一致，例如“看什么差评，快买啊”却选择“不推荐”，以及“不好玩，浪费青春”却选择“推荐”。另有 17 条混合或不明确文本，不宜强迫赋予单一情感标签。具体样本编号与汇总见 [`steam_audit_review.json`](steam_audit_review.json)；本地 CSV 的 `ai_note` 列对这些关键记录写了理由。

**这些是 AI 辅助的初步判断，不是独立人类标注金标准。** 抽样刻意平衡游戏和弱标签，不代表 Steam 自然评论比例；而且样本来自已经用于正式模型评价的测试集，所以这次只是误差与标签质量诊断，不能作为新的独立准确率，也不能据此反复调参后再宣称原测试集独立。课程报告应明确写成“AI 辅助抽样初审”；若要写“人工复核”，请组员在本地 CSV 的 `human_review`、`human_note` 列逐条确认，尤其是 6 条弱标签冲突和 17 条混合／不明确记录。

复现初审记录（需要本地测试集及模型；重新运行统计脚本会拒绝覆盖已填写的人工复核列）：

```powershell
.\.venv\Scripts\python.exe scripts\sample_steam_review_audit.py
.\.venv\Scripts\python.exe scripts\report_steam_review_audit.py
```
