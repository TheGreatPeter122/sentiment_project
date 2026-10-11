"""Build course-ready figures and narrative directly from a completed experiment."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'weibo_only': '微博单来源', 'mixed': '直接混合', 'balanced': '来源×标签平衡',
         'structure_v1': '混合＋结构 v1', 'structure_v2': '混合＋结构 v2'}


def main():
    r = json.loads((ROOT / 'reports/daily_review_results.json').read_text(encoding='utf-8'))
    run = ROOT / 'reports/daily_review_runs' / r['run_id']
    figures = run / 'figures'
    figures.mkdir(exist_ok=True)
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    arms = {a['name']: a for a in r['arms']}
    selected = r['selection']['selected_arm']

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(3)
    for offset, (name, color) in enumerate((('weibo_only', '#64748b'), ('mixed', '#38a3a5'), (selected, '#1d4ed8'))):
        v = arms[name]['summary']['validation']
        means = [v['source/' + d]['accuracy_mean'] * 100 for d in ('weibo', 'shopping', 'waimai')]
        stds = [v['source/' + d]['accuracy_std'] * 100 for d in ('weibo', 'shopping', 'waimai')]
        bars = ax.bar(x + (offset - 1) * .25, means, .23, color=color, label=NAMES[name], yerr=stds, capsize=3)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in means], padding=5, fontsize=10)
    ax.set(xticks=x, xticklabels=['微博（10,908 条）', '购物（6,239 条）', '外卖（1,184 条）'],
           ylim=(0, 105), ylabel='验证准确率（%）', title='多来源训练提高外部评论表现，微博表现略降')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.15)
    fig.text(.5, .015, '相同验证记录；柱高为三个种子的均值，误差线为种子间样本标准差，不是置信区间。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(figures / 'source_validation.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    keys = ('all', 'negation', 'contrast')
    x = np.arange(len(keys))
    for offset, (key, label, color) in enumerate((('weibo_control', '微博单来源对照', '#64748b'),
                                                ('selected', '锁定的新模型', '#1d4ed8'))):
        values = [r['external_holdout'][key][group]['accuracy'] * 100 for group in keys]
        bars = ax.bar(x + (offset - .5) * .32, values, .3, color=color, label=label)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=5)
    counts = [r['external_holdout']['selected'][k]['n'] for k in keys]
    ax.set(xticks=x, xticklabels=[f'{label}\n{n:,} 条' for label, n in zip(('全体', '否定候选', '转折候选'), counts)],
           ylim=(0, 105), ylabel='外部保留测试准确率（%）', title='锁定模型后评价购物和外卖保留测试数据')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.15)
    fig.text(.5, .015, '固定代表种子 20261010；子集为规则候选且可重叠；不包含原微博测试集。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(figures / 'external_holdout.png')
    plt.close(fig)

    split = r['provenance']['external_splits']['splits']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    base = np.zeros(3)
    for domain, label, color in (('weibo', '微博', '#64748b'), ('shopping', '购物', '#38a3a5'), ('waimai', '外卖', '#1d4ed8')):
        values = [r['training_and_validation']['train']['sources'].get(domain, 0),
                  r['training_and_validation']['validation']['sources'].get(domain, 0),
                  split['test']['sources'].get(domain, 0)]
        ax.bar(x, values, bottom=base, label=label, color=color)
        base += values
    for i, total in enumerate(base):
        ax.text(i, total + 2000, f'{int(total):,}', ha='center')
    ax.set(xticks=x, xticklabels=['训练', '验证', '外部保留测试'], ylabel='记录数',
           ylim=(0, base.max() * 1.13), title='按来源保留固定划分，验证和测试不参与拟合')
    ax.legend()
    ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    fig.savefig(figures / 'partition_sources.png')
    plt.close(fig)

    prefix = f'daily_review_runs/{r["run_id"]}'
    lines = [
        '# 日常评论优化与多来源数据处理记录', '',
        f'日期：2026-10-11（Asia/Singapore）。运行编号：`{r["run_id"]}`。', '',
        '**本轮完成多来源清洗核验、固定划分、来源权重对照、否定/转折规则优化、15 次训练和锁定后的外部测试评价。** 人工情感标注继续暂缓，社群黑话不在本轮范围。', '',
        '## 1. 数据从哪里来，如何处理', '',
        '微博沿用已有 88,814 条训练、10,908 条验证数据。购物和外卖来自 SophonPlus/ChineseNlpCorpus 固定版本 `a0b781d72158bde242c0626c9e354d2b328865ff`。', '',
        '两份外部原始数据共 74,761 条（购物 62,774、外卖 11,987）。校验原文件 SHA256 后，复用清洗脚本检查标签/空文本，统一空白，排除冲突、重复、过短指纹和与微博各划分重合的文本，得到 74,274 条；输出哈希与历史记录一致。各删减计数见 [原始清洗报告](external_data_audit.json)，不同层级的删减计数不可机械相加。', '',
        '其中 20 条同时来自购物和外卖，单独隔离，避免给共有文本随意指定来源或双重计数。剩余 74,254 条按来源×标签分层，以固定种子和 ID 哈希排序，划为训练/验证/保留测试。', '',
        '| 来源 | 训练 | 验证 | 新保留测试 |', '| --- | ---: | ---: | ---: |',
        '| 微博 | 88,814 | 10,908 | 原测试集未评价 |',
        '| 购物 | 49,919 | 6,239 | 6,242 |', '| 外卖 | 9,483 | 1,184 | 1,187 |',
        '| 本轮合计 | **148,216** | **18,331** | **7,429** |', '',
        '分组交集为 0；去重覆盖精确文本及既有正文指纹，仍无法排除全部语义近重复。新保留测试在选择记录锁定后才用于模型推理。来源标签保持 0=负向、1=正向，未人工逐条审定。', '',
        f'![各来源固定划分]({prefix}/figures/partition_sources.png)', '',
        '## 2. 从文本到特征', '',
        '保留原文，生成去表情正文、否定候选位置和后续片段、同小句多次否定候选、最后一个转折词前/后的文本。`start` 是从 0 开始的 **clean_text 偏移量**，去表情可能使其不同于原文偏移量。所有处理轨迹保存在本地忽略目录，便于按 ID 回查。', '',
        'v1 沿用前轮表面规则；v2 排除“特别、分别、别人、未来、无论、无数”等词内否定误命中和“但愿、不但、忙不过来”等转折误命中。排除候选不会改写正文或情感标签，局部片段最多 4 字，在标点/空白/有效转折处结束。', '',
        '| 验证候选 | v1 | v2 |', '| --- | ---: | ---: |',
    ]
    for key, label in (('negation', '否定'), ('contrast', '转折'), ('double_negation_candidate', '同小句多次否定')):
        lines.append(f'| {label} | {r["processing"]["v1"]["validation"]["candidate_counts"][key]:,} | {r["processing"]["v2"]["validation"]["candidate_counts"][key]:,} |')
    lines += ['', '候选数减少只说明规则筛选发生变化，不能直接解释为识别准确率提高。所有模型采用同一份 v2 掩码评价子集，使比较分母一致。', '',
        '## 3. 控制变量与训练', '',
        '先固定字符 1—2 元 TF-IDF 和 SGD 逻辑回归，比较微博单来源、直接混合、来源×标签平衡。平衡组使用训练样本权重：每个来源/标签格子的总权重相同，整体平均权重约为 1；不复制训练记录，也不使用验证/测试分布计算权重。', '',
        '阶段 A 选中直接混合，再比较基线、结构 v1、结构 v2。每组三个固定种子 20261010/11/12。词表、IDF、稀有否定特征筛选只拟合训练集。按三来源验证 Macro-F1 等权平均的三种子均值选方案，代表模型预先固定为 seed=20261010。', '',
        '| 方案 | 三来源等权 Macro-F1 均值 ± 种子标准差 | 全体验证准确率均值 |',
        '| --- | ---: | ---: |']
    for a in r['arms']:
        s = a['summary']
        lines.append(f'| {NAMES[a["name"]]} | {s["selection_mean"]:.6f} ± {s["selection_std"]:.6f} | {s["validation"]["all"]["accuracy_mean"]*100:.2f}% |')
    lines += ['', '最终按预定指标选中 **直接混合＋结构 v2**。三个种子共用同一份数据，标准差只反映本轮优化随机性的变化；v2 相对 v1 的差距很小，不能据此宣称稳定领先。', '',
        f'![三来源验证结果]({prefix}/figures/source_validation.png)', '',
        '微博验证准确率均值从同轮单来源的 77.58% 降到新模型的 76.79%；购物和外卖提升更明显。选择新模型是面向三来源日常评论的目标，并不表示它在所有来源都更好。', '',
        '| 同一验证子集 | 微博单来源 | 混合字符基线 | 混合＋结构 v2 |', '| --- | ---: | ---: | ---: |']
    for key, label in (('negation', '否定候选'), ('contrast', '转折候选')):
        values = [arms[name]['summary']['validation'][key]['accuracy_mean'] * 100 for name in ('weibo_only', 'mixed', selected)]
        lines.append(f'| {label} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |')
    lines += ['', '最大的收益来自训练数据组成变化；额外结构特征的增益较小。混合方案同时增加了训练条数和来源，不能把优势全部归因于多样性。', '',
        '## 4. 锁定方案后的保留测试结果', '',
        f'模型选择记录于 `{prefix}/selection_lock.json`，事件日志证明锁定先于保留测试推理。仅评价选定模型和同轮、同代表种子的微博单来源对照，测试后未再调参。', '',
        '| 外部保留测试 | 条数 | 微博单来源对照 | 新模型 |', '| --- | ---: | ---: | ---: |']
    for key, label in (('all', '全体'), ('source/shopping', '购物'), ('source/waimai', '外卖'), ('negation', '否定候选'), ('contrast', '转折候选')):
        old, new = r['external_holdout']['weibo_control'][key], r['external_holdout']['selected'][key]
        lines.append(f'| {label} | {new["n"]:,} | {old["accuracy"]*100:.2f}% | {new["accuracy"]*100:.2f}% |')
    changes = r['external_holdout']['changes']['all']
    lines += ['', f'相对单来源对照，新模型改对 {changes["corrected"]} 条、改错 {changes["regressed"]} 条、两者仍错 {changes["still_wrong"]} 条，净增 {changes["corrected"]-changes["regressed"]} 条正确。全体 Macro-F1 为 {r["external_holdout"]["selected"]["all"]["macro_f1"]:.4f}。', '',
        f'![外部保留测试]({prefix}/figures/external_holdout.png)', '',
        '这里的 91.68% 仅对应这份购物/外卖保留测试，不能与历史微博测试准确率直接比较或宣称在线整体达到同样水平。子集是规则候选且可重叠，原标签未人工核验；外卖转折子集仅 96 条，波动与场景局限更大。', '',
        '## 5. 已知用例和仍存在的问题', '',
        '新模型把“今天不开心”“我一点也不喜欢原神”判为负向。仍把“这次服务没有让我满意”“虽然包装很漂亮，但东西实在太难用了”判为正向；这些开发用例说明否定范围、转折和多个评价对象尚未可靠处理。', '',
        '诊断用例没有加入训练或用于选模型，也不计作独立准确率。具体分数是未经概率校准的决策值。本轮训练没有覆盖初版 `models/baseline.joblib`；网页当前启用的固定演示版本见 README，命令行与网页共用演示模型入口。', '',
        '## 6. 成本、环境与过程记录', '',
        f'本轮完整执行约 {r["elapsed_seconds"]:.1f} 秒，包括规则轨迹、五组×三种子训练、模型保存/重载和锁定测试评价。单机 Windows 11、Python 3.12.14、16 个逻辑处理器；本轮未进行 Spark/GPU 训练。模型核心包版本与前轮相同，详见结果 JSON。', '',
        '| 方案 | 特征维度 | 向量化秒数 | 三次分类器拟合秒数 |', '| --- | ---: | ---: | --- |']
    for a in r['arms']:
        lines.append(f'| {NAMES[a["name"]]} | {sum(a["dimensions"].values()):,} | {a["vectorization_seconds"]:.3f} | '+', '.join(f'{x["fit_seconds"]:.3f}' for x in a['runs'])+' |')
    lines += ['', '向量化时间包含训练拟合与验证转换，规则提取另计；模型序列化也有开销。所有计时为本机运行值，不作集群加速或不同工具速度优劣结论。', '',
        '## 7. 文件和使用方法', '',
        '- [训练前固定方案](daily_review_protocol.md)与[划分清单](daily_review_split_report.json)。',
        f'- [本次完整结果]({prefix}/results.json)与[事件日志]({prefix}/events.jsonl)。',
        f'- [本地逐条处理和预测目录](../{r["local_artifacts"]})：原文保存在 Git 忽略目录。',
        '- [训练脚本](../scripts/train_daily_reviews.py)、[规则与权重](../daily_review_features.py)。',
        '- [独立新模型测试入口](../scripts/predict_daily_reviews.py)。', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe scripts\\predict_daily_reviews.py "今天不开心。" "我一点也不喜欢原神"', '```', '',
        '复现训练会创建新的时间戳目录，并再次评价该已使用的外部测试；后续重跑不能再称为首次独立测试。', '',
        '```powershell', '.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v',
        '.\\.venv\\Scripts\\python.exe scripts\\train_daily_reviews.py --prepare-only',
        '.\\.venv\\Scripts\\python.exe scripts\\train_daily_reviews.py', '```', '',
        '生成报告图表需要安装 `requirements-report.txt`，运行 `python scripts/report_daily_reviews.py`。训练和推理仍使用 `requirements-demo.txt` 的核心依赖。', '',
        '以上工作可用于课程的多来源管理、去重与划分、文本特征处理、效果评价和问题挑战部分。下一轮如继续优化，应采用新的评价设计；本次保留测试已完成使用。', '']
    content = '\n'.join(lines)
    (ROOT / 'reports/daily_review_process.md').write_text(content, encoding='utf-8')
    snapshot = re.sub(r'\]\((?!https?://)([^)]+)\)', lambda m: '](../../' + m.group(1) + ')', content)
    (run / 'PROCESS.md').write_text(snapshot, encoding='utf-8')
    print(f'Wrote reports/daily_review_process.md and three figures in {figures.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
