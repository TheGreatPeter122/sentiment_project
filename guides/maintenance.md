# 维护与复现

命令都在 `sentiment_project` 目录执行。Windows 使用 `.\.venv\Scripts\python.exe`；示例以 `python` 简写，确保它指向安装依赖的环境。

## 环境与数据

安装 `requirements-demo.txt` 可预测三种模型，包含 jieba 0.42.1。运行实验另看 `requirements-experiments.txt`，Spark 另需 Java 与 PySpark。原始及划分文本默认未提交 Git，新克隆仓库不能凭模型文件直接重训。需要恢复本地原划分文件并满足各报告 SHA-256，不能随意重新随机划分替代。

## 训练与导出

1. 阅读 `reports/selectable_models_protocol.md`，确认数据及参数。
2. 运行 `python scripts/train_selectable_models.py`。结果已存在时脚本会拒绝覆盖，防止误重跑并混淆历史。需要新实验时先归档模型、结果和协议，创建新的实验记录。
3. 运行 `python scripts/export_selectable_models.py`，统一生成 `docs/model.json`、新增模型、词典、file 模式 JS 参数包和模型目录。旧入口 `export_pages_model.py` 也会转发到三模型导出。
4. 运行 `python scripts/verify_selectable_models.py`，需要 Node.js 以及本地固定划分。验证通过后再检查真实浏览器的三个模型、快速切换、加载失败重试和手机布局。
5. 运行 `python -m unittest discover -s tests`。旧实验测试如有失败，应判断是否与本轮更改有关，不要修改历史结果来凑通过。

仅改变 Python 模型不会自动改变网页。HTTP 与 file 模式文件必须一起导出。不要手改生成的哈希或把原始评论放进 `docs/`。

## 发布

审阅 `git diff`，提交源码、模型版本记录及完整网页文件，再推送 `main`。GitHub Pages 配置为该分支 `/docs`。等待部署完成，打开在线网址验证模型选择器和实际预测。Git 推送成功不等于 Pages 已部署。

此前 Git HTTPS 在 OpenSSL/代理组合下失败，可尝试一次性使用 `git -c http.sslBackend=schannel -c http.version=HTTP/1.1 push origin main`。不要关闭 TLS 证书校验。`127.0.0.1` 代理不可达时先确认代理程序是否运行。

## 数据流程入口

- 采集：`scripts/crawl_steam_reviews_bulk.py`。首次试跑用少量页，继续采集复用原参数和检查点。
- 清洗：采集脚本中的 `select_reviews`、`clean_text`、`build_dataset`。
- 原统一划分审计：`scripts/train_unified_reviews.py` 的 `prepare()`。
- Spark 对照：`powershell -ExecutionPolicy Bypass -File .\start_spark.ps1 -Script scripts/benchmark_steam_pipeline.py`。

[返回文档导航](README.md)
