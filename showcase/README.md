# 遥感演示界面

`remote_sensing_app.py` 是面向 Terrabox 答辩的轻量演示入口，包含：

- **Agent 任务执行**：输入中文或英文遥感任务，上传图片、GeoTIFF、GeoJSON、GPKG、SHP、KML 等文件，实时展示工具调用和结果。
- **工具广场**：从 Registry 选择任意工具，查看 Schema，填写 JSON 参数并执行；上传文件会同时提供 `files`、`data_files` 和兼容旧工具的图片字段。

启动：

```bash
PYTHONPATH=src:. python showcase/remote_sensing_app.py --port 7861
```

Agent 页面默认使用 DeepSeek，也可切换 LongCat。对应 API key 放在本地 `agent_config.yaml` 或环境变量中，不要提交到 Git。Agent 后端接口 `/v1/gui/agent/chat` 和 `/v1/gui/agent/chat/stream` 新增表单字段 `language`（`zh`/`en`/`auto`）和 `llm_provider`（`deepseek`/`longcat`/`local`）。
