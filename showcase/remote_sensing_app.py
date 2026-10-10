"""面向答辩的 Terrabox 遥感演示界面。

包含 Agent 任务执行和工具广场两个入口，不依赖灾害 SFT 样例数据。
启动：``python showcase/remote_sensing_app.py --port 7861``
"""
from __future__ import annotations

import json
import html
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from showcase.runner import run_sample
from showcase.tool_bridge import execute_tool, get_tool_schemas
from terrabox.core.utils.uploads import inspect_uploaded_files

OEA_CATALOG_PATH = ROOT / "data" / "oea_full_sft" / "tools_catalog.json"


def _oea_slugs() -> list[str]:
    if not OEA_CATALOG_PATH.exists():
        return []
    data = json.loads(OEA_CATALOG_PATH.read_text(encoding="utf-8"))
    return [str(row.get("slug")) for row in data if row.get("slug")]


def _message_tool_calls(message) -> list[dict]:
    calls = []
    for call in getattr(message, "tool_calls", []) or []:
        calls.append({"name": str(call.get("name", "")).replace("__", "."), "args": call.get("args") or {}})
    return calls


def _render_oea_result(result, allowed: list[str]) -> tuple[str, str, str]:
    blocks = [f"### OEA harness 完成（allowlist: {len(allowed)} tools，耗时 {result.elapsed:.1f}s）"]
    called: list[str] = []
    for message in result.messages:
        calls = _message_tool_calls(message)
        for call in calls:
            called.append(call["name"])
            blocks.append(f"### Tool call: `{html.escape(call['name'])}`\n```json\n{json.dumps(call['args'], ensure_ascii=False, indent=2)}\n```")
        if getattr(message, "__class__", type(message)).__name__ == "ToolMessage":
            content = str(getattr(message, "content", ""))
            blocks.append(f"**Tool observation:**\n```text\n{content[:4000]}\n```")
    return "\n\n".join(blocks), str(result.final or ""), ", ".join(called)


def run_oea_agent(prompt: str, files, provider: str, language: str, max_rounds: int):
    """Run the same eval-mode harness used by OEA trajectory experiments."""
    from terrabox.agent.config import load_config, reset_request_llm_provider, set_request_llm_provider
    from terrabox.agent.eval_modes import EvalModeContext, get_eval_mode_runner
    from terrabox.agent.llm import get_llm

    paths = [str(item) for item in (files or [])]
    if not (prompt or "").strip():
        yield "请输入遥感任务描述。", "", ""
        return
    token = set_request_llm_provider(provider)
    try:
        config = load_config()
        config.max_iterations = int(max_rounds)
        llm = get_llm(config)
        allowed = _oea_slugs()
        question = prompt.strip()
        if paths:
            question += "\n\n[Data files available]\n" + "\n".join(paths)
        question += "\n\n[Response language: English.]" if language == "en" else "\n\n[回答语言：中文。]"
        result = get_eval_mode_runner("standard").run(
            EvalModeContext(
                question=question,
                config=config,
                llm=llm,
                allowed_slugs=allowed,
                image_paths=paths,
                sequential_tool_turns=True,
                task_metadata={"images": paths, "data_files": paths, "available_tools": allowed},
                user=None,
                verbose=False,
            )
        )
        yield (*_render_oea_result(result, allowed),)
    except Exception as exc:
        yield f"**OEA harness 错误：** {html.escape(str(exc))}", "", ""
    finally:
        reset_request_llm_provider(token)


def _tool_choices() -> list[str]:
    return [item["function"]["name"].replace("__", ".", 1) for item in get_tool_schemas()]


def _schema_for(slug: str) -> str:
    for item in get_tool_schemas():
        if item["function"]["name"].replace("__", ".", 1) == slug:
            return json.dumps(item["function"], ensure_ascii=False, indent=2)
    return "{}"


def inspect_files(files):
    paths = [str(item) for item in (files or [])]
    if not paths:
        return "未选择文件"
    return "```json\n" + json.dumps(inspect_uploaded_files(paths), ensure_ascii=False, indent=2) + "\n```"


def run_agent(prompt: str, files, provider: str, language: str, max_rounds: int, harness: str):
    if harness == "oea":
        yield from run_oea_agent(prompt, files, provider, language, max_rounds)
        return
    if not (prompt or "").strip():
        yield "请输入遥感任务描述。", "", ""
        return
    paths = [str(item) for item in (files or [])]
    sample = {"prompt": prompt.strip(), "files": paths, "tool_calls": []}
    blocks: list[str] = []
    final = ""
    tools: list[str] = []
    for event in run_sample(sample, None, mode="llm_driven", max_rounds=int(max_rounds), provider=provider, language=language):
        typ = event.get("type")
        if typ == "llm_start":
            blocks.append("### Agent 开始规划")
        elif typ == "tool_start":
            tools.append(event.get("slug", ""))
            blocks.append(
                f"### Step {event.get('step')}: `{html.escape(event.get('slug', ''))}`\n"
                f"```json\n{json.dumps(event.get('args', {}), ensure_ascii=False, indent=2)}\n```\n执行中..."
            )
        elif typ == "tool_result":
            value = event.get("display_value")
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str) if isinstance(value, (dict, list)) else str(value)
            blocks.append(f"**工具返回：**\n```text\n{text[:4000]}\n```")
        elif typ == "tool_error":
            blocks.append(f"**工具失败：** `{html.escape(str(event.get('error', '')))}")
        elif typ == "llm_text":
            final = str(event.get("text", ""))
        elif typ == "error":
            blocks.append(f"**错误：** {html.escape(str(event.get('message', '')))}")
        elif typ == "done":
            blocks.append(f"### 完成\n工具调用：{len(tools)} 次")
        yield "\n\n".join(blocks), final, ", ".join(tools)


def run_single_tool(slug: str, args_text: str, files):
    if not slug:
        return "请选择工具。"
    try:
        args = json.loads(args_text or "{}")
    except json.JSONDecodeError as exc:
        return f"inputs JSON 无法解析：{exc}"
    paths = [str(item) for item in (files or [])]
    if paths:
        args.setdefault("files", paths)
        args.setdefault("data_files", paths)
        if len(paths) == 1:
            args.setdefault("image", paths[0])
            args.setdefault("image_path", paths[0])
        else:
            args.setdefault("images", paths)
    result = execute_tool(slug, args)
    return "```json\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n```"


CSS = ".panel {border:1px solid #d1d5db;border-radius:8px;padding:10px}.mono textarea{font-family:monospace}"


def build_ui():
    with gr.Blocks(title="Terrabox 遥感 Agent 演示", css=CSS) as demo:
        gr.Markdown("# Terrabox 遥感智能分析演示\n支持图片、GeoTIFF、GeoJSON、GPKG、SHP、KML 等输入文件。")
        with gr.Tabs():
            with gr.Tab("Agent 任务执行"):
                prompt = gr.Textbox(label="遥感任务描述 / Remote-sensing task", lines=5, placeholder="例如：分析上传的 GeoTIFF，计算 NDVI 并生成结果图。")
                files = gr.File(label="上传输入文件", file_count="multiple", type="filepath")
                file_info = gr.Markdown("选择文件后显示格式、CRS 和栅格元数据", elem_classes=["panel"])
                with gr.Row():
                    harness = gr.Radio([("OEA harness（推荐）", "oea"), ("Demo JSON harness", "demo")], value="oea", label="Agent harness")
                    language = gr.Radio([("中文", "zh"), ("English", "en")], value="zh", label="回答语言")
                    provider = gr.Radio([("DeepSeek", "deepseek"), ("LongCat", "longcat")], value="deepseek", label="Agent LLM")
                    rounds = gr.Slider(1, 15, value=8, step=1, label="最大工具轮数")
                run = gr.Button("运行 Agent", variant="primary")
                timeline = gr.Markdown("等待执行...", elem_classes=["panel"])
                with gr.Row():
                    answer = gr.Markdown("", elem_classes=["panel"])
                    called = gr.Textbox(label="工具调用序列", interactive=False)
                files.change(inspect_files, inputs=files, outputs=file_info)
                run.click(run_agent, inputs=[prompt, files, provider, language, rounds, harness], outputs=[timeline, answer, called])
            with gr.Tab("工具广场"):
                tool = gr.Dropdown(_tool_choices(), label="选择工具", interactive=True)
                schema = gr.Code(_schema_for(_tool_choices()[0]) if _tool_choices() else "{}", language="json", label="工具 Schema", interactive=False)
                tool_args = gr.Code("{}", language="json", label="输入参数 JSON", lines=10)
                tool_files = gr.File(label="工具文件输入", file_count="multiple", type="filepath")
                execute = gr.Button("执行工具", variant="primary")
                tool_result = gr.Markdown("等待执行...", elem_classes=["panel"])
                tool.change(_schema_for, inputs=tool, outputs=schema)
                execute.click(run_single_tool, inputs=[tool, tool_args, tool_files], outputs=tool_result)
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port, show_error=True)
