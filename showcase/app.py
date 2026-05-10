"""
app.py — Disaster SFT 可视化演示工具
=====================================
启动方式：
  conda activate unsloth
  cd /data1/yuhongjie2/terrabox
  python showcase/app.py

浏览器访问 http://localhost:7860
"""
from __future__ import annotations

import html
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# sys.path 优先设置，使 showcase.* 和 terrabox.* 可导入
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# 代理绕过：通过共享工具函数设置（必须在任何 HTTP 请求前执行）
from showcase.docker_utils import (
    ensure_local_no_proxy,
    get_system_status,
    find_running_llm_port,
    is_llm_running,
    perception_manager,
    cleanup_showcase_services,
    start_llm_service,
)
from showcase.runner import run_sample

ensure_local_no_proxy()

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("showcase")

# ---------------------------------------------------------------------------
# 加载数据集
# ---------------------------------------------------------------------------
# v2 = 修正版（移除不兼容RGB的多光谱工具链 + 冗余工具）
# 若 v2 不存在则回退到原始数据集
_DATASET_V2  = _ROOT / "data" / "disaster_sft_dataset_v2.json"
_DATASET_V1  = _ROOT / "data" / "disaster_sft_dataset.json"
_DATASET_PATH = _DATASET_V2 if _DATASET_V2.exists() else _DATASET_V1


def load_dataset() -> list[dict]:
    with open(_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", data) if isinstance(data, dict) and "samples" in data else (
        data if isinstance(data, list) else []
    )


def _load_samples() -> list[dict]:
    """加载并返回所有样本（从 JSON 顶层 key 自动探测列表位置）。"""
    with open(_DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # 数据集格式：顶层是 dict，样本在某个列表字段里
    if isinstance(raw, list):
        return raw
    for v in raw.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "tool_calls" in v[0]:
            return v
    return []


SAMPLES: list[dict] = _load_samples()

# 构建下拉选项：id + task_type
SAMPLE_CHOICES = [
    f"{s['id']} [{s.get('task_type', '')}] ({s.get('difficulty', '')})"
    for s in SAMPLES
]

# ---------------------------------------------------------------------------
# 结果记录：保存实际执行结果到 showcase_results.json
# ---------------------------------------------------------------------------
_RESULTS_PATH = _ROOT / "data" / "showcase_results.json"

# 模块级缓存：存储最近一次完成的运行结果，供保存按钮使用
_last_run_result: dict | None = None  # {"sample": dict, "actual_tool_calls": list}
_result_lock = threading.Lock()


def _load_results() -> dict:
    if _RESULTS_PATH.exists():
        with open(_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"version": "1.0", "results": []}


def save_run_result() -> str:
    """将 _last_run_result 追加到 showcase_results.json，重复则跳过。"""
    global _last_run_result
    with _result_lock:
        result_snapshot = _last_run_result

    if result_snapshot is None:
        return "⚠️ 没有可保存的结果，请先运行一条样本"

    sample = result_snapshot["sample"]
    actual_tool_calls = result_snapshot["actual_tool_calls"]
    sample_id = sample.get("id", "")

    data = _load_results()

    # 重复检查
    for existing in data["results"]:
        if existing.get("id") == sample_id:
            return f"⚠️ 样本 `{sample_id}` 已存在，未追加（共 {len(data['results'])} 条）"

    entry = {
        "id": sample_id,
        "task_type": sample.get("task_type", ""),
        "disaster_category": sample.get("disaster_category", ""),
        "difficulty": sample.get("difficulty", ""),
        "prompt": sample.get("prompt", ""),
        "tool_calls": actual_tool_calls,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["results"].append(entry)
    data["total_results"] = len(data["results"])

    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved result for {sample_id} → {_RESULTS_PATH}")
    return f"✅ 已保存 `{sample_id}`（当前共 {len(data['results'])} 条，文件：{_RESULTS_PATH.name}）"


def _get_sample(choice: str) -> dict | None:
    idx = SAMPLE_CHOICES.index(choice) if choice in SAMPLE_CHOICES else -1
    return SAMPLES[idx] if idx >= 0 else None


# ---------------------------------------------------------------------------
# 辅助：格式化函数
# ---------------------------------------------------------------------------

def _format_tool_chain(tool_calls: list[dict]) -> str:
    """将 SFT 工具链格式化为可读文本。"""
    lines = []
    for tc in tool_calls:
        step = tc.get("step", "?")
        tool = tc.get("tool", "unknown")
        note = tc.get("note", "")
        lines.append(f"**Step {step}** `{tool}`")
        if note:
            lines.append(f"  _{note}_")
    return "\n".join(lines)


def _format_args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False, indent=2)


def _format_dict_result(val) -> str:
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False, indent=2, default=str)
    return str(val)


# ---------------------------------------------------------------------------
# Gradio 回调函数
# ---------------------------------------------------------------------------

def on_sample_select(choice: str):
    """选择样本后，更新预览区域。"""
    sample = _get_sample(choice)
    if not sample:
        return gr.update(value=""), gr.update(value=""), gr.update(value="")

    prompt = sample.get("prompt", "")
    tool_chain_md = _format_tool_chain(sample.get("tool_calls", []))
    meta = (
        f"**ID:** {sample.get('id', '')}\n"
        f"**类型:** {sample.get('task_type', '')}\n"
        f"**灾害类别:** {sample.get('disaster_category', '')}\n"
        f"**难度:** {sample.get('difficulty', '')}\n"
        f"**步骤数:** {len(sample.get('tool_calls', []))}"
    )
    return gr.update(value=prompt), gr.update(value=tool_chain_md), gr.update(value=meta)


def on_refresh_status():
    """刷新 GPU/LLM 状态。"""
    status = get_system_status()

    lines = ["### 系统状态\n"]
    if status["gpus"]:
        for g in status["gpus"]:
            if g["llm_running"]:
                llm_state = f"🟢 LLM ready `{g['llm_container']}`"
            elif g.get("llm_container_running"):
                llm_state = f"🟡 LLM loading `{g['llm_container']}`"
            else:
                llm_state = "⚫ LLM idle"
            lines.append(
                f"- **GPU {g['gpu_id']}** 空闲 {g['free_mb']}MB / {g['total_mb']}MB "
                f"({g['free_pct']}%)  port {g['port']}  {llm_state}"
            )
    else:
        lines.append("- ⚠️ 未检测到 GPU")

    ports = status.get("llm_ports", [])
    llm_status = f"🟢 运行中 (ports {', '.join(map(str, ports))})" if ports else "⚫ 未运行"
    lines.append(f"\n**LLM 服务:** {llm_status}")

    # 感知容器状态
    running = perception_manager.running_containers()
    if running:
        lines.append("\n**感知容器（LRU 管理）:**")
        for c in running:
            idle_min = c["idle_s"] // 60
            idle_sec = c["idle_s"] % 60
            lines.append(f"- 🐳 `{c['container']}` 空闲 {idle_min}m{idle_sec:02d}s（5min 后自动关停）")
    else:
        lines.append("\n**感知容器:** 无运行中")

    return gr.update(value="\n".join(lines))


def on_start_llm(llm_port_input: int):
    """启动 LLM 服务。"""
    port = int(llm_port_input) if llm_port_input else None
    target = f"port {port}" if port else "自动选择端口"
    yield gr.update(value=f"⏳ 正在启动 Docker LLM（{target}），请稍候（可能需要数分钟）...", interactive=False)
    success, port, msg = start_llm_service(port=port)
    if success:
        yield gr.update(value=f"✅ {msg}", interactive=True)
    else:
        yield gr.update(value=f"❌ {msg}", interactive=True)


def run_demo(
    choice: str,
    mode: str,
    llm_port_input: int,
    max_rounds: int,
):
    """
    主执行函数，Generator，逐步 yield (steps_html, images, llm_output, compare_md)。
    """
    sample = _get_sample(choice)
    if not sample:
        yield (
            "<p class='tb-error-text'>请先选择一个样本</p>",
            [],
            "",
            "",
        )
        return

    # 确定 LLM 端口
    if mode == "llm_driven":
        port = int(llm_port_input) if llm_port_input else find_running_llm_port()
        if port is None or not is_llm_running(port):
            yield (
                "<p class='tb-error-text'>⚠️ LLM 服务未运行，请先点击「启动 LLM」或切换到「Replay 模式」</p>",
                [],
                "",
                "",
            )
            return
    else:
        port = 9100  # replay 模式不需要端口

    # 执行状态跟踪
    steps_blocks: list[str] = []  # HTML 片段列表
    images: list[str] = []        # 图像路径列表
    llm_final_text = ""
    actual_tools: list[str] = []
    expected_tools = [tc["tool"] for tc in sample.get("tool_calls", [])]

    def _render_steps() -> str:
        return "\n".join(steps_blocks) if steps_blocks else "<p class='tb-muted'>等待执行...</p>"

    def _render_compare() -> str:
        if not actual_tools:
            return ""
        exp_set = set(expected_tools)
        act_set = set(actual_tools)
        tp = exp_set & act_set
        fp = act_set - exp_set
        fn = exp_set - act_set

        lines = ["### 工具链对比\n"]
        lines.append(f"**预期步骤数:** {len(expected_tools)}  **实际步骤数:** {len(actual_tools)}\n")

        lines.append("| # | 预期工具 | 实际工具 | 匹配 |")
        lines.append("|---|---------|---------|------|")
        max_len = max(len(expected_tools), len(actual_tools))
        for i in range(max_len):
            exp = expected_tools[i] if i < len(expected_tools) else "—"
            act = actual_tools[i] if i < len(actual_tools) else "—"
            match = "✅" if exp == act else "❌"
            lines.append(f"| {i+1} | `{exp}` | `{act}` | {match} |")

        if tp:
            p = len(tp) / len(act_set) if act_set else 0
            r = len(tp) / len(exp_set) if exp_set else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            lines.append(f"\n**Precision:** {p:.2%}  **Recall:** {r:.2%}  **F1:** {f1:.2%}")

        return "\n".join(lines)

    # 初始渲染
    yield (_render_steps(), images[:], llm_final_text, "")

    for event in run_sample(sample, port, mode=mode, max_rounds=int(max_rounds)):
        etype = event["type"]

        if etype == "llm_start":
            steps_blocks.append(
                "<div class='tb-alert tb-alert-info'>🤖 <b>LLM 开始分析任务...</b></div>"
            )

        elif etype == "tool_start":
            step = event["step"]
            slug = event["slug"]
            args_str = html.escape(json.dumps(event["args"], ensure_ascii=False, indent=2))
            slug_html = html.escape(slug)
            block = (
                f"<div class='tb-step-card'>"
                f"<div class='tb-step-title'>Step {step}: <code>{slug_html}</code></div>"
                f"<div class='tb-step-body'>"
                f"<details><summary>📥 输入参数</summary>"
                f"<pre class='tb-pre'>{args_str}</pre></details>"
                f"<div id='result-step-{step}' class='tb-muted tb-running'>⏳ 执行中...</div>"
                f"</div></div>"
            )
            steps_blocks.append(block)

        elif etype == "service_starting":
            slug = event["slug"]
            container = event["container"]
            steps_blocks.append(
                f"<div class='tb-alert tb-alert-info'>"
                f"🐳 <b>启动容器</b> <code>{html.escape(container)}</code>（{html.escape(slug)}）... 请稍候</div>"
            )

        elif etype == "service_started":
            slug = event["slug"]
            container = event["container"]
            msg = event.get("message", "")
            steps_blocks.append(
                f"<div class='tb-alert tb-alert-success'>"
                f"✅ <b>容器就绪</b> <code>{html.escape(container)}</code>: {html.escape(msg)}</div>"
            )

        elif etype == "service_start_failed":
            slug = event["slug"]
            container = event["container"]
            msg = event.get("message", "")
            steps_blocks.append(
                f"<div class='tb-alert tb-alert-error'>"
                f"❌ <b>容器启动失败</b> <code>{html.escape(container)}</code>: {html.escape(msg)}</div>"
            )

        elif etype == "tool_service_warn":
            slug = event["slug"]
            msg = event["message"]
            steps_blocks.append(
                f"<div class='tb-alert tb-alert-warn'>"
                f"⚠️ <b>{html.escape(slug)}</b>: {html.escape(msg)}</div>"
            )

        elif etype == "tool_result":
            step = event["step"]
            slug = event["slug"]
            output_type = event["output_type"]
            display_val = event["display_value"]
            note = event.get("note", "")

            if output_type == "image_path" and display_val and Path(display_val).exists():
                images = images + [display_val]
                note_html = f"<br><span class='tb-note'>{html.escape(note)}</span>" if note else ""
                result_html = (
                    f"<div class='tb-result-ok'>✅ 输出图像 → <code>{html.escape(str(display_val))}</code>"
                    f"{note_html}</div>"
                )
            else:
                val_str = _format_dict_result(display_val)
                # 截断过长输出
                if len(val_str) > 800:
                    val_str = val_str[:800] + "\n... (已截断)"
                val_str = html.escape(val_str)
                note_html = f"<div class='tb-note'>{html.escape(note)}</div>" if note else ""
                result_html = (
                    f"<div class='tb-result-ok'>✅ 输出</div>"
                    f"<pre class='tb-pre tb-result-pre'>{val_str}</pre>"
                    f"{note_html}"
                )

            # 更新对应 step 的结果区域（追加到最后一个块）
            if steps_blocks:
                last = steps_blocks[-1]
                placeholder = f"<div id='result-step-{step}' class='tb-muted tb-running'>⏳ 执行中...</div>"
                if placeholder in last:
                    steps_blocks[-1] = last.replace(placeholder, f"<div>{result_html}</div>")
                else:
                    steps_blocks.append(f"<div class='tb-inline-result'>{result_html}</div>")

        elif etype == "tool_error":
            step = event["step"]
            slug = event["slug"]
            error = event["error"]
            err_html = (
                f"<div class='tb-alert tb-alert-error'>"
                f"❌ <b>Step {step} ({html.escape(slug)}) 错误:</b> {html.escape(error)}</div>"
            )
            if steps_blocks:
                last = steps_blocks[-1]
                placeholder = f"<div id='result-step-{step}' class='tb-muted tb-running'>⏳ 执行中...</div>"
                if placeholder in last:
                    steps_blocks[-1] = last.replace(placeholder, err_html)
                else:
                    steps_blocks.append(err_html)

        elif etype == "llm_text":
            llm_final_text = event["text"]

        elif etype == "done":
            actual_tools = event.get("tool_calls_made", [])
            actual_tc = event.get("actual_tool_calls", [])
            global _last_run_result
            with _result_lock:
                _last_run_result = {"sample": sample, "actual_tool_calls": actual_tc}
            steps_blocks.append(
                "<div class='tb-alert tb-alert-success'>"
                f"✅ <b>执行完成</b>，共调用 {len(actual_tools)} 个工具</div>"
            )

        elif etype == "error":
            steps_blocks.append(
                f"<div class='tb-alert tb-alert-error'>"
                f"❌ <b>错误:</b> {html.escape(event['message'])}</div>"
            )

        yield (_render_steps(), images[:], llm_final_text, _render_compare())


# ---------------------------------------------------------------------------
# Gradio UI 构建
# ---------------------------------------------------------------------------

_CSS = """
.gradio-container {
  color: #111827;
}
.sample-meta {
  font-size: 13px;
  color: #1f2937;
}
.step-panel {
  max-height: 600px;
  overflow-y: auto;
  background: #ffffff;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: 8px;
}
.tb-status-panel,
.tb-compare-panel,
.tb-save-status {
  background: #ffffff;
  color: #111827;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: 10px 12px;
}
.tb-status-panel p,
.tb-status-panel li,
.tb-compare-panel p,
.tb-compare-panel li,
.tb-save-status p {
  color: #111827;
}
.tb-compare-panel table {
  background: #ffffff;
  color: #111827;
}
.tb-muted {
  color: #4b5563;
}
.tb-error-text {
  color: #991b1b;
  font-weight: 600;
}
.tb-step-card {
  border: 1px solid #9ca3af;
  border-radius: 8px;
  margin: 8px 0;
  overflow: hidden;
  background: #ffffff;
  color: #111827;
}
.tb-step-title {
  background: #1d4ed8;
  color: #ffffff;
  padding: 7px 12px;
  font-weight: 700;
}
.tb-step-title code {
  background: #dbeafe;
  color: #0f172a;
}
.tb-step-body {
  padding: 8px;
  background: #ffffff;
  color: #111827;
}
.tb-step-body summary {
  cursor: pointer;
  color: #1f2937;
  font-weight: 600;
}
.tb-alert {
  padding: 8px 12px;
  margin: 6px 0;
  border-radius: 6px;
  border-left: 5px solid #374151;
  background: #ffffff;
  color: #111827;
}
.tb-alert-info {
  background: #eff6ff;
  border-left-color: #1d4ed8;
}
.tb-alert-success {
  background: #ecfdf5;
  border-left-color: #047857;
}
.tb-alert-warn {
  background: #fffbeb;
  border-left-color: #b45309;
}
.tb-alert-error {
  background: #fef2f2;
  border-left-color: #b91c1c;
}
.tb-result-ok {
  color: #065f46;
  font-weight: 600;
}
.tb-note {
  color: #92400e;
  font-weight: 600;
}
.tb-inline-result {
  padding: 4px 12px;
}
.tb-pre {
  margin: 6px 0;
  padding: 8px;
  border-radius: 6px;
  max-height: 220px;
  overflow: auto;
  background: #111827;
  color: #f9fafb;
  font-size: 12px;
  line-height: 1.45;
}
.tb-result-pre {
  background: #064e3b;
  color: #ecfdf5;
}
code {
  background: #e5e7eb;
  color: #111827;
  padding: 1px 4px;
  border-radius: 3px;
}
"""


def build_ui():
    with gr.Blocks(title="Terrabox Disaster SFT 演示") as demo:
        gr.Markdown(
            "# 🌏 Terrabox Disaster SFT 可视化演示\n"
            "选择一条灾害分析样本，让 LLM 调用工具链完成端到端分析。"
        )

        with gr.Row():
            # ----------------------------------------------------------------
            # 左侧：控制面板
            # ----------------------------------------------------------------
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 样本选择")
                sample_dd = gr.Dropdown(
                    choices=SAMPLE_CHOICES,
                    label="选择 SFT 样本",
                    value=SAMPLE_CHOICES[0] if SAMPLE_CHOICES else None,
                    interactive=True,
                )
                random_btn = gr.Button("🎲 随机样本", size="sm", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### 样本信息")
                meta_md = gr.Markdown(elem_classes=["sample-meta"])
                prompt_box = gr.Textbox(
                    label="任务描述",
                    lines=4,
                    interactive=False,
                    max_lines=6,
                )
                toolchain_md = gr.Markdown(label="预期工具链")

                gr.Markdown("---")
                gr.Markdown("### 执行配置")
                mode_radio = gr.Radio(
                    choices=["llm_driven", "replay"],
                    value="llm_driven",
                    label="执行模式",
                    info="llm_driven: LLM 自主决策  |  replay: 直接按 SFT 记录执行",
                )
                llm_port_num = gr.Number(
                    label="LLM 端口",
                    value=9100,
                    precision=0,
                    minimum=9100,
                    maximum=9103,
                    info="vLLM 服务端口 (9100-9103)",
                )
                max_rounds_num = gr.Slider(
                    label="最大 LLM 轮数",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                )

                gr.Markdown("---")
                gr.Markdown("### 服务状态")
                status_md = gr.Markdown("_点击刷新查看_", elem_classes=["tb-status-panel"])
                status_timer = gr.Timer(value=5, active=True)
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新状态", size="sm")
                    start_llm_btn = gr.Button("🚀 启动当前端口 Docker LLM", size="sm", variant="primary")
                llm_msg_box = gr.Textbox(label="LLM 启动日志", lines=2, interactive=False)

                gr.Markdown("---")
                run_btn = gr.Button("▶️ 运行", variant="primary", size="lg")

            # ----------------------------------------------------------------
            # 右侧：执行过程展示
            # ----------------------------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 执行过程")
                steps_html = gr.HTML(
                    value="<p class='tb-muted'>选择样本后点击「运行」开始演示</p>",
                    elem_classes=["step-panel"],
                )

                gr.Markdown("### 图像输出")
                images_gallery = gr.Gallery(
                    label="工具输出图像",
                    show_label=False,
                    columns=3,
                    height=300,
                    object_fit="contain",
                )

                with gr.Accordion("💬 LLM 最终回复", open=False):
                    llm_output_box = gr.Textbox(
                        label="",
                        lines=6,
                        interactive=False,
                    )

                gr.Markdown("### 工具链对比")
                compare_md = gr.Markdown("_执行完成后显示_", elem_classes=["tb-compare-panel"])

                gr.Markdown("---")
                gr.Markdown("### 保存结果")
                save_btn = gr.Button("💾 保存本次结果到 showcase_results.json", variant="secondary")
                save_status = gr.Markdown("_运行完成后可保存_", elem_classes=["tb-save-status"])

        # --------------------------------------------------------------------
        # 事件绑定
        # --------------------------------------------------------------------
        # 样本选择
        sample_dd.change(
            fn=on_sample_select,
            inputs=[sample_dd],
            outputs=[prompt_box, toolchain_md, meta_md],
        )

        # 随机样本
        def pick_random():
            choice = random.choice(SAMPLE_CHOICES)
            return gr.update(value=choice)

        random_btn.click(
            fn=pick_random,
            outputs=[sample_dd],
        ).then(
            fn=on_sample_select,
            inputs=[sample_dd],
            outputs=[prompt_box, toolchain_md, meta_md],
        )

        # 刷新状态
        refresh_btn.click(fn=on_refresh_status, outputs=[status_md], queue=False)
        status_timer.tick(fn=on_refresh_status, outputs=[status_md], show_progress="hidden", queue=False)

        # 启动 LLM（streaming，显示进度）
        start_llm_btn.click(
            fn=on_start_llm,
            inputs=[llm_port_num],
            outputs=[llm_msg_box],
        )

        # 运行演示
        run_btn.click(
            fn=run_demo,
            inputs=[sample_dd, mode_radio, llm_port_num, max_rounds_num],
            outputs=[steps_html, images_gallery, llm_output_box, compare_md],
        )

        # 保存结果
        save_btn.click(fn=save_run_result, outputs=[save_status])

        # 初始化：加载第一个样本 & 刷新状态
        demo.load(
            fn=on_sample_select,
            inputs=[sample_dd],
            outputs=[prompt_box, toolchain_md, meta_md],
        )
        demo.load(fn=on_refresh_status, outputs=[status_md])

    return demo


# ---------------------------------------------------------------------------
# 模块级 demo — Gradio CLI 热重载要求变量名必须为 demo
# 运行方式（热重载）：gradio showcase/app.py
# ---------------------------------------------------------------------------
demo = build_ui()
demo.queue()

import atexit
_cleanup_done = False
_cleanup_lock = threading.Lock()


def _cleanup_on_exit() -> None:
    global _cleanup_done
    with _cleanup_lock:
        if _cleanup_done:
            return
        _cleanup_done = True
    cleanup_showcase_services()


def _handle_exit_signal(signum, _frame) -> None:
    logger.info("Received signal %s; cleaning up showcase-owned Docker services", signum)
    _cleanup_on_exit()
    raise SystemExit(128 + int(signum))


atexit.register(_cleanup_on_exit)
if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGTERM, _handle_exit_signal)

# ---------------------------------------------------------------------------
# 直接运行入口（python showcase/app.py）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Terrabox Disaster SFT 可视化演示")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--share", action="store_true", help="生成公开分享链接")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="禁用普通运行时热重载，直接 launch 当前进程",
    )
    args = parser.parse_args()

    if not args.no_reload and "GRADIO_WATCH_DIRS" not in os.environ:
        env = os.environ.copy()
        env["GRADIO_SERVER_NAME"] = args.host
        env["GRADIO_SERVER_PORT"] = str(args.port)
        if args.share:
            env["GRADIO_SHARE"] = "true"
        os.chdir(_ROOT)
        cmd = [
            sys.executable,
            "-m",
            "gradio",
            "--watch-dirs",
            "showcase",
            "--watch-dirs",
            "src",
            "showcase/app.py",
        ]
        logger.info("Starting Gradio reload supervisor: %s", " ".join(cmd))
        os.execvpe(sys.executable, cmd, env)

    logger.info(f"Loaded {len(SAMPLES)} SFT samples")
    logger.info(f"Starting Gradio on {args.host}:{args.port}")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
