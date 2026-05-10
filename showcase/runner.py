"""
runner.py — LLM function calling 执行流水线
===========================================
Generator 函数，逐步 yield 每个工具调用的结果，供 Gradio streaming 消费。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Generator

# 确保 src/ 在路径中（runner.py 作为包被导入时，tool_bridge 也需要能 import terrabox）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

logger = logging.getLogger(__name__)

from .docker_utils import ensure_local_no_proxy
ensure_local_no_proxy()

# ---------------------------------------------------------------------------
# 影像路径映射（sft_image_mapping.json）
# ---------------------------------------------------------------------------
_IMAGE_MAPPING: dict[str, dict] = {}


def _load_image_mapping() -> dict[str, dict]:
    """懒加载 sft_image_mapping.json，返回 {sft_id: entry} 字典。"""
    global _IMAGE_MAPPING
    if _IMAGE_MAPPING:
        return _IMAGE_MAPPING
    mapping_path = _ROOT / "data" / "sft_image_mapping.json"
    if not mapping_path.exists():
        return {}
    try:
        with open(mapping_path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("mappings", []):
            sid = entry.get("sft_id")
            if sid:
                _IMAGE_MAPPING[sid] = entry
        logger.info(f"Loaded image mapping for {len(_IMAGE_MAPPING)} samples")
    except Exception as e:
        logger.warning(f"Failed to load image mapping: {e}")
    return _IMAGE_MAPPING


def _get_image_context(sample_id: str) -> str:
    """
    返回注入 prompt 的影像文件上下文块。
    格式：
      [影像文件]
      灾后影像: /abs/path/to/post.png
      灾前影像: /abs/path/to/pre.png  (如有)
      格式: PNG 1024×1024
    """
    mapping = _load_image_mapping()
    entry = mapping.get(sample_id)
    if not entry:
        return ""
    lines = ["[影像文件]"]
    if entry.get("image_post"):
        lines.append(f"灾后影像: {entry['image_post']}")
    if entry.get("image_pre"):
        lines.append(f"灾前影像: {entry['image_pre']}")
    fmt = entry.get("format", "")
    size = entry.get("image_size", "")
    if fmt or size:
        lines.append(f"格式: {fmt} {size}".strip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 事件类型定义（yield 的 dict 结构）
# ---------------------------------------------------------------------------
# {"type": "llm_start"}
# {"type": "tool_start",  "step": int, "slug": str, "fn_name": str, "args": dict}
# {"type": "tool_service_warn", "slug": str, "message": str}
# {"type": "tool_result",  "step": int, "slug": str, "output_type": str, "display_value": Any, "raw": Any}
# {"type": "tool_error",   "step": int, "slug": str, "error": str}
# {"type": "llm_text",     "text": str}      # LLM 最终文本回复
# {"type": "done",         "tool_calls_made": list[str]}
# {"type": "error",        "message": str}   # 全局错误


def run_sample(
    sample: dict,
    llm_port: int,
    mode: str = "llm_driven",
    max_rounds: int = 10,
) -> Generator[dict, None, None]:
    """
    执行单条 SFT 样本的工具链。

    Parameters
    ----------
    sample : dict
        来自 disaster_sft_dataset.json 的单条样本。
    llm_port : int
        vLLM 服务端口。
    mode : str
        "llm_driven"  — LLM 自主决策工具调用（推荐）
        "replay"      — 直接按 SFT 样本中的 tool_calls 顺序执行，不走 LLM
    max_rounds : int
        LLM function calling 最大轮数，防止死循环。
    """
    if mode == "replay":
        yield from _replay_mode(sample)
    else:
        yield from _llm_driven_mode(sample, llm_port, max_rounds)


# ---------------------------------------------------------------------------
# Replay 模式：直接执行 SFT 中记录的工具链
# ---------------------------------------------------------------------------

def _inject_real_paths(args: dict, entry: dict) -> dict:
    """
    将 SFT args 中的占位路径替换为真实路径：
    - 输入影像占位符（rgb.tif / post.png / pre.png 等）→ sft_image_mapping 中的真实路径
    - 相对路径的输出文件（output_path / result_path / out_path）→ /tmp/<filename>
    """
    post = entry.get("image_post", "") if entry else ""
    pre = entry.get("image_pre", "") if entry else ""

    # 输出字段名集合
    OUTPUT_KEYS = {"output_path", "result_path", "out_path"}

    def _replace_input(v: str) -> str:
        """替换输入影像占位符。"""
        if not isinstance(v, str) or "/" in v:
            return v
        lower = v.lower()
        if any(lower.startswith(p) for p in ("post", "rgb")):
            return post or v
        if lower.startswith("pre"):
            return pre or v
        return v

    def _fix_output(k: str, v: str) -> str:
        """将相对路径的输出文件重定向到 /tmp/。"""
        if not isinstance(v, str):
            return v
        if k in OUTPUT_KEYS and "/" not in v and v:
            return f"/tmp/showcase_{v}"
        return v

    injected = {}
    for k, val in args.items():
        if k in OUTPUT_KEYS:
            injected[k] = _fix_output(k, val)
        elif isinstance(val, list):
            injected[k] = [_replace_input(item) for item in val]
        else:
            injected[k] = _replace_input(val)
    return injected


def _replay_mode(sample: dict) -> Generator[dict, None, None]:
    """按 SFT 样本中的 tool_calls 顺序直接执行，不经过 LLM。"""
    from .tool_bridge import execute_tool
    from .docker_utils import perception_manager, TOOL_DOCKER_FULL_MAP

    sample_id = sample.get("id", "")
    mapping = _load_image_mapping()
    img_entry = mapping.get(sample_id, {})

    tool_calls = sample.get("tool_calls", [])
    tool_calls_made: list[str] = []
    actual_tool_calls: list[dict] = []
    step_results: dict[int, dict] = {}

    for tc in tool_calls:
        step = tc["step"]
        slug = tc["tool"]
        raw_args = tc.get("args", {})

        raw_args = _inject_real_paths(raw_args, img_entry)
        args = _resolve_refs(raw_args, step_results)

        yield {"type": "tool_start", "step": step, "slug": slug, "fn_name": slug, "args": args}

        # 感知类工具：通过 LRU 管理器按需启动
        if slug in TOOL_DOCKER_FULL_MAP:
            container_name = TOOL_DOCKER_FULL_MAP[slug][0]
            yield {"type": "service_starting", "slug": slug, "container": container_name}
            success, svc_msg, was_started = perception_manager.acquire(slug)
            if not success:
                yield {"type": "service_start_failed", "slug": slug, "container": container_name, "message": svc_msg}
                fallback = tc.get("sample_output", {})
                step_results[step] = fallback
                yield {
                    "type": "tool_result", "step": step, "slug": slug,
                    "output_type": "dict", "display_value": fallback, "raw": fallback,
                    "note": "⚠️ Service unavailable — using sample_output from dataset",
                }
                actual_tool_calls.append({"step": step, "tool": slug, "args": args, "sample_output": fallback})
                tool_calls_made.append(slug)
                continue
            if was_started:
                yield {"type": "service_started", "slug": slug, "container": container_name, "message": svc_msg}

        result = execute_tool(slug, args)

        # 工具执行完后刷新 LRU 时间
        if slug in TOOL_DOCKER_FULL_MAP:
            perception_manager.touch(slug)

        if result["output_type"] == "error":
            fallback = tc.get("sample_output", {})
            step_results[step] = fallback
            yield {"type": "tool_service_warn", "slug": slug,
                   "message": f"执行失败（{result['display_value'][:120]}）→ 使用 sample_output 代替"}
            yield {
                "type": "tool_result", "step": step, "slug": slug,
                "output_type": "dict", "display_value": fallback, "raw": fallback,
                "note": "⚠️ 执行错误 — 使用数据集 sample_output 模拟输出",
            }
            actual_tool_calls.append({"step": step, "tool": slug, "args": args, "sample_output": fallback})
        else:
            actual_output = result.get("raw") or {}
            step_results[step] = actual_output
            yield {
                "type": "tool_result", "step": step, "slug": slug,
                "output_type": result["output_type"],
                "display_value": result["display_value"],
                "raw": result["raw"],
            }
            actual_tool_calls.append({"step": step, "tool": slug, "args": args, "sample_output": actual_output})
        tool_calls_made.append(slug)

    yield {"type": "done", "tool_calls_made": tool_calls_made, "actual_tool_calls": actual_tool_calls}


def _resolve_refs(args: dict, step_results: dict[int, dict]) -> dict:
    """将 args 中的 '$stepN.field' 引用替换为实际值。"""
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("$step"):
            # 格式：$step3.count 或 $step3.bboxes
            try:
                rest = v[5:]  # 去掉 "$step"
                parts = rest.split(".", 1)
                step_num = int(parts[0])
                field = parts[1] if len(parts) > 1 else None
                step_val = step_results.get(step_num, {})
                if field and isinstance(step_val, dict):
                    resolved[k] = step_val.get(field, v)
                else:
                    resolved[k] = step_val
            except (ValueError, IndexError):
                resolved[k] = v
        else:
            resolved[k] = v
    return resolved


# ---------------------------------------------------------------------------
# LLM Driven 模式：JSON 行动协议（不依赖 --enable-auto-tool-choice）
# ---------------------------------------------------------------------------

def _build_tool_prompt(schemas: list[dict]) -> str:
    """将 OpenAI schema 列表转换为系统提示中的工具描述文本。"""
    lines = []
    for item in schemas:
        fn = item["function"]
        slug = fn["name"].replace("__", ".", 1)
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            req_mark = "*" if pname in required else ""
            pdesc = pinfo.get("description", "")
            param_parts.append(f"  - {pname}{req_mark} ({ptype}): {pdesc}")
        param_str = "\n".join(param_parts) if param_parts else "  (无参数)"
        lines.append(f"[{slug}]\n描述: {desc}\n参数:\n{param_str}")
    return "\n\n".join(lines)


def _parse_json_action(text: str) -> dict | None:
    """
    从 LLM 输出中提取 JSON action 对象。
    处理以下情况：
    - 纯 JSON 输出
    - Markdown 代码块包裹的 JSON
    - Qwen3 带 <think>...</think> 思考块的输出
    """
    import re

    # 去除 Qwen3 思考块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 去除 Markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip()).strip()

    # 尝试直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 提取第一个完整 JSON 对象（处理前后有多余文字的情况）
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _llm_driven_mode(
    sample: dict,
    llm_port: int,
    max_rounds: int,
) -> Generator[dict, None, None]:
    """
    JSON 行动协议驱动的多轮工具调用循环。

    不使用 OpenAI tools/tool_choice 参数，而是通过系统提示约定 LLM 输出固定 JSON 格式：
      调用工具：{"action": "call_tool", "tool": "<slug>", "args": {...}}
      完成分析：{"action": "finish", "text": "<中文总结>"}

    这样无需 vLLM --enable-auto-tool-choice 标志，兼容任意已运行的 vLLM 实例。
    """
    try:
        from openai import OpenAI
    except ImportError:
        yield {"type": "error", "message": "openai package not installed. Run: pip install openai"}
        return

    from .tool_bridge import get_expected_tool_schemas, execute_tool
    from .docker_utils import perception_manager, get_llm_model_name, TOOL_DOCKER_FULL_MAP

    tool_calls_in_sample = sample.get("tool_calls", [])
    schemas = get_expected_tool_schemas(tool_calls_in_sample)

    if not schemas:
        yield {"type": "error", "message": "No tool schemas available for this sample"}
        return

    tool_text = _build_tool_prompt(schemas)

    # 构建工具调用顺序提示（严格顺序，参数由 LLM 自决）
    tool_order_lines = [
        f"{i}. {tc['tool']}"
        for i, tc in enumerate(tool_calls_in_sample, 1)
    ]
    tool_order_text = "\n".join(tool_order_lines)

    system_prompt = f"""你是专业的地理空间 AI 助手，擅长灾害遥感分析。
根据用户的任务，严格按照指定顺序依次调用工具完成分析。

==== 工具调用顺序（必须严格按此顺序，不可跳过或改变）====
{tool_order_text}

==== 可用工具详细说明（* 表示必填参数）====
{tool_text}

==== 文件路径规则 ====
- 用户消息中 [影像文件] 块提供了实际可用的影像路径，工具参数中的图像路径必须使用这些真实路径
- 中间输出文件（如索引图、掩膜等）请写入 /tmp/ 目录，例如 /tmp/ndwi.tif、/tmp/mask.tif
- 不要使用任何中文字符或占位符作为文件路径

==== VLM 输出长度规则 ====
- geo_perception.vlm_analyze 的 max_tokens 是输出长度上限，不是图像输入上下文长度
- 对含图像的分析、bbox 提取、多图对比、灾害范围/面积分析，请默认使用 max_tokens 8192
- 不要给 geo_perception.vlm_analyze 使用 256 或 512 这类过小值

==== 输出规则（极其重要）====
每次只输出一个 JSON 对象，不要有任何其他文字或解释：

调用工具时：
{{"action": "call_tool", "tool": "工具slug", "args": {{"参数名": "参数值"}}}}

所有工具都调用完毕后给出总结：
{{"action": "finish", "text": "中文分析总结"}}

每次工具调用结果会以 [TOOL_RESULT] ... [/TOOL_RESULT] 形式返回给你，请据此继续分析。"""

    client = OpenAI(
        base_url=f"http://127.0.0.1:{llm_port}/v1",
        api_key="token-abc",
        http_client=_make_http_client(),
    )
    model_name = get_llm_model_name(llm_port)

    # 将实际影像路径注入 prompt
    sample_id = sample.get("id", "")
    image_ctx = _get_image_context(sample_id)
    user_content = sample["prompt"]
    if image_ctx:
        user_content = user_content + "\n\n" + image_ctx

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    yield {"type": "llm_start"}

    tool_calls_made: list[str] = []
    actual_tool_calls: list[dict] = []   # 记录实际执行的工具调用（slug + 实际参数 + 实际输出）
    step = 0
    step_results: dict[int, dict] = {}
    parse_fail_count = 0

    for round_idx in range(max_rounds):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                # 禁用 Qwen3 思考模式，确保输出简洁 JSON
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            # extra_body 不支持时降级重试
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )
            except Exception as e2:
                yield {"type": "error", "message": f"LLM call failed: {e2}"}
                return

        raw_content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": raw_content})

        action = _parse_json_action(raw_content)

        if action is None:
            parse_fail_count += 1
            if parse_fail_count >= 3:
                yield {"type": "error", "message": f"LLM 连续 3 次未输出有效 JSON，最后输出：{raw_content[:200]}"}
                break
            messages.append({
                "role": "user",
                "content": '请只输出一个 JSON 对象，格式：{"action": "call_tool", "tool": "...", "args": {...}} 或 {"action": "finish", "text": "..."}',
            })
            continue

        parse_fail_count = 0  # 重置计数

        act = action.get("action", "")

        if act == "finish":
            final_text = action.get("text", raw_content)
            yield {"type": "llm_text", "text": final_text}
            break

        elif act == "call_tool":
            step += 1
            slug = action.get("tool", "").strip()
            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}

            # 解析 $stepN.field 引用
            args = _resolve_refs(args, step_results)

            yield {
                "type": "tool_start",
                "step": step,
                "slug": slug,
                "fn_name": slug,
                "args": args,
            }

            # 感知类工具：通过 LRU 管理器按需启动
            if slug in TOOL_DOCKER_FULL_MAP:
                container_name = TOOL_DOCKER_FULL_MAP[slug][0]
                yield {"type": "service_starting", "slug": slug, "container": container_name}
                svc_ok, svc_msg, was_started = perception_manager.acquire(slug)
                if not svc_ok:
                    yield {"type": "service_start_failed", "slug": slug, "container": container_name, "message": svc_msg}
                    fallback = _find_sample_output(tool_calls_in_sample, slug)
                    step_results[step] = fallback
                    result_str = json.dumps(fallback, ensure_ascii=False, default=str)
                    messages.append({"role": "user", "content": f"[TOOL_RESULT] {result_str} [/TOOL_RESULT]"})
                    yield {
                        "type": "tool_result", "step": step, "slug": slug,
                        "output_type": "dict", "display_value": fallback, "raw": fallback,
                        "note": "⚠️ Service unavailable — using sample_output from dataset",
                    }
                    actual_tool_calls.append({"step": step, "tool": slug, "args": args, "sample_output": fallback})
                    tool_calls_made.append(slug)
                    continue
                if was_started:
                    yield {"type": "service_started", "slug": slug, "container": container_name, "message": svc_msg}

            result = execute_tool(slug, args)

            # 刷新 LRU 时间
            if slug in TOOL_DOCKER_FULL_MAP:
                perception_manager.touch(slug)

            actual_output = result.get("raw") or {}
            step_results[step] = actual_output

            raw_for_llm = result.get("raw")
            if isinstance(raw_for_llm, dict):
                result_str = json.dumps(raw_for_llm, ensure_ascii=False, default=str)
            elif raw_for_llm is None:
                result_str = "{}"
            else:
                result_str = str(raw_for_llm)

            # 截断过长的结果（避免 context 爆炸）
            if len(result_str) > 800:
                result_str = result_str[:800] + "...(truncated)"

            messages.append({
                "role": "user",
                "content": f"[TOOL_RESULT] {result_str} [/TOOL_RESULT]",
            })

            if result["output_type"] == "error":
                yield {
                    "type": "tool_error",
                    "step": step,
                    "slug": slug,
                    "error": result["display_value"],
                }
            else:
                yield {
                    "type": "tool_result",
                    "step": step,
                    "slug": slug,
                    "output_type": result["output_type"],
                    "display_value": result["display_value"],
                    "raw": result["raw"],
                }
            actual_tool_calls.append({"step": step, "tool": slug, "args": args, "sample_output": actual_output})
            tool_calls_made.append(slug)

        else:
            # 未知 action
            messages.append({
                "role": "user",
                "content": 'action 字段无效，请输出 "call_tool" 或 "finish"。',
            })

    else:
        yield {"type": "error", "message": f"Exceeded max_rounds ({max_rounds}) without completion"}

    # 容器由 perception_manager 统一管理（LRU + 5min 超时），此处不手动停止
    yield {"type": "done", "tool_calls_made": tool_calls_made, "actual_tool_calls": actual_tool_calls}


def _find_sample_output(tool_calls: list[dict], slug: str) -> dict:
    """从 SFT 样本的 tool_calls 中找到对应 slug 的 sample_output（降级用）。"""
    for tc in tool_calls:
        if tc.get("tool") == slug:
            return tc.get("sample_output", {})
    return {}


def _make_http_client():
    """创建绕过代理的 httpx 客户端。

    httpx ≥ 0.28 移除了 proxies 参数，改用 mounts={url: None} 表示直连。
    """
    try:
        import httpx
        return httpx.Client(
            mounts={
                "http://127.0.0.1": None,
                "http://localhost": None,
            },
            timeout=60.0,
        )
    except ImportError:
        return None
