"""
tool_bridge.py — CoreRegistry 加载 & 工具直接调用
================================================
不依赖 Terrabox FastAPI/DB，直接加载 CoreRegistry 全局单例并调用 handler。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# 将 src/ 加入路径，使 terrabox 包可被导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# 启用 Docker 模式：让 vlm_analyze / sam2 等工具使用 Docker manager
# 而非尝试启动本地子进程（子进程模式需要额外配置 VLLM_PYTHON_EXEC 等环境变量）
os.environ.setdefault("TERRABOX_USE_DOCKER", "true")

from terrabox.extensions import load_builtin_toolkits
from terrabox.core.registry import registry  # 全局单例

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 初始化（只需调用一次）
# ---------------------------------------------------------------------------
_initialized = False


def init_registry():
    """加载所有内置工具包到全局 registry。"""
    global _initialized
    if _initialized:
        return
    load_builtin_toolkits()
    _initialized = True
    logger.info(f"Registry initialized: {len(registry.list_tools())} tools loaded.")


# ---------------------------------------------------------------------------
# OpenAI Function Calling Schema
# ---------------------------------------------------------------------------

def _slug_to_fn_name(slug: str) -> str:
    """geo_raster.calculate_index → geo_raster__calculate_index"""
    return slug.replace(".", "__", 1)


def _fn_name_to_slug(fn_name: str) -> str:
    """geo_raster__calculate_index → geo_raster.calculate_index"""
    return fn_name.replace("__", ".", 1)


def get_tool_schemas(slugs: list[str] | None = None) -> list[dict]:
    """
    将 ToolSpec 列表转换为 OpenAI function calling 格式。

    Parameters
    ----------
    slugs : list[str] | None
        若指定则只导出这些 slug；否则导出全部工具。

    Returns
    -------
    list[dict]
        OpenAI tools 参数格式：[{"type": "function", "function": {...}}, ...]
    """
    init_registry()
    specs = registry.list_tools()
    if slugs:
        slug_set = set(slugs)
        specs = [s for s in specs if s.slug in slug_set]

    tools = []
    for spec in specs:
        fn_name = _slug_to_fn_name(spec.slug)
        # ToolSpec.parameters 已经是 JSON Schema object，直接用
        params = spec.parameters if isinstance(spec.parameters, dict) else {}
        tools.append({
            "type": "function",
            "function": {
                "name": fn_name,
                "description": spec.description,
                "parameters": params,
            },
        })
    return tools


def get_expected_tool_schemas(tool_calls: list[dict]) -> list[dict]:
    """根据 SFT 样本中的 tool_calls 提取所需 slug，只返回这些工具的 schema。"""
    slugs = list({tc["tool"] for tc in tool_calls})
    return get_tool_schemas(slugs)


# ---------------------------------------------------------------------------
# 工具执行
# ---------------------------------------------------------------------------

# Docker 依赖工具（这些工具需要对应服务运行）
DOCKER_DEPENDENT_TOOLS = {
    "geo_perception.vlm_analyze",
    "geo_perception.remotesam",
    "geo_perception.sam2_segment",
    "geo_perception.strip_rcnn_detect",
    "geo_perception.remoteclip_analysis",
    "geo_perception.instructsam",
}


def _detect_output_type(result: Any) -> tuple[str, Any]:
    """
    检测工具输出类型，返回 (type, value)。

    type 可以是：
    - "image_path": result 包含指向图像文件的路径
    - "image_array": result 是 numpy 数组
    - "text": result 是字符串
    - "dict": result 是 dict/list
    """
    try:
        import numpy as np
        if isinstance(result, np.ndarray):
            return "image_array", result
    except ImportError:
        pass

    try:
        from PIL import Image
        if isinstance(result, Image.Image):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
            result.save(tmp.name)
            return "image_path", tmp.name
    except ImportError:
        pass

    if isinstance(result, dict):
        # 检查是否包含图像路径字段
        for key in ("output_path", "image_path", "path", "result_path"):
            val = result.get(key)
            if val and isinstance(val, str) and _is_image_file(val):
                return "image_path", val
        return "dict", result

    if isinstance(result, str):
        if _is_image_file(result):
            return "image_path", result
        return "text", result

    return "dict", result


def _is_image_file(path: str) -> bool:
    """判断路径是否指向图像文件（不要求文件存在）。"""
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _raster_to_png(tif_path: str) -> str | None:
    """将 GeoTIFF 转换为 PNG，返回临时文件路径。"""
    try:
        import numpy as np
        import rasterio
        from rasterio.plot import reshape_as_image

        with rasterio.open(tif_path) as src:
            data = src.read()  # (bands, H, W)

        if data.shape[0] == 1:
            # 单波段 → 灰度归一化
            band = data[0].astype(float)
            vmin, vmax = band.min(), band.max()
            if vmax > vmin:
                band = ((band - vmin) / (vmax - vmin) * 255).astype("uint8")
            else:
                band = np.zeros_like(band, dtype="uint8")
            from PIL import Image
            img = Image.fromarray(band, mode="L").convert("RGB")
        else:
            # 多波段：取前三波段作为 RGB
            rgb = data[:3]
            rgb = rgb.astype(float)
            for i in range(rgb.shape[0]):
                ch = rgb[i]
                lo, hi = ch.min(), ch.max()
                if hi > lo:
                    rgb[i] = (ch - lo) / (hi - lo) * 255
            rgb = rgb.astype("uint8")
            from PIL import Image
            img = Image.fromarray(reshape_as_image(rgb))

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
        img.save(tmp.name)
        return tmp.name
    except Exception as e:
        logger.warning(f"Failed to convert raster to PNG: {e}")
        return None


def execute_tool(slug: str, args: dict) -> dict:
    """
    执行一个工具，返回结构化结果。

    Returns
    -------
    dict with keys:
        - raw: 原始 handler 返回值
        - output_type: "image_path" | "image_array" | "text" | "dict" | "error"
        - display_value: 用于展示的值（图像路径、格式化文本等）
        - error: 错误信息（仅在 output_type=="error" 时）
    """
    init_registry()

    handler = registry.get_handler(slug)
    if handler is None:
        return {
            "raw": None,
            "output_type": "error",
            "display_value": f"Tool not found: {slug}",
            "error": f"Tool not found: {slug}",
        }

    try:
        raw = handler(args, context=None, account=None)
    except TypeError:
        # 有些 handler 只接受 arguments 一个参数
        try:
            raw = handler(args)
        except Exception as e:
            return {
                "raw": None,
                "output_type": "error",
                "display_value": str(e),
                "error": str(e),
            }
    except Exception as e:
        logger.exception(f"Tool {slug} raised exception")
        return {
            "raw": None,
            "output_type": "error",
            "display_value": str(e),
            "error": str(e),
        }

    output_type, value = _detect_output_type(raw)

    # GeoTIFF → PNG 转换
    if output_type == "image_path" and value and Path(value).suffix.lower() in {".tif", ".tiff"}:
        if Path(value).exists():
            png_path = _raster_to_png(value)
            if png_path:
                value = png_path
            else:
                # 转换失败，降级为文本
                output_type = "dict"
                value = raw

    # numpy array → PNG
    if output_type == "image_array":
        try:
            import numpy as np
            from PIL import Image
            arr = value.astype(float)
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = ((arr - lo) / (hi - lo) * 255).astype("uint8")
            else:
                arr = np.zeros_like(arr, dtype="uint8")
            img = Image.fromarray(arr if arr.ndim == 2 else arr[:, :, :3])
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
            img.save(tmp.name)
            output_type = "image_path"
            value = tmp.name
        except Exception as e:
            logger.warning(f"Array to PNG failed: {e}")
            output_type = "dict"
            value = {"array_shape": str(value.shape) if hasattr(value, "shape") else str(type(value))}

    return {
        "raw": raw,
        "output_type": output_type,
        "display_value": value,
    }


def list_all_slugs() -> list[str]:
    """返回所有已注册工具的 slug 列表。"""
    init_registry()
    return [s.slug for s in registry.list_tools()]
