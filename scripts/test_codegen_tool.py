"""
测试 codegen.generate_and_run 工具。

用法：
    # 确保 Docker LLM 已启动（端口 9100），然后：
    conda run -n terra env PYTHONPATH=src no_proxy=localhost,127.0.0.1 \\
        python scripts/test_codegen_tool.py

输出：LLM 生成的代码 + Docker 执行结果
"""
import json
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from terrabox.toolkits.codegen import codegen_generate_and_run_handler

# ── 测试用例：多灾种复合风险叠加 ──────────────────────────────────────────────
DESCRIPTION = """
给定三个0-1归一化的风险矩阵 flood_risk、quake_risk、fire_risk（均为二维列表），
生成复合风险图：标出同时面临2种以上高风险（值>0.6）的像素（标记为1，其余为0），
并统计高风险像素占总像素数的百分比。

输入格式：{"flood_risk": [[...]], "quake_risk": [[...]], "fire_risk": [[...]]}
输出格式：{"compound_risk_map": [[...]], "high_risk_pct": <float>}
""".strip()

INPUT_DATA = json.dumps({
    "flood_risk": [[0.8, 0.3], [0.9, 0.1]],
    "quake_risk": [[0.7, 0.2], [0.4, 0.8]],
    "fire_risk":  [[0.2, 0.9], [0.6, 0.7]],
})

# ── 运行 ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PROMPT (description):")
print(DESCRIPTION)
print()
print("INPUT_DATA:")
print(json.dumps(json.loads(INPUT_DATA), indent=2))
print("=" * 60)

result = codegen_generate_and_run_handler(
    arguments={"description": DESCRIPTION, "input_data": INPUT_DATA},
    context={},
)

print()
print("=" * 60)
print(f"SUCCESS : {result['success']}")
print(f"ATTEMPTS: {result.get('attempts', '?')}")
print()

if result.get("code"):
    print("── LLM 生成的代码 ──────────────────────────────────────")
    print(result["code"])
    print()

if result["success"]:
    print("── 执行结果 ─────────────────────────────────────────────")
    output = result["output"]
    try:
        parsed = json.loads(output)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(output)
else:
    print("── 错误信息 ─────────────────────────────────────────────")
    print(result.get("error", "unknown error"))

print("=" * 60)
