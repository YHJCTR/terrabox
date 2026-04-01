"""
用 Claude Haiku API 对 disaster SFT 数据集中每条样本生成 N 条 prompt 变体。

输入  : data/disaster_sft_dataset.json  (90 条)
输出  : data/disaster_sft_augmented.json (~450 条，含原始 90 条)

用法:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python scripts/augment_sft_prompts.py

可选参数:
  --n-variants    每条样本生成几个变体 (默认 4)
  --only-passed   只对流程验证通过的样本扩增（需先运行 test_sft_flows.py）
  --resume        跳过已处理的样本（断点续跑）
  --batch-size    并发 API 请求数（默认 5）
  --dry-run       只打印 prompt，不调用 API
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT_PATH    = ROOT / "data" / "disaster_sft_dataset.json"
REPORT_PATH = ROOT / "data" / "flow_test_report.json"
AUG_OUT     = ROOT / "data" / "disaster_sft_augmented.json"
CACHE_PATH  = ROOT / "data" / ".augment_cache.json"  # 断点续跑缓存

SYSTEM_PROMPT = """你是遥感灾害分析领域专家和数据标注师。
给你一个灾害分析任务描述，请生成若干条语义等价但表达不同的改写版本。

改写要求：
1. 保留所有技术要素（灾害类型、遥感数据类型、分析目标、输出需求）
2. 改变表达视角：可以是操作员视角、应急管理者视角、数据分析师视角
3. 改变语气：可以是简短指令、详细叙述、疑问形式、报告风格
4. 保持中文，但可以包含必要的英文技术术语
5. 每条改写不超过 150 字
6. 仅返回 JSON 数组，不要其他内容

输出格式示例（生成 3 条）：
["改写版本1", "改写版本2", "改写版本3"]"""


def load_passed_ids(report_path: Path) -> set[str] | None:
    """从测试报告加载通过验证的样本 ID"""
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    fv = report.get("flow_validation", {})
    return {k for k, v in fv.items() if isinstance(v, dict) and v.get("passed", False)}


def call_claude_api(client, sample: dict, n_variants: int, dry_run: bool) -> list[str]:
    """调用 Claude Haiku API 生成 prompt 变体"""
    task_type = sample["task_type"]
    disaster  = sample["disaster_category"]
    prompt    = sample["prompt"]

    user_content = (
        f"任务类型：{task_type}（{disaster}）\n"
        f"原始描述：{prompt}\n\n"
        f"请生成 {n_variants} 条改写版本，输出 JSON 数组。"
    )

    if dry_run:
        print(f"  [DRY-RUN] {sample['id']}: {user_content[:80]}...")
        return [f"[dry-run variant {i+1} of {sample['id']}]" for i in range(n_variants)]

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = response.content[0].text.strip()
        # Parse JSON array from response
        # Sometimes the model wraps it in markdown code blocks
        text = re.sub(r"^```json\s*|^```\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        variants = json.loads(text)
        if not isinstance(variants, list):
            raise ValueError(f"Expected list, got {type(variants)}")
        return [str(v).strip() for v in variants[:n_variants]]
    except Exception as e:
        print(f"  [WARN] API error for {sample['id']}: {e}")
        return []


def make_augmented_sample(original: dict, new_prompt: str, variant_idx: int) -> dict:
    """基于原始样本和新 prompt 生成扩增样本"""
    return {
        "id": f"{original['id']}_aug{variant_idx}",
        "task_type": original["task_type"],
        "disaster_category": original["disaster_category"],
        "difficulty": original["difficulty"],
        "prompt": new_prompt,
        "tool_calls": original["tool_calls"],  # 完全不变
        "_augmented": True,
        "_source_id": original["id"],
    }


def main():
    parser = argparse.ArgumentParser(description="Augment SFT prompts with Claude API")
    parser.add_argument("--n-variants",  type=int, default=4,    help="每条样本生成变体数（默认 4）")
    parser.add_argument("--only-passed", action="store_true",    help="仅扩增流程验证通过的样本")
    parser.add_argument("--resume",      action="store_true",    help="断点续跑（跳过已处理样本）")
    parser.add_argument("--batch-size",  type=int, default=5,    help="并发请求数（默认 5）")
    parser.add_argument("--dry-run",     action="store_true",    help="不调用 API，只打印")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] 请设置环境变量 ANTHROPIC_API_KEY")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    # Load dataset
    print(f"读取数据集: {SFT_PATH}")
    data = json.loads(SFT_PATH.read_text())
    samples = data["samples"]
    print(f"  原始样本: {len(samples)} 条")

    # Filter by passed validation if requested
    if args.only_passed:
        passed_ids = load_passed_ids(REPORT_PATH)
        if passed_ids:
            samples = [s for s in samples if s["id"] in passed_ids]
            print(f"  过滤后 (仅通过验证): {len(samples)} 条")
        else:
            print("  [warn] 未找到测试报告，使用全部样本")

    # Load resume cache
    cache: dict[str, list[str]] = {}
    if args.resume and CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        print(f"  断点续跑，已缓存 {len(cache)} 条")

    # Initialize API client
    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    # Generate variants
    all_samples = list(data["samples"])  # start with original 90
    new_count = 0
    total = len(samples)

    print(f"\n开始生成变体 (每条 {args.n_variants} 个)...")
    for i, sample in enumerate(samples):
        sid = sample["id"]

        # Check cache
        if sid in cache:
            variants = cache[sid]
            print(f"  [{i+1}/{total}] {sid}: 命中缓存 ({len(variants)} 条)")
        else:
            print(f"  [{i+1}/{total}] {sid}: 调用 API...", end="", flush=True)
            variants = call_claude_api(client, sample, args.n_variants, args.dry_run)
            print(f" 生成 {len(variants)} 条")
            cache[sid] = variants

            # Save cache after each successful call
            if not args.dry_run:
                CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

            # Rate limiting
            if not args.dry_run and (i + 1) % args.batch_size == 0:
                time.sleep(1.0)

        # Create augmented samples
        for j, new_prompt in enumerate(variants, start=1):
            aug_sample = make_augmented_sample(sample, new_prompt, j)
            all_samples.append(aug_sample)
            new_count += 1

    # Build output
    output = {
        "version": "1.1",
        "description": "灾害救援遥感分析 SFT 数据集（扩增版）",
        "original_samples": len(data["samples"]),
        "augmented_samples": new_count,
        "total_samples": len(all_samples),
        "n_variants_per_sample": args.n_variants,
        "augmentation_model": "claude-haiku-4-5-20251001",
        "task_types": sorted(set(s["task_type"] for s in all_samples)),
        "disaster_categories": sorted(set(s["disaster_category"] for s in all_samples)),
        "samples": all_samples,
    }

    AUG_OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n[OK] 写入 {AUG_OUT}")
    print(f"     原始: {output['original_samples']} 条")
    print(f"     新增: {output['augmented_samples']} 条")
    print(f"     总计: {output['total_samples']} 条")

    # Cleanup cache if not dry run and completed
    if not args.dry_run and not args.resume:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()


if __name__ == "__main__":
    main()
