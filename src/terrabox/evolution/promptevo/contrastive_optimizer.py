"""对比式优化器(领域无关):看两版提示词 + 两版配对轨迹 + 指标差,
做跨版本对比归因(credit assignment),产出下一版候选。

降方差:多批配对各诊断一次,聚合反复出现的归因;再 best-of-N 出候选。
LLM 通过 interfaces.LLMClient 注入,核心不绑定 terrabox。
"""
from __future__ import annotations

from typing import Optional

from .interfaces import LLMClient
from .schemas import PairedCase, Attribution

_SYSTEM = (
    "You are a prompt-protocol engineer. Given TWO versions of an agent's static "
    "system prompt and paired execution traces (same task under each version) plus "
    "per-dimension metric deltas, you attribute which prompt edits helped or hurt "
    "which dimensions, then propose the next version. Rules must stay GENERAL and "
    "domain-agnostic; keep changes minimal."
)

# 默认优化目标(通用兜底,不预设要优化哪个具体指标);使用者可通过 objective 覆盖。
_DEFAULT_OBJECTIVE = (
    "综合提升整体表现:尽量让 higher_better 的指标上升、lower_better 的指标下降,"
    "同时**不让任何主要维度明显退步**(宁可小步前进,也不要拆东补西)。"
    "哪个指标更重要,依据各指标简介与其方向自行权衡,不预设单一目标。"
)

_DIAGNOSE = """对比新旧两版静态系统提示词在**同一批任务**上的行为差异,做归因。

================ 当前版本(B,完整) ================
{prompt_b}

================ B 相对 A 的**真实改动清单**(代码精确算出,仅此几处,不要在此之外臆造) ================
{real_diff}

================ 指标变化(A -> B;每个指标附简介,含其方向) ================
{metric_summary}
注意:**不同指标好坏方向不同**(有的越高越好、有的越低越好)。请**根据每个指标的简介自行判断**改善还是退步,不要假设数值变大或变小就一定是好或坏。

================ 优化目标(仅作判断 help/hurt 的背景,勿为迎合它而编造证据) ================
{objective}

================ 配对轨迹(同一 task 在 A / B 下的走法) ================
{cases}

任务:对每个退步,做**三步链式归因**(像反向传播,从指标穿过"行为"反传到"提示词子句"):
  (1) **行为**:**自己仔细读 B 的轨迹**,指出 B 反复出现的反常/浪费/低效行为(例如:反复调用同一工具、为凑更精确结果而反复重查、报错后仍硬调、动作远多于 A 却没进展…由你从轨迹观察归纳,不限于这些例子);
  (2) **子句定位(关键)**:在【真实改动清单】对应的那条规则里,**逐字引用是哪一个子句/短语指示或允许了这个行为**——这才是"原因在提示词文本的哪里";
  (3) **指标**:这个行为对应哪个指标退步。
约束:**只针对真实改动清单里的改动**(用 edit_id);**严禁归因到清单之外**;无法由这几条改动解释的指标变化放进 unexplained,不要编造。同一条改动可能**双刃**(help A + hurt B),如实标 mixed。
严格 JSON:
{{"attributions":[{{"edit_id":"edit-1","behavior":"B 的问题行为(来自行为事实)",
  "offending_clause":"该规则里肇事的子句/短语(逐字引用)","effect":"help|hurt|mixed",
  "dims":["受影响指标名,如 f1_gis/turn_cap_rate/success_rate"],"evidence":"支撑的任务/现象"}}],
  "unexplained":["无法归因到上述真实改动的指标变化(可空)"]}}
"""

_PROPOSE = """基于下面对两版提示词的**对比归因**,产出**下一版**静态系统提示词。

================ 优化目标(你出新版必须朝它走) ================
{objective}

================ 当前最新版(B) ================
{prompt_b}

================ 归因结论(哪条改动在哪个维度有益/有害) ================
{attributions}

要求:
1. 一切改动都要服务于上面的**优化目标**:**保留**被判 help 的改动;**回退或收窄**被判 hurt 的改动。
2. **子句级因果分解(关键)**:被判 hurt / mixed 的规则往往**整条很长、含多个子句**,但真正肇事的常常只是其中**一个子句/短语**。对每条 hurt/mixed 规则,先逐子句想清楚:**agent 读到这个子句会实际做出什么动作?** 尤其注意那些**指示或暗示 agent 去"多做一个动作"的子句**——例如"再发起一次调用 / 重试 / 换个方式重查 / 为再确认而重复操作"。这类"无界额外动作"的子句,正是**过度调用、来回打转、耗尽步数**这类 hurt 的常见根因。**精确定位到那个肇事子句,只改它**,别动同一规则里无辜的其它部分。
3. 定位后的改法:把**无界的额外动作**改成**有界/有条件**——例如"只在确有必要时再做一次,且不得仅为再确认/凑更精确结果而重复";**双刃规则(同时 help A、hurt B)改成条件规则**(在 help 的场景保留、只在 hurt 的场景退让),不要因为它在某些场景 hurt 就整条删除或全局弱化。**但若某处改动被判为纯 hurt、找不到任何 help 场景,则可以直接回退/删除它**(与要求1一致)。
4. 规则保持**通用、领域无关**(条件用"是否确有必要""能否用现有结果"这类通用判据,**不要写死任何工具名/领域术语/任务**);保持克制、最小改动、不要显著变长。
5. 严格 JSON。**changes 必须逐字引用你实际改动的原文短语→新短语**(便于核对你确实改了,而非空泛声称):
{{"revised_prompt":"完整下一版提示词","rationale":"一句话",
  "changes":[{{"edit_id":"针对哪条归因","offending_clause":"你判定为肇事的那个子句/短语",
    "old_phrase":"被改前的原文(逐字)","new_phrase":"改后的文本(逐字)","why":"为什么这半句导致该 hurt"}}]}}
"""


def build_emphasized_prompt(old_prompt: str, new_prompt: str,
                            changes: Optional[list] = None,
                            header: str = "## 需特别遵守（本版相对上一版改动的规则，请优先严格执行）") -> str:
    """在 new_prompt 末尾追加"需特别遵守"小节,提升新规则在 rollout 时的显著性(近因效应)。
    **领域无关、确定性**:用代码 diff 取出 new 相对 old 改动的"新规则那侧"(地面真值,不靠 LLM 自述);
    `changes` 里 LLM 的 why 只作注解,且**只能附在代码验证过的真实改动上**(锚定,编不出清单外的)。
    无改动则原样返回。供 rollout 端用 toggle 决定加不加。"""
    import difflib
    a = old_prompt.strip().splitlines()
    b = new_prompt.strip().splitlines()
    sm = difflib.SequenceMatcher(None, [l.strip() for l in a], [l.strip() for l in b])
    new_lines = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            new_lines += [l.strip() for l in b[j1:j2] if l.strip()]
    if not new_lines:
        return new_prompt
    whys = {}
    for ch in (changes or []):
        if isinstance(ch, dict):
            npz = (ch.get("new_phrase") or "").strip()
            why = (ch.get("why") or "").strip()
            if npz and why:
                whys[npz[:30]] = why
    lines = [new_prompt.strip(), "", header]
    for nl in new_lines:
        note = ""
        for k, w in whys.items():
            if k and k in nl:
                note = f"\n  （为什么重要：{w}）"
                break
        clean = nl[2:].strip() if nl.startswith("- ") else nl   # 规则本就带"- ",去掉避免双重项目符
        lines.append(f"- {clean}{note}")
    return "\n".join(lines)


def _compute_diff(prompt_a: str, prompt_b: str) -> str:
    """代码精确算出 A->B 的真实改动(按行/规则粒度),渲染成枚举清单喂给 LLM,杜绝它臆造改动。
    按行粒度:一句话里只改一个词,也会给出**整条规则**的旧→新(上下文完整),而非孤立单词。"""
    import difflib
    a = [l for l in prompt_a.strip().splitlines()]
    b = [l for l in prompt_b.strip().splitlines()]
    sm = difflib.SequenceMatcher(None, [l.strip() for l in a], [l.strip() for l in b])
    out, eid = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        eid += 1
        old = " / ".join(x.strip() for x in a[i1:i2]) or "(无)"
        new = " / ".join(x.strip() for x in b[j1:j2]) or "(删除)"
        out.append(f"[edit-{eid}] ({tag})\n  旧: {old}\n  新: {new}")
    return "\n\n".join(out) if out else "(两版完全相同,无改动)"


def _fmt_cases(cases: list[PairedCase], max_cases: int = 6) -> str:
    out = []
    for c in cases[:max_cases]:
        out.append(f"## task {c.task_id}  Δ={c.metric_delta}  退步维度={c.regressed_dims}\n"
                   f"[A]\n{c.a_render}\n[B]\n{c.b_render}")
    return "\n\n".join(out)


def _fmt_metric_summary(agg_a: dict, agg_b: dict, specs=None) -> str:
    """渲染'指标: A -> B (Δ) — 简介(方向)'。不假设正负好坏,方向交给简介+LLM 判断。"""
    desc = {s.name: (s.description, s.direction) for s in (specs or [])}
    keys = [k for k in agg_b if k not in ("n",)]
    rows = []
    for k in keys:
        a, b = agg_a.get(k), agg_b.get(k)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue
        d, direction = desc.get(k, ("", "unknown"))
        note = f"  — {d}" if d else ""
        if direction and direction != "unknown":
            note += f" [{direction}]"
        rows.append(f"  {k}: {a:.3f} -> {b:.3f} (Δ {b-a:+.3f}){note}")
    return "\n".join(rows)


class ContrastiveOptimizer:
    def __init__(self, llm: Optional[LLMClient] = None):
        if llm is None:
            from ..shared.llm_client import EvolutionLLMClient
            llm = EvolutionLLMClient()
        self.llm = llm

    def diagnose(self, prompt_a: str, prompt_b: str, agg_a: dict, agg_b: dict,
                 cases: list[PairedCase], n_batches: int = 2,
                 batch_size: int = 4, metric_specs=None,
                 objective: str = _DEFAULT_OBJECTIVE) -> list[Attribution]:
        """多批配对各诊断一次,聚合反复出现的归因(降方差)。
        metric_specs: 使用者提供的各指标简介(含方向),让 LLM 自判改善/退步。
        objective: 优化目标;默认通用兜底(不预设优化某个具体指标),使用者可覆盖。"""
        from collections import Counter
        tally: Counter = Counter()
        store: dict[str, Attribution] = {}
        metric_summary = _fmt_metric_summary(agg_a, agg_b, metric_specs)
        real_diff = _compute_diff(prompt_a, prompt_b)   # 代码算精确改动清单(hybrid:防臆造)
        batches = [cases[i:i + batch_size] for i in range(0, max(1, len(cases)), batch_size)][:n_batches] or [cases]
        for batch in batches:
            data = self.llm.call_json(
                _DIAGNOSE.format(prompt_b=prompt_b, real_diff=real_diff,
                                 metric_summary=metric_summary, objective=objective,
                                 cases=_fmt_cases(batch)),
                system=_SYSTEM, max_tokens=2500)
            if not isinstance(data, dict):
                continue
            # edit_id -> 真实改动文本,把归因映射回可读的真实改动
            id2text = {}
            for blk in real_diff.split("\n\n"):
                if blk.startswith("[edit-"):
                    eid = blk[1:blk.index("]")]
                    id2text[eid] = blk.replace("\n", " ")[:120]
            for a in data.get("attributions", []) or []:
                if not isinstance(a, dict):
                    continue
                eid = str(a.get("edit_id") or a.get("edit_ref", ""))
                ref = f"{eid}: {id2text.get(eid, '(清单外!)')}"
                clause = str(a.get("offending_clause", "")).strip()
                behavior = str(a.get("behavior", "")).strip()
                # 把"行为→肇事子句"并进 evidence,好让 propose 知道改哪条子句
                ev = (f"行为:{behavior} | 肇事子句:「{clause}」 | {a.get('evidence', '')}"
                      if clause or behavior else str(a.get("evidence", "")))
                key = (a.get("effect", ""), tuple(sorted(a.get("dims", []))))
                tally[key] += 1
                store.setdefault(str(key), Attribution(
                    edit_ref=ref, effect=str(a.get("effect", "")),
                    dims=list(a.get("dims", []) or []), evidence=ev))
        # 只保留出现≥1次(单批时)/多批时优先反复出现的
        ranked = sorted(store.values(), key=lambda at: -tally[(at.effect, tuple(sorted(at.dims)))])
        return ranked

    @staticmethod
    def _grounded(cand: dict, prompt_b: str) -> bool:
        """M3 自动门:候选是否"说到做到"——
        (a) revised 必须真的不同于 B;
        (b) changes 里每条 new_phrase 必须真出现在 revised 里(否则=假改动,如之前 edit-2)。"""
        rev = (cand.get("revised_prompt") or "").strip()
        if not rev or rev == prompt_b.strip():
            return False
        for ch in cand.get("changes", []) or []:
            if not isinstance(ch, dict):
                continue
            npz = (ch.get("new_phrase") or "").strip()
            if npz and npz[:40] not in rev:      # 声称改成的短语没出现 → 假改动
                return False
        return True

    def propose_candidates(self, prompt_b: str, attributions: list[Attribution],
                           n: int = 3, max_tokens: int = 3500,
                           objective: str = _DEFAULT_OBJECTIVE) -> list[dict]:
        """best-of-N:生成 n 个下一版候选,并用自动门过滤"说了没做"的假候选。
        objective: 优化目标;默认通用兜底,使用者可覆盖。"""
        attr_txt = "\n".join(
            f"- [{a.effect}] {a.edit_ref}  (维度 {a.dims}; 证据 {a.evidence[:160]})"
            for a in attributions) or "(无显著归因)"
        cands = []
        for _ in range(n):
            data = self.llm.call_json(
                _PROPOSE.format(prompt_b=prompt_b, attributions=attr_txt, objective=objective),
                system=_SYSTEM, max_tokens=max_tokens)
            if isinstance(data, dict) and data.get("revised_prompt"):
                cands.append(data)
        grounded = [c for c in cands if self._grounded(c, prompt_b)]
        # 全部没过门时,退回原始候选(至少有东西),并标记
        return grounded if grounded else cands
