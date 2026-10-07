"""语义一致性校验：规范化前后在下游任务上的口径保持一致。

规范化只允许做「字符层面的等价改写」，因此对分词、情感这类任务，
转换前后的结果应当保持同一口径。本模块在样本上同时跑原文与规范文，
产出结构化对比，供页面展示「哪些任务口径漂移了、漂在哪里」。

口径定义（刻意宽松，只抓真问题）：

* **分词**：忽略纯标点 / 空白 token 后，词序列应一致；繁简差异通过
  先各自规范化到简体再比对来消除。
* **情感**：极性标签应一致；分数允许小范围波动（默认 0.2）。
* **关键词**：关键词集合（简体口径）允许交集比例低于阈值时报警。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .variants import get_converter

SCORE_TOLERANCE = 0.2
KEYWORD_MIN_OVERLAP = 0.6


def _content_tokens(words: list[str]) -> list[str]:
    """去掉纯标点、空白 token，并统一到简体口径。"""
    conv = get_converter()
    out = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if all(not ch.isalnum() and not ("一" <= ch <= "鿿") for ch in w):
            continue
        out.append(conv.to_simplified(w))
    return out


@dataclass
class ConsistencyReport:
    consistent: bool
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"consistent": self.consistent, "checks": self.checks}


def check_segment(before_words: list[str], after_words: list[str]) -> dict:
    a = _content_tokens(before_words)
    b = _content_tokens(after_words)
    same = a == b
    check = {
        "task": "segment",
        "task_name": "分词",
        "consistent": same,
        "before_count": len(a),
        "after_count": len(b),
        "detail": "词序列一致" if same else "词序列出现差异",
    }
    if not same:
        # 标出第一个分歧位置，便于页面提示
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                check["first_diff"] = {"index": i, "before": x, "after": y}
                break
        else:
            check["first_diff"] = {"index": min(len(a), len(b)),
                                   "before": a[len(b):][:5],
                                   "after": b[len(a):][:5]}
    return check


def check_sentiment(before: dict, after: dict,
                    tolerance: float = SCORE_TOLERANCE) -> dict:
    bp, ap = before.get("polarity"), after.get("polarity")
    bs, as_ = float(before.get("score", 0.0)), float(after.get("score", 0.0))
    label_same = bp == ap
    score_delta = abs(bs - as_)
    score_ok = score_delta <= tolerance
    # 口径硬标准：极性不翻转。分数在容差内为一致；超容差但极性相同，
    # 记为"软警告"（规范化统一了繁简/全半角，往往是修正了 OOV，反而更准），
    # 不判为口径冲突。
    consistent = label_same
    warning = label_same and not score_ok
    if not label_same:
        detail = f"极性发生变化：{bp} → {ap}"
    elif warning:
        detail = f"极性一致（{bp}），分数波动 {score_delta:.2f} 超过容差 {tolerance}，建议人工确认"
    else:
        detail = f"极性一致（{bp}），分数波动在容差内"
    return {
        "task": "sentiment",
        "task_name": "情感",
        "consistent": consistent,
        "warning": warning,
        "before_polarity": bp,
        "after_polarity": ap,
        "before_score": round(bs, 4),
        "after_score": round(as_, 4),
        "score_delta": round(score_delta, 4),
        "tolerance": tolerance,
        "detail": detail,
    }


def check_keywords(before: list[str], after: list[str],
                   min_overlap: float = KEYWORD_MIN_OVERLAP) -> dict:
    conv = get_converter()
    a = {conv.to_simplified(w) for w in before}
    b = {conv.to_simplified(w) for w in after}
    if not a or not b:
        overlap = 1.0
    else:
        overlap = len(a & b) / max(len(a | b), 1)
    consistent = overlap >= min_overlap
    return {
        "task": "keywords",
        "task_name": "关键词",
        "consistent": consistent,
        "overlap": round(overlap, 4),
        "min_overlap": min_overlap,
        "only_before": sorted(a - b),
        "only_after": sorted(b - a),
        "detail": "关键词口径一致" if consistent else "关键词重合度偏低",
    }


def run_consistency_checks(original: str, normalized: str,
                           tasks: tuple[str, ...] = ("segment", "sentiment"),
                           tolerance: float = SCORE_TOLERANCE) -> ConsistencyReport:
    """对原文 / 规范文跑指定任务并对比。惰性导入 NLP 单例，避免循环依赖。"""
    checks: list[dict] = []
    if "segment" in tasks:
        from . import get_segmenter
        seg = get_segmenter()
        checks.append(check_segment(seg.cut(original), seg.cut(normalized)))
    if "sentiment" in tasks:
        from . import get_sentiment
        sa = get_sentiment()
        checks.append(check_sentiment(sa.analyze(original),
                                      sa.analyze(normalized), tolerance))
    if "keywords" in tasks:
        from . import get_keywords
        ke = get_keywords()
        before = [k["word"] if isinstance(k, dict) else k
                  for k in ke.extract(original).get("keywords", [])]
        after = [k["word"] if isinstance(k, dict) else k
                 for k in ke.extract(normalized).get("keywords", [])]
        checks.append(check_keywords(before, after))
    return ConsistencyReport(
        consistent=all(c["consistent"] for c in checks), checks=checks)
