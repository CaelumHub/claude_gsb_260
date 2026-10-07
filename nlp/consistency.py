"""规范化前后的语义一致性校验。

规范化只允许改变 *形式*（字形、全半角、空白），不应改变 *内容*。
本模块用平台已有的分词 / 情感 / NER 等任务对「原文」与「规范文本」
各跑一遍，给出结构化对比，供回流前确认口径一致：

* 分词：比较归一化（繁→简、大小写、全半角）后 *内容 token 的多重集合*
  —— 繁简形态下词典覆盖率不同会导致切分边界差异（如「工程/師」vs
  「工程师」），多重集合一致即说明语义内容没有增减；
* 情感：极性标签必须相同，分值漂移不超过阈值；
* NER：实体文本归一化后的集合应一致（允许偏移变化），类型不一致
  会单独报告（繁简形态下规则 NER 可能出现 PERSON/LOCATION 抖动）；
* 长度 / 句数：给出形式层面的变化量作为参考。
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from . import charconv
from .langid import split_paragraphs
from .segmenter import Segmenter
from .sentiment import SentimentAnalyzer
from .ner import NERExtractor


def _canonical(token: str) -> str:
    """归一化一个 token 用于跨形式比较：繁→简、全角→半角、小写。"""
    out = []
    for ch in token:
        ch = charconv.to_simplified(ch)
        code = ord(ch)
        if code == 0x3000:
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        out.append(ch)
    return "".join(out).strip().lower()


def _content_tokens(words: list[str]) -> list[str]:
    return [c for w in words if (c := _canonical(w)) and not _is_punct(c)]


def _is_punct(s: str) -> bool:
    return all(not ch.isalnum() and not ("一" <= ch <= "鿿") for ch in s)


def _content_chars(text: str) -> str:
    """归一化为只含内容字符的序列：繁→简、全角→半角、小写、去标点空白。"""
    out = []
    for ch in text:
        c = _canonical(ch)
        if not c:
            continue
        if all(not x.isalnum() and not ("一" <= x <= "鿿") for x in c):
            continue
        out.append(c)
    return "".join(out)


def _merge_number_runs(tokens: list[str]) -> list[str]:
    """把连续数字 token 合并为一个（全角 1/9/9 与半角 199 应视为同一内容）。"""
    merged: list[str] = []
    for tok in tokens:
        if tok.isdigit() and merged and merged[-1].isdigit():
            merged[-1] += tok
        else:
            merged.append(tok)
    return merged


def verify_consistency(original: str, normalized: str,
                       *, segmenter: Optional[Segmenter] = None,
                       sentiment: Optional[SentimentAnalyzer] = None,
                       ner: Optional[NERExtractor] = None,
                       polarity_tol: float = 0.25) -> dict:
    """对比原文与规范文本在下游任务上的结果，返回一致性报告。"""
    segmenter = segmenter or Segmenter()
    checks: list[dict] = []

    # 1) 分词口径：核心判据是「内容字符序列」归一化后完全一致
    #    （形式规范化只改字形/标点/空白，不得增删内容字）；
    #    词边界因繁简词典覆盖率不同可能不同，仅作参考展示。
    words_before = _merge_number_runs(_content_tokens(segmenter.cut(original)))
    words_after = _merge_number_runs(_content_tokens(segmenter.cut(normalized)))
    chars_before = _content_chars(original)
    chars_after = _content_chars(normalized)
    seg_match = chars_before == chars_after
    bag_before, bag_after = Counter(words_before), Counter(words_after)
    checks.append({
        "task": "segment",
        "name": "分词",
        "consistent": seg_match,
        "detail": {
            "tokens_before": len(words_before),
            "tokens_after": len(words_after),
            "content_chars_before": len(chars_before),
            "content_chars_after": len(chars_after),
            "token_boundary_diff": {
                "missing": dict(bag_before - bag_after),
                "added": dict(bag_after - bag_before),
            },
            "note": "以归一化后的内容字符序列为一致性判据；繁简词典造成的切分边界差异仅作参考",
        },
    })

    # 2) 情感口径
    if sentiment is None:
        sentiment = SentimentAnalyzer()
    s_before = sentiment.analyze(original)
    s_after = sentiment.analyze(normalized)
    polarity_same = s_before.get("polarity") == s_after.get("polarity")
    score_drift = abs(float(s_before.get("score", 0)) - float(s_after.get("score", 0)))
    sent_ok = polarity_same and score_drift <= polarity_tol
    checks.append({
        "task": "sentiment",
        "name": "情感分析",
        "consistent": sent_ok,
        "detail": {
            "polarity_before": s_before.get("polarity"),
            "polarity_after": s_after.get("polarity"),
            "score_before": s_before.get("score"),
            "score_after": s_after.get("score"),
            "score_drift": round(score_drift, 4),
        },
    })

    # 3) NER 口径：实体文本集合须一致；类型不一致单独列出
    if ner is None:
        ner = NERExtractor()
    typed_before = {(_canonical(e["text"]), e["type"])
                    for e in ner.recognize(original)}
    typed_after = {(_canonical(e["text"]), e["type"])
                   for e in ner.recognize(normalized)}
    texts_before = {t for t, _ in typed_before}
    texts_after = {t for t, _ in typed_after}
    text_only_before = texts_before - texts_after
    text_only_after = texts_after - texts_before
    types_before = {t: ty for t, ty in typed_before}
    types_after = {t: ty for t, ty in typed_after}
    type_mismatch = sorted(
        f"{t}: {types_before[t]} → {types_after[t]}"
        for t in texts_before & texts_after
        if types_before[t] != types_after[t])
    ner_ok = not text_only_before and not text_only_after
    checks.append({
        "task": "ner",
        "name": "命名实体",
        "consistent": ner_ok,
        "detail": {
            "entities_before": len(texts_before),
            "entities_after": len(texts_after),
            "only_before": sorted(text_only_before),
            "only_after": sorted(text_only_after),
            "type_mismatch": type_mismatch,
        },
    })

    # 4) 形式层面参考量（不影响一致性判定）
    checks.append({
        "task": "form",
        "name": "形式变化（参考）",
        "consistent": True,
        "detail": {
            "chars_before": len(original),
            "chars_after": len(normalized),
            "paragraphs_before": len(split_paragraphs(original)),
            "paragraphs_after": len(split_paragraphs(normalized)),
        },
    })

    return {
        "consistent": all(c["consistent"] for c in checks),
        "checks": checks,
    }
