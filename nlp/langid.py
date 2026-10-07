"""语言识别：主语言判定 + 混合文本按语言分段。

完全基于 Unicode 字符类别统计，无需任何模型或第三方库：

1. :func:`detect_language` 对整段文本给出主语言（zh / en / ja / ko /
   digits / symbols）及各语言字符占比、中文简繁倾向。
2. :func:`split_languages` 按连续同类字符切成「语言段」，调用方可以
   据此对中英混排文本逐段处理或着色展示。

设计上对短文本同样稳定：以有效字符（字母/汉字，忽略标点数字空白）为
分母，避免 "hello！" 这种文本被标点带偏。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .variants import get_converter

# ---------------------------------------------------------------------------
# 字符类别判定
# ---------------------------------------------------------------------------

# CJK 统一表意文字（含扩展 A）
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
# 日文：平假名 / 片假名
_KANA_RE = re.compile(r"[぀-ヿ]")
# 韩文音节 / 兼容字母
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ]")
# 拉丁字母（含带音标的扩展拉丁）
_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")
# 数字（全/半角阿拉伯数字、罗马数字）
_DIGIT_RE = re.compile(r"[0-9０-９]")
# 常见英文缩写：连续大写（IBM、GPU）或带点的缩写（U.S.A.、Ph.D.）
_ABBREV_RE = re.compile(
    r"(?:[A-Z][A-Z0-9\-&+]{1,}|[A-Za-z]{1,4}(?:\.[A-Za-z]{1,4})+\.?)"
)


def _char_category(ch: str) -> str:
    if _KANA_RE.search(ch):
        return "ja"
    if _HANGUL_RE.search(ch):
        return "ko"
    if _CJK_RE.search(ch):
        return "zh"
    if _LATIN_RE.search(ch):
        return "en"
    if _DIGIT_RE.search(ch):
        return "digits"
    if ch.isspace():
        return "space"
    return "symbols"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LanguageSpan:
    """混合文本中的一个语言段。"""
    text: str
    lang: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {"text": self.text, "lang": self.lang,
                "start": self.start, "end": self.end}


@dataclass
class LanguageProfile:
    """整段文本的语言画像。"""
    language: str                     # 主语言
    language_name: str
    confidence: float                 # 主语言在有效字符中的占比
    ratios: dict[str, float]          # 各类字符占比（以全部字符为分母）
    counts: dict[str, int]
    chinese_variant: str = "unknown"  # zh 子类型：simplified / traditional / mixed
    spans: list[LanguageSpan] = field(default_factory=list)
    effective_chars: int = 0
    length: int = 0

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_name": self.language_name,
            "confidence": round(self.confidence, 4),
            "ratios": {k: round(v, 4) for k, v in self.ratios.items()},
            "counts": self.counts,
            "chinese_variant": self.chinese_variant,
            "effective_chars": self.effective_chars,
            "length": self.length,
            "spans": [s.to_dict() for s in self.spans],
        }


LANGUAGE_NAMES = {
    "zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文",
    "digits": "纯数字", "symbols": "符号", "unknown": "未知",
}

VARIANT_NAMES = {"simplified": "简体", "traditional": "繁体",
                 "mixed": "简繁混合", "unknown": "未判定"}


# ---------------------------------------------------------------------------
# 主语言判定
# ---------------------------------------------------------------------------

def _counts(text: str) -> dict[str, int]:
    counts = {"zh": 0, "en": 0, "ja": 0, "ko": 0,
              "digits": 0, "symbols": 0, "space": 0}
    for ch in text:
        counts[_char_category(ch)] += 1
    return counts


def detect_chinese_variant(text: str) -> str:
    """判断中文部分是简体、繁体还是混合。

    依据转换器字表中「繁体独有字」与「简体独有字」的命中情况。
    两类都有明显命中时判为简繁混合。
    """
    conv = get_converter()
    trad_hits = simp_hits = 0
    for ch in text:
        if ch in conv.trad_only:
            trad_hits += 1
        elif ch in conv.simp_only:
            simp_hits += 1
    if trad_hits and simp_hits:
        return "mixed"
    if trad_hits:
        return "traditional"
    if simp_hits:
        return "simplified"
    return "unknown"


def detect_language(text: str, split_spans: bool = False) -> LanguageProfile:
    """识别一段文本的主语言。

    ``split_spans`` 为真时顺带产出语言分段结果（批量场景可省掉重复计算）。
    """
    counts = _counts(text)
    total = len(text)
    # 有效字符：四种语言字符（数字、标点、空白不计入主语言判定）
    effective = counts["zh"] + counts["en"] + counts["ja"] + counts["ko"]
    ratios = {k: (counts[k] / total if total else 0.0)
              for k in ("zh", "en", "ja", "ko", "digits", "symbols", "space")}

    if effective:
        lang = max(("zh", "en", "ja", "ko"), key=lambda k: counts[k])
        confidence = counts[lang] / effective
    elif counts["digits"]:
        lang, confidence = "digits", 1.0
    elif total:
        lang, confidence = "symbols", 1.0
    else:
        lang, confidence = "unknown", 0.0

    profile = LanguageProfile(
        language=lang,
        language_name=LANGUAGE_NAMES.get(lang, lang),
        confidence=confidence,
        ratios=ratios,
        counts=counts,
        effective_chars=effective,
        length=total,
    )
    if lang == "zh":
        profile.chinese_variant = detect_chinese_variant(text)
    if split_spans:
        profile.spans = split_languages(text)
    return profile


# ---------------------------------------------------------------------------
# 混合文本语言分段
# ---------------------------------------------------------------------------

# 归一到展示层的大类：标点数字并入相邻的主语言段，空白单独保留
_CORE_LANGS = ("zh", "en", "ja", "ko")


def split_languages(text: str, include_punct: bool = False) -> list[LanguageSpan]:
    """把混合文本切成连续的语言段。

    标点 / 数字 / 空白这类「语言中立」字符会被并入紧邻的主语言段，
    从而得到用户期望的「中文段 + 英文段」而不是碎片。孤立存在
    （两侧主语言不同或位于边界）时归入前一个主语言段；整段没有任何
    主语言字符时，整体作为一个 ``symbols`` 段返回。

    ``include_punct=True`` 时标点数字也单独成段（调试 / 精细展示用）。
    """
    if not text:
        return []

    # 先按字符类别做粗切分
    raw: list[LanguageSpan] = []
    start = 0
    cur = _char_category(text[0])
    for i in range(1, len(text)):
        cat = _char_category(text[i])
        if cat != cur:
            raw.append(LanguageSpan(text[start:i], cur, start, i))
            start, cur = i, cat
    raw.append(LanguageSpan(text[start:], cur, start, len(text)))

    if include_punct:
        return raw

    return _merge_neutral(raw)


def _merge_neutral(raw: list[LanguageSpan]) -> list[LanguageSpan]:
    """把 digits/symbols/space 段并入相邻的主语言段。

    归属启发式：中立段若含 CJK 标点（，。！？等），优先跟随中文；
    否则跟随左侧主语言；纯空白若另一侧为中文也归中文。
    """
    core_idx = [i for i, s in enumerate(raw) if s.lang in _CORE_LANGS]
    if not core_idx:
        # 没有语言字符：整段合并为一个 symbols 段
        return [LanguageSpan(
            "".join(s.text for s in raw), "symbols",
            raw[0].start, raw[-1].end)]

    def _side_lang(neutral_text: str, left: Optional[str],
                   right: Optional[str]) -> str:
        has_cjk_punct = bool(re.search(r"[，。！？；：、（）【】《》“”‘’…·]",
                                       neutral_text))
        if has_cjk_punct:
            # 夹在两段之间：优先跟随中文一侧
            if right == "zh":
                return right
            if left == "zh":
                return left
            # 边界处（另一侧为空）或两侧都非中文：CJK 标点仍归中文
            return "zh"
        return left or right or "symbols"

    first_core, last_core = core_idx[0], core_idx[-1]
    merged: list[LanguageSpan] = []

    def _append(seg: LanguageSpan) -> None:
        if merged and merged[-1].lang == seg.lang:
            prev = merged[-1]
            merged[-1] = LanguageSpan(
                prev.text + seg.text, prev.lang, prev.start, seg.end)
        else:
            merged.append(seg)

    # 边界前的中立内容
    if first_core > 0:
        leading = raw[:first_core + 1]
        lang = _side_lang("".join(s.text for s in raw[:first_core]),
                          None, raw[first_core].lang)
        _append(LanguageSpan("".join(s.text for s in leading), lang,
                             leading[0].start, leading[-1].end))
    else:
        _append(raw[0])

    i = first_core + 1
    while i <= last_core:
        if raw[i].lang in _CORE_LANGS:
            _append(raw[i])
            i += 1
            continue
        j = i
        while j <= last_core and raw[j].lang not in _CORE_LANGS:
            j += 1
        neutral_text = "".join(raw[k].text for k in range(i, j))
        left_lang = merged[-1].lang
        right_lang = raw[j].lang if j <= last_core else None
        lang = _side_lang(neutral_text, left_lang, right_lang)
        _append(LanguageSpan(neutral_text, lang, raw[i].start, raw[j - 1].end))
        i = j

    if last_core + 1 < len(raw):
        tail = raw[last_core + 1:]
        tail_text = "".join(s.text for s in tail)
        lang = _side_lang(tail_text, merged[-1].lang, None)
        _append(LanguageSpan(tail_text, lang, tail[0].start, tail[-1].end))

    return merged


def find_abbreviations(text: str) -> list[tuple[int, int]]:
    """返回文本中疑似英文缩写（GPU、U.S.A.）的区间，供规范化时保护。"""
    return [(m.start(), m.end()) for m in _ABBREV_RE.finditer(text)]
