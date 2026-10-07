"""语种识别：篇章级、段落级与中英混合片段切分。

零依赖的轻量实现，针对本平台语料的典型来源（中英混排、繁简混排、
全角半角混排）优化，不追求支持几十种语言，而是把三件事做准：

1. **篇章主语言**：``zh``（中文，繁/简由字形证据区分）/ ``en``（英文）
   / ``mixed``（中英占比都不低）/ ``unknown``。
2. **段落归属**：逐段（空行或换行切分）给出语言标签。
3. **混合文本切段**：把一段混合文本切成连续的 ``zh`` / ``en`` /
   ``number`` / ``punct`` / ``space`` 片段，供页面分段着色与逐处确认。

判别依据
--------
* CJK 统一表意文字区间 + 繁简独有字形（:mod:`nlp.charconv`）；
* 拉丁字母与常见英文功能词；
* 数字（含全角数字）、标点、空白不参与主语言投票，但保留在片段里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .charconv import SIMP_ONLY, TRAD_ONLY

# 字符类别
_CJK = re.compile(r"[一-鿿㐀-䶿]")
_LATIN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
_DIGIT = re.compile(r"[0-9０-９]")
_SPACE = re.compile(r"\s")
# 全角 ASCII 段（！ ～ 与全角空格），规范化时可无损折半
_FULLWIDTH_ASCII = re.compile(r"[！-～　]")

# 英文强信号词（独立成词时显著加分）
_EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "at", "by",
    "from", "not", "no", "we", "you", "they", "he", "she", "his", "her",
    "their", "our", "your", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "than",
    "then", "so", "such", "which", "who", "whom", "what", "when", "where",
    "why", "how", "also", "about", "into", "over", "after", "before",
}

_PARA_SPLIT = re.compile(r"\n[ \t]*(?:\n[ \t]*)+|\n+")
_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass
class LangProfile:
    """一段文本的语言画像。"""
    lang: str                         # zh / en / mixed / unknown
    script: str = ""                  # zh-CN / zh-TW / zh-Hans-Hant / ""
    cjk_chars: int = 0
    latin_chars: int = 0
    digit_chars: int = 0
    trad_marks: int = 0               # 繁体独有字形计数
    simp_marks: int = 0               # 简体独有字形计数
    en_hits: int = 0                  # 英文功能词命中
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lang": self.lang, "script": self.script,
            "cjk_chars": self.cjk_chars, "latin_chars": self.latin_chars,
            "digit_chars": self.digit_chars,
            "trad_marks": self.trad_marks, "simp_marks": self.simp_marks,
            "en_hits": self.en_hits, "scores": self.scores,
        }


def profile(text: str) -> LangProfile:
    """统计文本的语言特征并判定主语言 / 繁简字形。"""
    cjk = latin = digits = trad = simp = 0
    for ch in text:
        if _CJK.match(ch):
            cjk += 1
            if ch in TRAD_ONLY:
                trad += 1
            if ch in SIMP_ONLY:
                simp += 1
        elif _LATIN.match(ch):
            latin += 1
        elif _DIGIT.match(ch):
            digits += 1

    words = [w.lower() for w in _WORD_RE.findall(text)]
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)

    # 主语言：以 CJK / 拉丁字母字数为主，英文功能词作为辅助信号
    letters = cjk + latin
    scores = {
        "zh": cjk / letters if letters else 0.0,
        "en": latin / letters if letters else 0.0,
    }
    if letters == 0:
        lang = "unknown"
    else:
        zh_share = scores["zh"]
        # 英文功能词密集出现可提升英文置信，但不改变绝对多数
        if zh_share >= 0.80:
            lang = "zh"
        elif zh_share <= 0.20 and (latin > 0 or en_hits > 0):
            lang = "en"
        elif latin == 0:
            lang = "zh"
        elif cjk == 0:
            lang = "en"
        else:
            lang = "mixed"

    script = ""
    if cjk:
        if trad > 0 and simp == 0:
            script = "zh-Hant"
        elif simp > 0 and trad == 0:
            script = "zh-Hans"
        elif trad > 0 and simp > 0:
            # 以占优字形命名，同时标注混用
            script = "zh-Hant" if trad >= simp else "zh-Hans"
            script += "+mixed-script"
    if lang == "zh" and script:
        mapping = {"zh-Hans": "zh-CN", "zh-Hant": "zh-TW"}
        script_name = mapping.get(script.split("+")[0], script)
        if "+" in script:
            script_name += "+mixed"
        script = script_name
    return LangProfile(lang=lang, script=script, cjk_chars=cjk,
                       latin_chars=latin, digit_chars=digits,
                       trad_marks=trad, simp_marks=simp,
                       en_hits=en_hits, scores={k: round(v, 4) for k, v in scores.items()})


def detect(text: str) -> str:
    """便捷接口：返回篇章主语言代码。"""
    return profile(text).lang


def split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """切分自然段，返回 ``(起始偏移, 结束偏移, 段落文本)``。

    单个换行也视作分段（语料常来自表格/日志列），但偏移保留原文位置，
    便于结果回流时对齐。
    """
    paragraphs: list[tuple[int, int, str]] = []
    pos = 0
    for match in _PARA_SPLIT.finditer(text):
        seg = text[pos:match.start()]
        if seg:
            paragraphs.append((pos, match.start(), seg))
        pos = match.end()
    if pos < len(text):
        seg = text[pos:]
        if seg.strip() or seg:
            paragraphs.append((pos, len(text), seg))
    return paragraphs


def detect_paragraphs(text: str) -> list[dict]:
    """逐段识别语言，返回带偏移的段落画像列表。"""
    result = []
    for start, end, seg in split_paragraphs(text):
        p = profile(seg)
        result.append({
            "start": start, "end": end, "text": seg,
            **p.to_dict(),
        })
    return result


# 片段级类别 -> 可读名
SEGMENT_LABELS = {
    "zh": "中文", "en": "英文", "number": "数字",
    "punct": "标点", "space": "空白", "other": "其他",
}


def _char_class(ch: str) -> str:
    if _CJK.match(ch):
        return "zh"
    if _LATIN.match(ch):
        return "en"
    if _DIGIT.match(ch):
        return "number"
    if _SPACE.match(ch):
        return "space"
    if ch.isalnum():
        # 其他文字（日文假名、韩文等）先归入 other，不当作中文/英文
        return "other"
    return "punct"


def segment_mixed(text: str, *, merge_punct: bool = True) -> list[dict]:
    """把混合文本切成连续语言片段。

    ``merge_punct=True`` 时标点/数字/空白并入相邻的主语言片段，
    只在「中→英」边界处保留独立片段，使逐处确认的条目更聚焦。
    """
    raw: list[dict] = []
    for ch in text:
        cls = _char_class(ch)
        if raw and raw[-1]["type"] == cls:
            raw[-1]["text"] += ch
            raw[-1]["end"] += 1
        else:
            start = raw[-1]["end"] if raw else 0
            raw.append({"type": cls, "text": ch, "start": start,
                        "end": start + 1, "label": SEGMENT_LABELS[cls]})

    if not merge_punct:
        for seg in raw:
            seg["lang"] = "zh" if seg["type"] == "zh" else (
                "en" if seg["type"] == "en" else "")
        return raw

    # 将非主语言片段并入相邻主片段；位于两个主片段之间时，
    # 挂到前一个片段后面（符合「中文句内夹英文标点」的常见情形）。
    merged: list[dict] = []
    for seg in raw:
        if seg["type"] in ("zh", "en", "other"):
            merged.append(seg)
        else:
            if merged:
                merged[-1]["text"] += seg["text"]
                merged[-1]["end"] = seg["end"]
                if merged[-1]["type"] not in ("zh", "en"):
                    merged[-1]["type"] = seg["type"]
            else:
                merged.append(seg)
    out = []
    for seg in merged:
        seg["lang"] = "zh" if seg["type"] == "zh" else (
            "en" if seg["type"] == "en" else "")
        seg["label"] = SEGMENT_LABELS.get(seg["type"], "其他")
        out.append(seg)
    return out


def is_fullwidth_heavy(text: str, threshold: float = 0.05) -> bool:
    """全角字符（含全角标点/字母/数字）占比是否偏高。"""
    if not text:
        return False
    fw = len(_FULLWIDTH_ASCII.findall(text))
    return fw / max(len(text), 1) >= threshold
