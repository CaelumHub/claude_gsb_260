"""文本规范化：可指定目标规范，输出逐字符的变更清单供逐处确认。

能力
----
* 繁简转换（:mod:`nlp.charconv`，简→繁采用保守策略，不破坏专有名词）
* 全角 / 半角统一：标点、拉丁字母、数字分别可配
* 空白统一：CRLF→LF、Tab→空格、连续空白折一、CJK 字间空格清除
* 引号 / 省略号 / 破折号风格统一
* **保护机制**：用户词表与正则命中的片段（专有名词、品牌、缩写）
  跳过字形与标点转换；拉丁缩写（IBM、U.S.A.、GPU/NPU）天然不受影响
* 每次规范化产出带原文偏移的变更列表，前端可逐处接受 / 驳回 / 改写，
  :meth:`apply_decisions` 按决策重建文本，被驳回的改动原样保留

所有变换都是幂等的：``normalize(normalize(x)) == normalize(x)``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import charconv
from .langid import profile, split_paragraphs

# ---------------------------------------------------------------------------
# 字符映射
# ---------------------------------------------------------------------------

# CJK 全角标点 → 半角 ASCII（mode="half" 时使用）
_CJK_PUNCT_HALF = {
    "，": ",", "、": ",", "。": ".", "！": "!", "？": "?", "：": ":",
    "；": ";", "（": "(", "）": ")", "【": "[", "】": "]", "《": "<",
    "》": ">", "「": '"', "」": '"', "『": "'", "』": "'",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "·": ".", "％": "%",
}
# 弯引号 → 直引号（quotes="straight"）
_CURLY_QUOTES = {"“": '"', "”": '"', "‘": "'", "’": "'",
                 "「": '"', "」": '"', "『": "'", "』": "'"}
# 半角 ASCII 标点 → 中文全角（mode="full" 时使用）
_HALF_TO_CJK = {
    ",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；",
    "(": "（", ")": "）",
}
_FULLWIDTH_BEGIN, _FULLWIDTH_END = 0xFF01, 0xFF5E
_FULLWIDTH_SPACE = "　"


def _IS_CJK(ch: str) -> bool:
    return bool(ch) and "一" <= ch <= "鿿"

CHANGE_REASONS = {
    "script": "繁简字形转换",
    "punct": "全角/半角标点",
    "quote": "引号风格",
    "digit": "全角/半角数字",
    "letter": "全角/半角字母",
    "space": "空白统一",
    "ellipsis": "省略号/破折号",
}


# ---------------------------------------------------------------------------
# 规范配置
# ---------------------------------------------------------------------------

@dataclass
class NormalizationSpec:
    """目标规范。所有字段都可在页面上指定。"""
    script: str = "simplified"     # simplified / traditional / auto / none
    punctuation: str = "half"      # half / cjk / full / keep
    digits: str = "half"           # half / full / keep
    letters: str = "half"          # half / full / keep
    quotes: str = "keep"           # straight / keep
    ellipsis: str = "collapse"     # collapse（…/……→...，—/——→--）/ keep
    whitespace: str = "collapse"   # collapse / keep（保留段落换行）
    remove_cjk_spaces: bool = True
    protect: list[str] = field(default_factory=list)
    protect_regex: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "NormalizationSpec":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]


def _fullwidth_to_half(ch: str) -> Optional[str]:
    code = ord(ch)
    if code == 0x3000:
        return " "
    if _FULLWIDTH_BEGIN <= code <= _FULLWIDTH_END:
        return chr(code - 0xFEE0)
    return None


def _half_to_fullwidth(ch: str) -> Optional[str]:
    code = ord(ch)
    if ch == " ":
        return _FULLWIDTH_SPACE
    if 0x21 <= code <= 0x7E:
        return chr(code + 0xFEE0)
    return None


# ---------------------------------------------------------------------------
# 保护区间
# ---------------------------------------------------------------------------

def _protected_spans(text: str, spec: NormalizationSpec) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for term in spec.protect:
        if not term:
            continue
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(term)))
            start = idx + len(term)
    for pattern in spec.protect_regex:
        if not pattern:
            continue
        try:
            rx = re.compile(pattern)
        except re.error:
            continue
        for m in rx.finditer(text):
            spans.append(m.span())
    # 合并重叠区间
    spans.sort()
    merged: list[tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _is_protected(pos: int, spans: list[tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= pos < b:
            return True
        if a > pos:
            break
    return False


# ---------------------------------------------------------------------------
# 规范化核心
# ---------------------------------------------------------------------------

@dataclass
class _Change:
    para: int
    start: int          # 相对原文的全局偏移
    end: int
    original: str
    replacement: str
    kind: str

    def to_dict(self, idx: int) -> dict:
        return {
            "idx": idx, "para": self.para,
            "start": self.start, "end": self.end,
            "original": self.original, "replacement": self.replacement,
            "kind": self.kind, "reason": CHANGE_REASONS.get(self.kind, self.kind),
            "auto": True,
        }


def _resolve_script_direction(text: str, spec: NormalizationSpec) -> str:
    mode = spec.script
    if mode in ("simplified", "traditional", "none"):
        return mode
    p = profile(text)
    # auto：出现繁体字形就收敛为简体；纯简体不动；纯英文不做字形转换
    if p.trad_marks > 0:
        return "simplified"
    return "none"


def _normalize_spaces(text: str, base: int, para: int, spec: NormalizationSpec,
                      protected: list[tuple[int, int]]) -> list[_Change]:
    """生成空白类变更（偏移均基于全局原文）。"""
    changes: list[_Change] = []
    if spec.whitespace == "keep":
        # 即便保留，也统一 CRLF 与行尾空白属于另一选项，这里完全不动
        return changes

    # 1) 含 Tab / 全角空格 / CR 的空白串，以及两个及以上连续空格
    run_re = re.compile(r"[ \t\u3000\r]*[\t\u3000\r][ \t\u3000\r]*| {2,}")

    def emit_run(m: re.Match) -> str:
        run = m.group(0)
        global_start = base + m.start()
        if _is_protected(global_start, protected):
            return run
        replacement = run.replace("\t", " ").replace("\r", "").replace("\u3000", " ")
        replacement = re.sub(r" +", " ", replacement)
        # \u7a7a\u767d\u4e32\u4e24\u4fa7\u90fd\u662f\u6c49\u5b57\uff08\u4e14\u4e0d\u542b\u6362\u884c\uff09\u65f6\u76f4\u63a5\u6e05\u9664\uff1a\u300c\u4e00\u500b\u3000\u6e2c\u8a66\u300d\u2192\u300c\u4e00\u4e2a\u6d4b\u8bd5\u300d
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if replacement == " " and "\n" not in run and _IS_CJK(before) and _IS_CJK(after):
            replacement = ""
        if replacement != run:
            changes.append(_Change(
                para, global_start, base + m.end(), run, replacement, "space"))
        return run  # 偏移始终基于原文，故不改变输入

    run_re.sub(emit_run, text)

    # 2) CJK 字间的孤立空格（抓取/排版噪声）；避开第 1 步已覆盖的位置
    if spec.remove_cjk_spaces:
        def covered(pos: int) -> bool:
            return any(c.start <= pos < c.end for c in changes)

        def emit_cjk_space(m: re.Match) -> str:
            space_pos = base + m.start(1) + 1
            if covered(space_pos) or _is_protected(space_pos, protected):
                return m.group(0)
            changes.append(_Change(
                para, space_pos, space_pos + 1, " ", "", "space"))
            return m.group(0)

        re.sub(r"([一-鿿]) ([一-鿿])", emit_cjk_space, text)

    changes.sort(key=lambda c: c.start)
    return changes


def _ellipsis_changes(text: str, base: int, para: int,
                      protected: list[tuple[int, int]],
                      existing: list[_Change]) -> list[_Change]:
    """省略号 ``…`` / ``……`` → ``...``，破折号 ``—`` / ``——`` → ``--``。"""
    out: list[_Change] = []

    def covered(a: int, b: int) -> bool:
        for c in existing + out:
            if a < c.end and c.start < b:
                return True
        return False

    rx = re.compile(r"…+|—+")
    for m in rx.finditer(text):
        g0, g1 = base + m.start(), base + m.end()
        if covered(g0, g1) or _is_protected(g0, protected):
            continue
        run = m.group(0)
        # 中文惯例：两个省略号字符仍对应三点；破折号统一两点
        repl = "..." if run[0] == "…" else "--"
        out.append(_Change(para, g0, g1, run, repl, "ellipsis"))
    return out


def normalize(text: str, spec: Optional[dict | NormalizationSpec] = None) -> dict:
    """对文本执行规范化，返回原文、规范文本、逐段识别结果与变更清单。

    返回结构::

        {
          "original", "normalized",
          "profile": {...}, "spec": {...},
          "paragraphs": [{idx, start, end, text, normalized_text,
                          lang, script, changes:[...]}],
          "changes": [...], "stats": {...}
        }
    """
    if isinstance(spec, dict):
        spec = NormalizationSpec.from_dict(spec)
    spec = spec or NormalizationSpec()

    doc_profile = profile(text)
    direction = _resolve_script_direction(text, spec)
    protected = _protected_spans(text, spec)

    paragraphs = split_paragraphs(text)
    all_changes: list[_Change] = []

    for p_idx, (p_start, p_end, p_text) in enumerate(paragraphs):
        # 字符级替换
        for i, ch in enumerate(p_text):
            global_pos = p_start + i
            if _is_protected(global_pos, protected):
                continue
            new = _convert_char(ch, spec, direction)
            if new is not None and new != ch:
                kind = _change_kind(ch, new, spec, direction)
                all_changes.append(_Change(
                    p_idx, global_pos, global_pos + 1, ch, new, kind))

        # 段落级空白处理（在原文段落上做，偏移仍按全局对齐）
        all_changes.extend(
            _normalize_spaces(p_text, p_start, p_idx, spec, protected))

    # 所有变更互不重叠（字符变换与空白作用于不同字符）
    all_changes.sort(key=lambda c: c.start)

    # 省略号 / 破折号折叠（多字符，扫描时避开已覆盖位置）
    if spec.ellipsis == "collapse":
        for p_idx, (p_start, p_end, p_text) in enumerate(paragraphs):
            all_changes.extend(
                _ellipsis_changes(p_text, p_start, p_idx, protected, all_changes))
        all_changes.sort(key=lambda c: c.start)
    normalized = _apply_changes(text, all_changes, {})

    by_kind: dict[str, int] = {}
    for c in all_changes:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    change_dicts = [c.to_dict(i) for i, c in enumerate(all_changes)]

    # 逐段视图：用该段自己的变更在段落原文上重建，保证与总文本一致
    para_views = []
    for p_idx, (p_start, p_end, p_text) in enumerate(paragraphs):
        seg_changes = [c for c in all_changes if c.para == p_idx]
        seg_out = _apply_changes(
            p_text, [_shift(c, -p_start) for c in seg_changes], {})
        pp = profile(p_text)
        para_change_dicts = []
        for c in seg_changes:
            gi = next(i for i, gc in enumerate(all_changes) if gc is c)
            d = change_dicts[gi]
            para_change_dicts.append({**d, "start": d["start"] - p_start,
                                      "end": d["end"] - p_start})
        para_views.append({
            "idx": p_idx,
            "start": p_start, "end": p_end,
            "text": p_text,
            "normalized_text": seg_out,
            "lang": pp.lang,
            "script": pp.script,
            "profile": pp.to_dict(),
            "changes": para_change_dicts,
        })

    return {
        "original": text,
        "normalized": normalized,
        "profile": doc_profile.to_dict(),
        "script_direction": direction,
        "spec": spec.to_dict(),
        "paragraphs": para_views,
        "changes": change_dicts,
        "stats": {
            "total": len(all_changes),
            "by_kind": by_kind,
            "protected_spans": len(protected),
        },
    }


def _shift(change: _Change, delta: int) -> "_Change":
    return _Change(change.para, change.start + delta, change.end + delta,
                   change.original, change.replacement, change.kind)


def _convert_char(ch: str, spec: NormalizationSpec, direction: str):
    """返回字符替换结果；None 表示不变。优先级：字形 > 引号 > 标点/字母/数字。"""
    # 1) 繁简字形
    if direction == "simplified":
        mapped = charconv.to_simplified(ch)
        if mapped != ch:
            return mapped
    elif direction == "traditional":
        mapped = charconv.to_traditional(ch)
        if mapped != ch:
            return mapped

    # 2) 引号风格
    if spec.quotes == "straight" and ch in _CURLY_QUOTES:
        return _CURLY_QUOTES[ch]

    # 3) 标点
    if spec.punctuation == "half":
        if ch in _CJK_PUNCT_HALF and ch not in _CURLY_QUOTES:
            return _CJK_PUNCT_HALF[ch]
        if ch in _CJK_PUNCT_HALF and spec.quotes != "straight":
            # 引号受 quotes 选项控制
            if ch not in ("“", "”", "‘", "’", "「", "」", "『", "』"):
                return _CJK_PUNCT_HALF[ch]
        hw = _fullwidth_to_half(ch)
        if hw is not None and not ch.isspace() and not ch.isalnum() and not hw.isalnum():
            return hw
    elif spec.punctuation == "cjk":
        # 只折半全角 ASCII 标点，中文标点保留
        hw = _fullwidth_to_half(ch)
        if hw is not None and not ch.isspace() and not ch.isalnum() and not hw.isalnum():
            return hw
    elif spec.punctuation == "full":
        if ch in _HALF_TO_CJK:
            return _HALF_TO_CJK[ch]

    # 4) 全角字母
    if spec.letters == "half":
        hw = _fullwidth_to_half(ch)
        if hw is not None and hw.isalpha() and hw.isascii():
            return hw
    elif spec.letters == "full":
        fw = _half_to_fullwidth(ch)
        if fw is not None and ch.isalpha() and ch.isascii():
            return fw

    # 5) 全角数字
    if spec.digits == "half":
        hw = _fullwidth_to_half(ch)
        if hw is not None and hw.isdigit():
            return hw
    elif spec.digits == "full":
        fw = _half_to_fullwidth(ch)
        if fw is not None and ch.isdigit():
            return fw
    return None


def _change_kind(original: str, replacement: str,
                 spec: NormalizationSpec, direction: str) -> str:
    if direction == "simplified" and charconv.to_simplified(original) == replacement:
        return "script"
    if direction == "traditional" and charconv.to_traditional(original) == replacement:
        return "script"
    if original in _CURLY_QUOTES:
        return "quote"
    if original.isdigit() or replacement.isdigit():
        return "digit"
    if original.isalpha() or replacement.isalpha():
        return "letter"
    return "punct"


def _apply_changes(text: str, changes: list[_Change],
                   overrides: dict[int, str | None]) -> str:
    """按（可能重叠已排除的）变更重建文本。

    ``overrides``: {变更在 changes 中的下标: 自定义替换串}，
    值为 ``None`` 表示驳回该变更（保留原文）。
    """
    if not changes:
        return text
    out: list[str] = []
    cursor = 0
    for i, c in enumerate(changes):
        out.append(text[cursor:c.start])
        if i in overrides:
            if overrides[i] is None:
                out.append(text[c.start:c.end])    # 驳回
            else:
                out.append(overrides[i])           # 用户改写
        else:
            out.append(c.replacement)              # 接受
        cursor = c.end
    out.append(text[cursor:])
    return "".join(out)


def apply_decisions(plan: dict, decisions: Optional[dict | list] = None) -> dict:
    """根据用户对变更清单的决策重建规范文本。

    ``decisions`` 支持两种形式：

    * ``{变更idx: True/False/"自定义文本"}``；
    * ``[{"idx": 3, "accepted": false}, {"idx": 5, "replacement": "，"}]``。

    未出现的变更视为接受（与初版 plan 一致）。
    """
    overrides: dict[int, Optional[str]] = {}
    if isinstance(decisions, dict):
        for k, v in decisions.items():
            idx = int(k)
            if v is True:
                continue
            overrides[idx] = None if v is False else str(v)
    elif decisions:
        for item in decisions:
            idx = item["idx"]
            if "replacement" in item:
                overrides[idx] = str(item["replacement"])
            elif not item.get("accepted", True):
                overrides[idx] = None

    original = plan["original"]
    changes = [
        _Change(c["para"], c["start"], c["end"], c["original"],
                c["replacement"], c["kind"])
        for c in plan["changes"]
    ]
    # 驳回 / 改写后，重叠的空白变更可能相邻冲突；按偏移应用是安全的，
    # 因为每条变更互不重叠（空白与字符变换作用于不同字符）。
    final = _apply_changes(original, changes, overrides)

    accepted = [c for i, c in enumerate(plan["changes"]) if i not in overrides]
    rejected = [plan["changes"][i] for i in overrides if overrides[i] is None]
    edited = [{"idx": i, **plan["changes"][i], "final": overrides[i]}
              for i in overrides if overrides[i] is not None]

    # 逐段视图重建（段落文本本身没被改动，沿用原切分）
    final_views = []
    for view in plan["paragraphs"]:
        seg_changes_idx = [c["idx"] for c in view["changes"]]
        seg_changes = [changes[i] for i in seg_changes_idx]
        local_overrides = {
            local_i: overrides[gi]
            for local_i, gi in enumerate(seg_changes_idx) if gi in overrides
        }
        seg_out = _apply_changes(view["text"],
                                 [_shift(c, -view["start"]) for c in seg_changes],
                                 local_overrides)
        final_views.append({**view, "final_text": seg_out})

    return {
        "original": original,
        "normalized": final,
        "spec": plan.get("spec", {}),
        "profile": plan.get("profile", {}),
        "script_direction": plan.get("script_direction"),
        "paragraphs": final_views,
        "changes": plan["changes"],
        "stats": {
            "total": len(plan["changes"]),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "edited": len(edited),
            "rejected_items": rejected,
            "edited_items": edited,
        },
    }


def _slice_normalized(text, normalized, changes, start, end, overrides):
    """占位：实际段落视图在调用处整体重算。"""
    return ""
