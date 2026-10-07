"""文本规范化：全/半角、简/繁体、数字与空白统一，且可逐处回溯。

处理目标
--------
来源杂乱的语料（中英混排、繁简混排、全角半角混用）统一到一份可配置的
目标规范，同时：

* **不改原意**——只做字符层面的等价变换（全角半角、繁简、空白折叠）；
* **不破坏专有名词与英文缩写**——缩写、拉丁专名、NER 实体及用户给定
  保护词先用占位符罩住，变换结束后原样还原；
* **逐处可确认**——每一处改动都带有原文片段、目标片段、规则编号与
  原文偏移，前端可以逐条接受 / 拒绝。

输出结构（:meth:`TextNormalizer.normalize`）以「段落」为单位，与页面上
分段落展示、逐处确认的交互一致。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Optional

from .langid import (LanguageSpan, detect_language, find_abbreviations)
from .variants import get_converter

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

VARIANT_CHOICES = ("simplified", "traditional", "keep")
PUNCT_CHOICES = ("half", "full", "keep")
DIGIT_CHOICES = ("half", "full", "keep")
SPACE_CHOICES = ("collapse", "single", "keep")

CHANGE_RULE_NAMES = {
    "variant": "繁简转换",
    "punct": "标点全半角",
    "digit": "数字全半角",
    "letter": "字母全半角",
    "space": "空白规范化",
    "space_insert": "中英文间补空格",
    "space_remove": "删除多余空白",
    "linebreak": "换行规范化",
    "zero_width": "删除零宽字符",
    "quote": "引号统一",
}


@dataclass
class NormalizeConfig:
    """目标规范。所有项目都可指定，``keep`` 表示该项不动。"""
    variant: str = "simplified"        # simplified / traditional / keep
    punct: str = "half"                # half（全角标点→半角）/ full / keep
    digits: str = "half"               # half（全角数字→半角）/ full / keep
    letters: str = "half"              # 全角拉丁字母处理（half/full/keep）
    spaces: str = "collapse"           # collapse：连续空白压成一个 / single / keep
    space_cjk_latin: Optional[bool] = False  # 中英文之间是否补空格（None=不动）
    normalize_linebreak: bool = True   # CRLF/CR → LF
    remove_zero_width: bool = True
    normalize_quotes: bool = False     # 弯引号 → 直引号（默认关，避免争议）
    protected_terms: list[str] = field(default_factory=list)
    protect_ner: bool = True
    protect_abbrev: bool = True

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "NormalizeConfig":
        data = data or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 常量映射
# ---------------------------------------------------------------------------

def _fullwidth_ascii(ch: str) -> Optional[str]:
    """全角 ASCII（！～ 以及全角空格）→ 半角，非全角字符返回 None。"""
    code = ord(ch)
    if code == 0x3000:
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    return None


# 常用全角中文标点 → 半角 ASCII（目标规范=半角时使用）
_CJK_PUNCT_HALF = {
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
    "“": '"', "”": '"', "‘": "'", "’": "'", "　": " ",
    "、": ",", "…": "...", "—": "-", "－": "-", "～": "~",
}
# 反向：半角 → 全角（仅做无歧义的几个；引号交给专门选项）
_HALF_PUNCT_FULL = {
    ",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：",
    "(": "（", ")": "）",
}

# 零宽字符 / BOM / 双向控制符
_ZERO_WIDTH_CODEPOINTS = {
    0x200B, 0x200C, 0x200D, 0xFEFF,
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
    0x2066, 0x2067, 0x2068, 0x2069,
}
_ZERO_WIDTH_RE = re.compile("[" + "".join(chr(c) for c in _ZERO_WIDTH_CODEPOINTS) + "]")
_WS_RUN_RE = re.compile(r"[ \t　]+")
_QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'"}
_QUOTE_RE = re.compile("[" + "".join(_QUOTE_MAP) + "]")

_CJK_CHAR_RE = re.compile(r"[一-鿿㐀-䶿぀-ヿ가-힯]")
_LATIN_DIGIT_RE = re.compile(r"[A-Za-z0-9]")
_CJK_PUNCT_RE = re.compile(r"[，。！？；：、（）【】《》“”‘’]")


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_CHAR_RE.match(ch))


def _is_latin_digit(ch: str) -> bool:
    return bool(_LATIN_DIGIT_RE.match(ch))


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------

@dataclass
class Change:
    rule: str
    start: int          # 原文偏移
    end: int
    original: str
    changed: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "rule_name": CHANGE_RULE_NAMES.get(self.rule, self.rule),
                "start": self.start, "end": self.end,
                "original": self.original, "changed": self.changed,
                "accepted": True}


@dataclass
class ParagraphResult:
    index: int
    original: str
    normalized: str
    changes: list[Change]
    language: str
    language_name: str
    chinese_variant: str
    spans: list[LanguageSpan]
    protected: list[str]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "original": self.original,
            "normalized": self.normalized,
            "changes": [c.to_dict() for c in self.changes],
            "change_count": len(self.changes),
            "changed": len(self.changes) > 0,
            "language": self.language,
            "language_name": self.language_name,
            "chinese_variant": self.chinese_variant,
            "spans": [s.to_dict() for s in self.spans],
            "protected": self.protected,
        }


@dataclass
class NormalizeResult:
    original: str
    normalized: str
    paragraphs: list[ParagraphResult]
    config: NormalizeConfig
    protected: list[str]

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "change_count": sum(len(p.changes) for p in self.paragraphs),
            "config": self.config.to_dict(),
            "protected": self.protected,
            "rule_names": CHANGE_RULE_NAMES,
        }


# ---------------------------------------------------------------------------
# 规范化器
# ---------------------------------------------------------------------------

_MASK_BASE = 0xE100  # 保护词占位符；转换器内部使用 E000 段，互不干扰


def _align_linear(source: str, target: str) -> dict[int, str]:
    """线性对齐两个长度可能不同的字符串，返回 ``{源位置: 目标片段}``。

    繁简/全半角转换**绝大多数是等长**的（一字对一字），只有极少数词组
    改变长度（如 胡同→衚衕）。因此先用双指针在「相同字符」上快速推进，
    只在长度不等的局部小区间退化为局部对齐，整体保持 O(n)，
    避免 :class:`SequenceMatcher` 在高度重复文本上的平方级退化。
    """
    mapping: dict[int, str] = {}
    i = j = 0
    n, m = len(source), len(target)
    while i < n and j < m:
        if source[i] == target[j]:
            mapping[i] = source[i]
            i += 1
            j += 1
            continue
        # 出现分歧：向右寻找下一个共同锚点字符（限定窗口，避免退化）
        anchor_src = anchor_tgt = None
        window = 8
        for a in range(1, window):
            for b in range(0, window):
                if i + a < n and j + b < m and source[i + a] == target[j + b]:
                    anchor_src, anchor_tgt = a, b
                    break
            if anchor_src is not None:
                break
        if anchor_src is None:
            # 找不到近锚点：把剩余源字符整体映射到剩余目标后结束
            rest = target[j:]
            mapping[i] = mapping.get(i, "") + rest
            for k in range(i + 1, n):
                mapping[k] = ""
            return mapping
        seg_src = source[i:i + anchor_src]
        seg_tgt = target[j:j + anchor_tgt]
        if seg_src:
            # 源区间首字承担整个目标片段，其余源字映射为空
            mapping[i] = seg_tgt
            for k in range(i + 1, i + len(seg_src)):
                mapping[k] = ""
        else:
            # 纯插入：挂在前一个源字上
            prev = max(i - 1, 0)
            mapping[prev] = mapping.get(prev, "") + seg_tgt
        i += anchor_src
        j += anchor_tgt

    # 尾部
    if i < n:
        tail_tgt = target[j:]
        if i in mapping:
            mapping[i] += tail_tgt
        else:
            mapping[i] = tail_tgt
        for k in range(i + 1, n):
            mapping.setdefault(k, "")
    elif j < m and i > 0:
        mapping[i - 1] = mapping.get(i - 1, "") + target[j:]
    for k in range(n):
        mapping.setdefault(k, source[k])
    return mapping


class TextNormalizer:
    """按 :class:`NormalizeConfig` 对文本做规范化，产出逐处变更明细。"""

    def __init__(self, ner=None):
        self.converter = get_converter()
        self._ner = ner  # 惰性注入，避免循环导入

    # -- 保护区间 ---------------------------------------------------------
    def _protected_spans(self, text: str, config: NormalizeConfig) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if config.protect_abbrev:
            spans.extend(find_abbreviations(text))
        for term in config.protected_terms:
            if term and len(term) >= 2:
                start = 0
                while True:
                    idx = text.find(term, start)
                    if idx < 0:
                        break
                    spans.append((idx, idx + len(term)))
                    start = idx + len(term)
        if config.protect_ner:
            ner = self._get_ner()
            if ner is not None:
                try:
                    for ent in ner.recognize(text):
                        # 只保护拉丁字母组成的实体（中文专名仍应做繁简转换）
                        piece = ent.get("text", "")
                        if piece and any(c.isascii() and c.isalpha() for c in piece) \
                                and not _is_cjk(piece):
                            spans.append((ent["start"], ent["end"]))
                except Exception:  # noqa: BLE001
                    pass
        return _merge_spans(spans)

    def _get_ner(self):
        if self._ner is None:
            try:
                from . import get_ner
                self._ner = get_ner()
            except Exception:  # noqa: BLE001
                self._ner = False
        return self._ner or None

    # -- 繁简对齐 ---------------------------------------------------------
    def _variant_alignment(self, masked: str,
                           config: NormalizeConfig) -> Optional[dict[int, str]]:
        """对（已加保护掩码的）文本做繁简转换，并对齐回每个原始位置。

        返回 ``{原字位置: 转换后字符串}``；未变化的字也在表中。
        长度不一致的词组（如 胡同→衚衕）把整个输出挂在区间首字上，
        其余字映射为空串。
        """
        if config.variant == "simplified":
            converted = self.converter.to_simplified(masked)
        elif config.variant == "traditional":
            converted = self.converter.to_traditional(masked)
        else:
            return None
        if converted == masked:
            return {i: ch for i, ch in enumerate(masked)}
        return _align_linear(masked, converted)

    # -- 单段落 -----------------------------------------------------------
    def normalize_paragraph(self, text: str, config: NormalizeConfig,
                            index: int = 0) -> ParagraphResult:
        spans = self._protected_spans(text, config)

        # 1) 构造带掩码的工作串：每个保护区间压成一个占位字符
        masked_chars: list[str] = []
        mask_orig: list[tuple[int, int]] = []  # 每个占位字符对应的原文区间
        mask_values: list[str] = []
        cursor = 0
        for s, e in spans:
            if cursor < s:
                for k in range(cursor, s):
                    masked_chars.append(text[k])
                    mask_orig.append((k, k + 1))
            masked_chars.append(chr(_MASK_BASE + len(mask_values)))
            mask_orig.append((s, e))
            mask_values.append(text[s:e])
            cursor = e
        for k in range(cursor, len(text)):
            masked_chars.append(text[k])
            mask_orig.append((k, k + 1))
        masked = "".join(masked_chars)

        # 2) 繁简转换（在掩码串上做，词组消歧由转换器负责）
        variant_map = self._variant_alignment(masked, config)

        # 3) 逐字符应用各条规则，产出 (cell, src) 序列
        #    src 形如：
        #      ("keep", orig_pos)
        #      ("map", rule, orig_pos)        字符被改写（cell 可能为空=删除）
        #      ("ins", rule)                  规则新增（如中英文间空格）
        #      ("mask", idx)                  保护词还原
        cells: list[str] = []
        src: list[tuple] = []

        def _emit(ch: str, source: tuple) -> None:
            cells.append(ch)
            src.append(source)

        i, n = 0, len(masked)
        while i < n:
            ch = masked[i]
            orig_pos = mask_orig[i][0]

            # 保护占位符：原样还原
            if _MASK_BASE <= ord(ch) < _MASK_BASE + len(mask_values):
                _emit(ch, ("mask", ord(ch) - _MASK_BASE))
                i += 1
                continue

            # 零宽字符
            if config.remove_zero_width and ord(ch) in _ZERO_WIDTH_CODEPOINTS:
                _emit("", ("map", "zero_width", orig_pos))
                i += 1
                continue

            # 连续空白
            if ch in " \t　":
                m = _WS_RUN_RE.match(masked, i)
                j = m.end()
                run = masked[i:j]
                if config.spaces == "collapse":
                    # 行首/行尾由收尾阶段处理；先放一个候选空格
                    _emit(" ", ("map", "space_remove", mask_orig[i][0],
                                mask_orig[j - 1][1]))
                elif config.spaces == "single":
                    for k, c in enumerate(run):
                        out = " " if c == "　" else c
                        rule = "space" if c == "　" else None
                        if rule:
                            _emit(out, ("map", rule, mask_orig[i][0] + k))
                        else:
                            _emit(out, ("keep", mask_orig[i][0] + k))
                else:
                    for k, c in enumerate(run):
                        _emit(c, ("keep", mask_orig[i][0] + k))
                i = j
                continue

            # 全角 ASCII（字母/数字/ASCII 标点/全角空格）
            fw = _fullwidth_ascii(ch)
            if fw is not None:
                if ch.isdigit():
                    if config.digits == "half":
                        _emit(fw, ("map", "digit", orig_pos))
                    else:
                        _emit(ch, ("keep", orig_pos))
                elif fw.isalpha():
                    if config.letters == "half":
                        _emit(fw, ("map", "letter", orig_pos))
                    else:
                        _emit(ch, ("keep", orig_pos))
                else:
                    if config.punct == "half":
                        _emit(fw, ("map", "punct", orig_pos))
                    else:
                        _emit(ch, ("keep", orig_pos))
                i += 1
                continue

            # CJK 标点（半角化）
            if ch in _CJK_PUNCT_HALF:
                if config.punct == "half":
                    _emit(_CJK_PUNCT_HALF[ch], ("map", "punct", orig_pos))
                else:
                    _emit(ch, ("keep", orig_pos))
                i += 1
                continue

            # 弯引号 → 直引号
            if config.normalize_quotes and _QUOTE_RE.fullmatch(ch):
                _emit(_QUOTE_MAP[ch], ("map", "quote", orig_pos))
                i += 1
                continue

            # 半角标点 → 全角（仅无歧义项）
            if config.punct == "full" and ch in _HALF_PUNCT_FULL:
                _emit(_HALF_PUNCT_FULL[ch], ("map", "punct", orig_pos))
                i += 1
                continue

            # 半角数字 → 全角
            if config.digits == "full" and ch.isascii() and ch.isdigit():
                _emit(chr(ord(ch) + 0xFEE0), ("map", "digit", orig_pos))
                i += 1
                continue

            # 半角拉丁字母 → 全角
            if config.letters == "full" and ch.isascii() and ch.isalpha():
                _emit(chr(ord(ch) + 0xFEE0), ("map", "letter", orig_pos))
                i += 1
                continue

            # 繁简转换结果
            if variant_map is not None:
                mapped = variant_map.get(i, ch)
                if mapped != ch:
                    _emit(mapped, ("map", "variant", mask_orig[i][0],
                                   mask_orig[i][1]))
                    i += 1
                    continue

            _emit(ch, ("keep", orig_pos))
            i += 1

        # 4) collapse 模式：去掉行首行尾的候选空白
        if config.spaces == "collapse":
            while cells and cells[0] == " " and src[0][0] == "map" \
                    and src[0][1] == "space_remove":
                cells.pop(0)
                src.pop(0)
            while cells and cells[-1] == " " and src[-1][0] == "map" \
                    and src[-1][1] == "space_remove":
                cells.pop()
                src.pop()

        # 5) 中英文之间补空格
        if config.space_cjk_latin is True:
            cells, src = self._insert_cjk_latin_spaces(cells, src)

        # 6) 还原保护词
        out_cells: list[str] = []
        out_src: list[tuple] = []
        for c, s in zip(cells, src):
            if s[0] == "mask":
                token = mask_values[s[1]]
                start = spans[s[1]][0]
                for k, ch in enumerate(token):
                    out_cells.append(ch)
                    out_src.append(("keep", start + k))
            else:
                out_cells.append(c)
                out_src.append(s)

        normalized = "".join(out_cells)
        changes = self._collect_changes(text, out_cells, out_src)

        profile = detect_language(text, split_spans=True)
        return ParagraphResult(
            index=index, original=text, normalized=normalized, changes=changes,
            language=profile.language, language_name=profile.language_name,
            chinese_variant=profile.chinese_variant, spans=profile.spans,
            protected=mask_values)

    @staticmethod
    def _insert_cjk_latin_spaces(cells: list[str], src: list[tuple]):
        out_cells: list[str] = []
        out_src: list[tuple] = []
        for i, ch in enumerate(cells):
            if out_cells and ch and out_cells[-1] not in (" ", "") and ch != " ":
                prev, prev_src = out_cells[-1], out_src[-1]
                cur_masked = src[i][0] == "mask"
                prev_masked = prev_src[0] == "mask"
                if not cur_masked and not prev_masked:
                    if _is_cjk(prev) and _is_latin_digit(ch):
                        out_cells.append(" ")
                        out_src.append(("ins", "space_insert"))
                    elif _is_latin_digit(prev) and _is_cjk(ch):
                        out_cells.append(" ")
                        out_src.append(("ins", "space_insert"))
            out_cells.append(ch)
            out_src.append(src[i])
        return out_cells, out_src

    # -- 变更收集 ---------------------------------------------------------
    @staticmethod
    def _collect_changes(original: str, cells: list[str],
                         src: list[tuple]) -> list[Change]:
        """按输出侧的规则标注，把连续同一规则的改动聚成一条变更。"""
        changes: list[Change] = []
        rule: Optional[str] = None
        orig_positions: list[int] = []
        changed_parts: list[str] = []

        def _flush():
            nonlocal rule, orig_positions, changed_parts
            if rule is None or not orig_positions:
                rule, orig_positions, changed_parts = None, [], []
                return
            lo, hi = min(orig_positions), max(orig_positions) + 1
            old = original[lo:hi]
            new = "".join(changed_parts)
            # 仅插入（如中英文空格）时 old 为空、new 为空格
            if old != new and (old or new.strip() or rule == "space_insert"):
                changes.append(Change(rule, lo, hi, old, new))
            rule, orig_positions, changed_parts = None, [], []

        for ch, s in zip(cells, src):
            if s[0] == "map":
                r = s[1]
                pos = s[2]
                if rule is not None and r != rule:
                    _flush()
                rule = r
                if isinstance(pos, int):
                    orig_positions.append(pos)
                changed_parts.append(ch)  # 删除类改动 ch == ""
            elif s[0] == "ins":
                r = s[1]
                if rule is not None and r != rule:
                    _flush()
                rule = r
                changed_parts.append(ch)
            else:  # keep / mask 还原
                _flush()
        _flush()

        # 兜底：用 SequenceMatcher 验证重建，若有遗漏（理论上不该有），
        # 退化为 diff 结果，保证变更集合总能还原出规范化文本。
        normalized = "".join(cells)
        rebuilt = original
        for c in sorted(changes, key=lambda x: x.start, reverse=True):
            rebuilt = rebuilt[:c.start] + c.changed + rebuilt[c.end:]
        if rebuilt != normalized:
            return _diff_fallback(original, normalized, src)
        changes.sort(key=lambda c: c.start)
        return changes

    # -- 整篇文本 ---------------------------------------------------------
    def normalize(self, text: str,
                  config: Optional[NormalizeConfig | dict] = None) -> NormalizeResult:
        config = (config if isinstance(config, NormalizeConfig)
                  else NormalizeConfig.from_dict(config))

        # 段落切分：段落与换行符交替，换行本身也可产生规范化变更（CRLF→LF）
        if config.normalize_linebreak:
            tokens = re.split(r"(\r\n|\r|\n)", text)
        else:
            tokens = re.split(r"(\n)", text)

        paragraphs: list[ParagraphResult] = []
        out_parts: list[str] = []
        protected: list[str] = []
        para_idx = 0
        offset = 0
        for ti, token in enumerate(tokens):
            if ti % 2 == 0:
                # 正文段落（首尾可能是空串：空行 / 边界）
                para = self.normalize_paragraph(token, config, index=para_idx)
                for ch in para.changes:
                    ch.start += offset
                    ch.end += offset
                for value in para.protected:
                    if value not in protected:
                        protected.append(value)
                paragraphs.append(para)
                out_parts.append(para.normalized)
                para_idx += 1
                offset += len(token)
            else:
                # 换行符
                norm_break = "\n" if config.normalize_linebreak else token
                if norm_break != token:
                    prev = paragraphs[-1] if paragraphs else None
                    change = Change(rule="linebreak", start=offset,
                                    end=offset + len(token),
                                    original=token, changed="\n")
                    if prev is not None:
                        prev.changes.append(change)
                out_parts.append(norm_break)
                offset += len(token)

        return NormalizeResult(
            original=text, normalized="".join(out_parts),
            paragraphs=paragraphs, config=config, protected=protected)


def _diff_fallback(original: str, normalized: str,
                   src: list[tuple]) -> list[Change]:
    """变更收集的兜底路径：直接对齐原文与规范文，规则按输出侧标注猜测。"""
    changes: list[Change] = []
    matcher = SequenceMatcher(None, original, normalized, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        rule = "punct"
        for j in range(j1, min(j2, len(src))):
            s = src[j]
            if s[0] in ("map", "ins"):
                rule = s[1]
                break
        old, new = original[i1:i2], normalized[j1:j2]
        if old != new:
            changes.append(Change(rule, i1, i2, old, new))
    return changes


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# 逐处确认：应用用户决策
# ---------------------------------------------------------------------------

def apply_revisions(original: str, changes: list[dict],
                    accepted_text: Optional[str] = None) -> str:
    """根据用户对逐处变更的接受/拒绝结果重建文本。

    ``changes`` 形如 ``[{"start","end","changed","accepted"}, ...]``，
    偏移基于原文。被拒绝的变更用原文片段回填，接受的用 changed 回填；
    ``accepted_text`` 不为空时表示用户手改过目标文本，直接采用。
    """
    if accepted_text is not None and accepted_text != "":
        return accepted_text
    accepted = [c for c in changes if c.get("accepted", True)]
    accepted.sort(key=lambda c: c["start"])
    out: list[str] = []
    cursor = 0
    for ch in accepted:
        s, e = ch["start"], ch["end"]
        if s < cursor:
            continue  # 重叠变更忽略后到者
        out.append(original[cursor:s])
        out.append(ch.get("changed", original[s:e]))
        cursor = e
    out.append(original[cursor:])
    return "".join(out)
