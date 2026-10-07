"""内置流水线阶段：把 NLP 算法封装成可编排的阶段。"""

from __future__ import annotations

from nlp import (get_keywords, get_ner, get_parser, get_segmenter,
                 get_sentiment, get_summarizer, get_tagger, get_translator,
                 get_constituency_parser, normalize_text,
                 apply_norm_decisions)
from nlp.langid import profile as lang_profile
from nlp.lexicon import STOPWORDS

from .stage import Stage


def _source_text(ctx) -> str:
    """统一的取文入口：规范化阶段之后一律使用规范文本，保证口径一致。"""
    return (ctx.get("normalized_text")
            or ctx.get("clean_text")
            or ctx.get("text", ""))


def _norm_params(params) -> dict:
    """从阶段参数中抽取规范化配置（忽略无关键）。"""
    keys = ("script", "punctuation", "digits", "letters", "quotes", "ellipsis",
            "whitespace", "remove_cjk_spaces", "protect", "protect_regex")
    return {k: params[k] for k in keys if k in params}


def _normalize(ctx, params):
    text = ctx.get("text", "")
    spec = _norm_params(params)
    plan = normalize_text(text, spec or None)
    # 允许在批量执行时通过 decisions 传入人工确认结果（回流时口径固定）
    decisions = params.get("decisions") or ctx.get("norm_decisions")
    if decisions:
        final = apply_norm_decisions(plan, decisions)
        normalized = final["normalized"]
        plan["norm_stats"] = final["stats"]
    else:
        normalized = plan["normalized"]
    return {
        "normalized_text": normalized,
        "norm_plan": plan,
        "language": plan["profile"],
    }


def _clean(ctx, params):
    text = _source_text(ctx)
    import re
    # 去空白、统一标点
    text = re.sub(r"\s+", " ", text).strip()
    if params.get("remove_stopwords", True):
        seg = get_segmenter()
        words = [w for w in seg.cut(text) if w not in STOPWORDS]
        return {"clean_text": " ".join(words)}
    return {"clean_text": text}


def _segment(ctx, params):
    seg = get_segmenter()
    return {"words": seg.cut(_source_text(ctx))}


def _pos(ctx, params):
    tagger = get_tagger()
    text = _source_text(ctx)
    return {"pos": [[w, t] for w, t in tagger.tag(text)]}


def _ner(ctx, params):
    ner = get_ner()
    return {"ner": ner.recognize(_source_text(ctx))}


def _sentiment(ctx, params):
    return {"sentiment": get_sentiment().analyze(_source_text(ctx))}


def _keywords(ctx, params):
    return {"keywords": get_keywords().extract(
        _source_text(ctx), top_k=params.get("top_k", 10))}


def _summary(ctx, params):
    return {"summary": get_summarizer().summarize(
        _source_text(ctx), ratio=params.get("ratio", 0.3),
        max_sentences=params.get("max_sentences"))}


def _translate(ctx, params):
    return {"translation": get_translator().translate(
        _source_text(ctx), direction=params.get("direction", "zh2en"))}


def _parse(ctx, params):
    text = _source_text(ctx)
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    return {"parse": {"dependency": dep, "constituency": const}}


def _language_id(ctx, params):
    return {"language": lang_profile(_source_text(ctx)).to_dict()}


NORMALIZE_DEFAULT_PARAMS = {
    "script": "simplified",
    "punctuation": "half",
    "digits": "half",
    "letters": "half",
    "quotes": "keep",
    "ellipsis": "collapse",
    "whitespace": "collapse",
    "remove_cjk_spaces": True,
    "protect": [],
    "protect_regex": [],
}


BUILTIN_STAGES = [
    Stage("normalize", _normalize, inputs=["text"],
          outputs=["normalized_text", "norm_plan", "language"],
          description="语言识别与文本规范化（繁简/全半角/空白，可逐处确认）",
          params=dict(NORMALIZE_DEFAULT_PARAMS)),
    Stage("language_id", _language_id,
          inputs=["text", "normalized_text"], outputs=["language"],
          description="语种识别（中英/繁简/混合）"),
    Stage("clean", _clean, inputs=["text", "normalized_text"], outputs=["clean_text"],
          description="文本清洗：去空白、去停用词", params={"remove_stopwords": True}),
    Stage("segment", _segment,
          inputs=["text", "clean_text", "normalized_text"], outputs=["words"],
          description="中文分词"),
    Stage("pos", _pos,
          inputs=["text", "clean_text", "normalized_text"], outputs=["pos"],
          description="词性标注"),
    Stage("ner", _ner,
          inputs=["text", "clean_text", "normalized_text"], outputs=["ner"],
          description="命名实体识别"),
    Stage("sentiment", _sentiment,
          inputs=["text", "clean_text", "normalized_text"], outputs=["sentiment"],
          description="情感分析"),
    Stage("keywords", _keywords,
          inputs=["text", "clean_text", "normalized_text"], outputs=["keywords"],
          description="关键词提取", params={"top_k": 10}),
    Stage("summary", _summary,
          inputs=["text", "clean_text", "normalized_text"], outputs=["summary"],
          description="文本摘要", params={"ratio": 0.3}),
    Stage("translate", _translate,
          inputs=["text", "clean_text", "normalized_text"], outputs=["translation"],
          description="机器翻译（模拟）", params={"direction": "zh2en"}),
    Stage("parse", _parse,
          inputs=["text", "clean_text", "normalized_text"], outputs=["parse"],
          description="句法分析"),
]
