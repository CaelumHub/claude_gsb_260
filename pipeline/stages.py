"""内置流水线阶段：把 NLP 算法封装成可编排的阶段。"""

from __future__ import annotations

from nlp import (get_keywords, get_ner, get_parser, get_segmenter,
                 get_sentiment, get_summarizer, get_tagger, get_translator,
                 get_constituency_parser, get_normalizer)
from nlp.normalize import NormalizeConfig
from nlp.lexicon import STOPWORDS

from .stage import Stage


def _source_text(ctx) -> str:
    """下游任务的取数优先级：清洗文本 > 规范文本 > 原始文本。"""
    return ctx.get("clean_text") or ctx.get("normalized_text") or ctx.get("text", "")


def _normalize(ctx, params):
    """语言识别 + 文本规范化；产出 normalized_text 供下游统一口径。"""
    text = ctx.get("text", "")
    config = NormalizeConfig.from_dict(params)
    result = get_normalizer().normalize(text, config)
    first = result.paragraphs[0] if result.paragraphs else None
    return {
        "normalized_text": result.normalized,
        "normalize_report": {
            "change_count": sum(len(p.changes) for p in result.paragraphs),
            "language": first.language if first else "unknown",
            "chinese_variant": first.chinese_variant if first else "unknown",
        },
    }


def _clean(ctx, params):
    import re
    # 优先在规范化后的文本上清洗（全半角/繁简已统一）
    text = ctx.get("normalized_text") or ctx.get("text", "")
    # 去空白、统一标点
    text = re.sub(r"\s+", " ", text).strip()
    if params.get("remove_stopwords", True):
        seg = get_segmenter()
        words = [w for w in seg.cut(text) if w not in STOPWORDS]
        return {"clean_text": " ".join(words)}
    return {"clean_text": text}


def _segment(ctx, params):
    seg = get_segmenter()
    text = _source_text(ctx)
    return {"words": seg.cut(text)}


def _pos(ctx, params):
    tagger = get_tagger()
    text = _source_text(ctx)
    return {"pos": [[w, t] for w, t in tagger.tag(text)]}


def _ner(ctx, params):
    ner = get_ner()
    text = _source_text(ctx)
    return {"ner": ner.recognize(text)}


def _sentiment(ctx, params):
    text = _source_text(ctx)
    return {"sentiment": get_sentiment().analyze(text)}


def _keywords(ctx, params):
    text = _source_text(ctx)
    return {"keywords": get_keywords().extract(text, top_k=params.get("top_k", 10))}


def _summary(ctx, params):
    text = _source_text(ctx)
    return {"summary": get_summarizer().summarize(
        text, ratio=params.get("ratio", 0.3),
        max_sentences=params.get("max_sentences"))}


def _translate(ctx, params):
    text = _source_text(ctx)
    return {"translation": get_translator().translate(
        text, direction=params.get("direction", "zh2en"))}


def _parse(ctx, params):
    text = _source_text(ctx)
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    return {"parse": {"dependency": dep, "constituency": const}}


BUILTIN_STAGES = [
    Stage("normalize", _normalize, inputs=["text"],
          outputs=["normalized_text", "normalize_report"],
          description="语言识别与文本规范化（全半角/繁简/数字/空白）",
          params=NormalizeConfig().to_dict()),
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
