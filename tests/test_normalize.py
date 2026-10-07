"""语言识别与文本规范化测试。

运行：``python -m unittest tests.test_normalize -v``
覆盖：主语言判定、中英混排分段、繁简转换（含词组消歧）、
全半角 / 数字 / 空白规范化、逐处确认、保护词、下游口径一致性、
存储 update 与规范化流水线阶段。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import (detect_language, split_languages, get_normalizer, get_converter,
                 apply_revisions, run_consistency_checks)
from nlp.normalize import NormalizeConfig
from nlp.langid import find_abbreviations
from pipeline import PipelineEngine
from storage import ShardedStore


class TestLanguageId(unittest.TestCase):
    def test_english(self):
        p = detect_language("This is an English sentence.")
        self.assertEqual(p.language, "en")
        self.assertGreater(p.confidence, 0.9)

    def test_chinese(self):
        p = detect_language("这是一个纯粹的中文句子")
        self.assertEqual(p.language, "zh")
        self.assertEqual(p.chinese_variant, "simplified")

    def test_traditional_detected(self):
        p = detect_language("這裡是繁體中文的範例")
        self.assertEqual(p.language, "zh")
        self.assertEqual(p.chinese_variant, "traditional")

    def test_mixed_spans(self):
        spans = split_languages("使用 Python 和 GPU 加速")
        langs = [s.lang for s in spans if s.text.strip()]
        self.assertIn("zh", langs)
        self.assertIn("en", langs)
        # 英文段内容应完整
        en = [s.text for s in spans if s.lang == "en"]
        self.assertTrue(any("Python" in t for t in en))

    def test_punct_attaches_to_cjk(self):
        # 句末中文标点应跟随中文段，而不是变成孤立符号段
        spans = split_languages("你好world！")
        zh = "".join(s.text for s in spans if s.lang == "zh")
        self.assertIn("！", zh)

    def test_abbreviation(self):
        text = "GPU and U.S.A. are abbreviations"
        spans = find_abbreviations(text)
        self.assertIn((0, 3), spans)                       # GPU
        self.assertEqual(text[8:14], "U.S.A.")
        self.assertIn((8, 14), spans)                      # U.S.A.


class TestChineseConverter(unittest.TestCase):
    def setUp(self):
        self.c = get_converter()

    def test_basic_t2s(self):
        self.assertEqual(self.c.to_simplified("臺灣與香港"), "台湾与香港")
        self.assertEqual(self.c.to_simplified("魚"), "鱼")

    def test_keep_proper_phrases(self):
        # 乾隆 / 乾坤 绝不能被逐字转换成"干"
        self.assertEqual(self.c.to_simplified("乾隆皇帝"), "乾隆皇帝")
        self.assertEqual(self.c.to_simplified("乾坤"), "乾坤")

    def test_phrase_disambiguation(self):
        self.assertEqual(self.c.to_simplified("乾杯"), "干杯")
        self.assertEqual(self.c.to_simplified("頭髮"), "头发")
        self.assertEqual(self.c.to_simplified("麵條"), "面条")
        self.assertEqual(self.c.to_simplified("看著"), "看着")

    def test_s2t_phrase_disambiguation(self):
        # 发→發/髮 的区分
        self.assertEqual(self.c.to_traditional("头发"), "頭髮")
        self.assertEqual(self.c.to_traditional("发现"), "發現")
        # 后→後，皇后保留
        self.assertEqual(self.c.to_traditional("以后"), "以後")
        self.assertEqual(self.c.to_traditional("皇后"), "皇后")

    def test_identity_roundtrip_words(self):
        # 简体身份词组在转繁时应保持
        self.assertEqual(self.c.to_traditional("游泳"), "游泳")
        self.assertEqual(self.c.to_traditional("胡同"), "衚衕")


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.norm = get_normalizer()

    def test_fullwidth_halfwidth(self):
        r = self.norm.normalize("ＡＢＣ１２３！")
        self.assertEqual(r.normalized, "ABC123!")

    def test_punct_to_half(self):
        r = self.norm.normalize("你好，世界。")
        self.assertEqual(r.normalized, "你好,世界.")
        rules = {c.rule for p in r.paragraphs for c in p.changes}
        self.assertIn("punct", rules)

    def test_t2s_with_variant(self):
        r = self.norm.normalize("這裡是臺灣")
        self.assertEqual(r.normalized, "这里是台湾")
        rules = {c.rule for p in r.paragraphs for c in p.changes}
        self.assertIn("variant", rules)

    def test_change_reconstruction(self):
        text = "Ｈello，這裡是　ＧＰＵ！"
        r = self.norm.normalize(text)
        all_changes = [c.to_dict() for p in r.paragraphs for c in p.changes]
        # 全接受 → 规范化文本
        self.assertEqual(apply_revisions(text, all_changes), r.normalized)
        # 全拒绝 → 原文
        for c in all_changes:
            c["accepted"] = False
        self.assertEqual(apply_revisions(text, all_changes), text)

    def test_reject_single_change(self):
        text = "ＡＢＣ，１２３"
        r = self.norm.normalize(text)
        changes = [c.to_dict() for p in r.paragraphs for c in p.changes]
        # 拒绝第一处（字母转换）
        changes[0]["accepted"] = False
        out = apply_revisions(text, changes)
        self.assertTrue(out.startswith("ＡＢＣ"))  # 字母保留
        self.assertIn("123", out)                 # 数字仍转换

    def test_protected_abbrev_unchanged(self):
        # GPU 是缩写，应被保护；但全角字母不构成缩写，仍会转换
        text = "使用GPU集群，Ｘ１"
        r = self.norm.normalize(text, {"letters": "half"})
        self.assertIn("GPU", r.normalized)
        self.assertIn("GPU", r.protected)

    def test_protected_terms(self):
        text = "OpenAI 发布了新模型"
        r = self.norm.normalize(text, {"protected_terms": ["OpenAI"]})
        self.assertIn("OpenAI", r.normalized)
        self.assertIn("OpenAI", r.protected)

    def test_zero_width_removed(self):
        text = "hello​world"
        r = self.norm.normalize(text)
        self.assertNotIn("​", r.normalized)
        self.assertTrue(any(c.rule == "zero_width"
                            for p in r.paragraphs for c in p.changes))

    def test_space_collapse(self):
        r = self.norm.normalize("a　　b    c")
        self.assertEqual(r.normalized, "a b c")

    def test_linebreak_normalize(self):
        r = self.norm.normalize("第一行\r\n第二行\r第三行")
        self.assertEqual(r.normalized, "第一行\n第二行\n第三行")
        self.assertEqual(len(r.paragraphs), 3)

    def test_keep_variant(self):
        text = "這裡是臺灣"
        r = self.norm.normalize(text, {"variant": "keep", "punct": "keep",
                                       "digits": "keep", "letters": "keep",
                                       "spaces": "keep"})
        self.assertEqual(r.normalized, text)
        self.assertEqual(sum(len(p.changes) for p in r.paragraphs), 0)

    def test_paragraph_language_metadata(self):
        r = self.norm.normalize("中文段落\nEnglish paragraph")
        langs = [p.language for p in r.paragraphs]
        self.assertEqual(langs, ["zh", "en"])

    def test_spans_count_matches_text(self):
        text = "中英 mixed 混排 text"
        p = detect_language(text, split_spans=True)
        self.assertEqual("".join(s.text for s in p.spans), text)


class TestConsistency(unittest.TestCase):
    def test_punct_only_keeps_sentiment(self):
        # 仅标点全半角 + 繁简差异，情感极性不应翻转
        original = "這個產品非常好用，我很喜歡！"
        normalized = "这个产品非常好用,我很喜欢!"
        rep = run_consistency_checks(original, normalized,
                                     tasks=("segment", "sentiment"))
        sent = next(c for c in rep.checks if c["task"] == "sentiment")
        self.assertTrue(sent["consistent"])
        seg = next(c for c in rep.checks if c["task"] == "segment")
        # 繁简统一后内容词序列应一致
        self.assertTrue(seg["consistent"], seg.get("first_diff"))


class TestNormalizePipelineStage(unittest.TestCase):
    def test_normalize_feeds_downstream(self):
        eng = PipelineEngine().register_builtin()
        cfg = {"name": "p", "stages": [
            {"name": "normalize"}, {"name": "segment"}, {"name": "sentiment"}]}
        out = eng.build(cfg).run({"text": "這裡是ＧＰＵ集群，非常好用！"})
        self.assertEqual(out["normalized_text"], "这里是GPU集群,非常好用!")
        # 下游在规范文本上分词
        self.assertIn("这里", out["words"])
        self.assertIn("GPU", out["words"])

    def test_batch_one_failure_isolated(self):
        eng = PipelineEngine().register_builtin()
        cfg = {"name": "p", "stages": [{"name": "normalize"}]}
        results = eng.run_batch(cfg, ["正常文本", "", "另一段"], max_workers=2)
        self.assertEqual(len(results), 3)
        # 空字符串也能规范化（不应抛错）
        self.assertTrue(all(r and r["ok"] for r in results))


class TestStoreUpdate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = ShardedStore(self.dir, "normalize_batch", shard_size=4)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_update_progress(self):
        rid = self.store.insert({"status": "running", "done": 0})
        updated = self.store.update(rid, {"done": 5, "status": "finished"})
        self.assertEqual(updated["done"], 5)
        self.assertEqual(updated["status"], "finished")
        again = self.store.get(rid)
        self.assertEqual(again["done"], 5)

    def test_update_missing(self):
        self.assertIsNone(self.store.update("nope", {"x": 1}))


if __name__ == "__main__":
    unittest.main()
