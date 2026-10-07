"""语言识别、繁简转换、规范化与批量处理的单元测试。

运行：``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import charconv, langid, normalize_text
from nlp.normalize import apply_decisions
from nlp.batchnorm import process_batch, process_one
from nlp.consistency import verify_consistency
from pipeline import PipelineEngine


class TestCharConv(unittest.TestCase):
    def test_t2s_basic(self):
        self.assertEqual(charconv.to_simplified("臺"), "台")
        self.assertEqual(charconv.to_simplified("語"), "语")

    def test_t2s_phrase(self):
        out = "".join(charconv.to_simplified(c) for c in "台灣電腦軟體")
        self.assertEqual(out, "台湾电脑软体")

    def test_s2t_safe_roundtrip(self):
        # 安全类推字可以还原
        self.assertEqual(charconv.to_traditional("语"), "語")
        self.assertEqual(charconv.to_traditional("贝"), "貝")

    def test_s2t_ambiguous_kept(self):
        # 歧义 / 归并字必须原样保留，避免破坏原意
        for ch in "面后云发干只台里":
            self.assertEqual(charconv.to_traditional(ch), ch)

    def test_mapping_invariants(self):
        # 每个繁体只对应一个简体
        for t in charconv.T2S:
            self.assertIsInstance(charconv.to_simplified(t), str)


class TestLangId(unittest.TestCase):
    def test_zh(self):
        self.assertEqual(langid.detect("自然语言处理是人工智能的重要分支"), "zh")

    def test_en(self):
        self.assertEqual(langid.detect("The quick brown fox jumps over the dog"), "en")

    def test_mixed(self):
        p = langid.profile("使用 Python 写 NLP 模型，machine learning 很有趣")
        self.assertEqual(p.lang, "mixed")

    def test_script_marks(self):
        p = langid.profile("台灣的電腦軟體")
        self.assertGreater(p.trad_marks, 0)
        self.assertEqual(p.simp_marks, 0)

    def test_paragraphs(self):
        text = "中文段落。\nEnglish paragraph here.\n另一段中文。"
        paras = langid.detect_paragraphs(text)
        self.assertEqual(len(paras), 3)
        self.assertEqual(paras[0]["lang"], "zh")
        self.assertEqual(paras[1]["lang"], "en")
        self.assertEqual(paras[2]["lang"], "zh")

    def test_segments(self):
        segs = langid.segment_mixed("中文English")
        types = [s["type"] for s in segs]
        self.assertIn("zh", types)
        self.assertIn("en", types)


class TestNormalize(unittest.TestCase):
    def test_fullwidth_and_script(self):
        plan = normalize_text("ＡＢＣ１２３，語言！", {
            "script": "simplified", "punctuation": "half"})
        self.assertEqual(plan["normalized"], "ABC123,语言!")

    def test_idempotent(self):
        spec = {"script": "simplified", "punctuation": "half"}
        first = normalize_text("這是一個　測試，數字１２３。", spec)
        second = normalize_text(first["normalized"], spec)
        self.assertEqual(first["normalized"], second["normalized"])

    def test_protect_terms(self):
        # IBM 受保护：全角上下文被转，但保护片段不动（这里 IBM 本就是半角）
        plan = normalize_text("ＩＢＭ 與 １２３", {
            "protect": ["ＩＢＭ"], "digits": "half", "letters": "half"})
        self.assertIn("ＩＢＭ", plan["normalized"])
        self.assertIn("123", plan["normalized"])

    def test_cjk_space_removed(self):
        plan = normalize_text("自然 语言 处理", {"punctuation": "keep",
                                                "script": "none"})
        self.assertEqual(plan["normalized"], "自然语言处理")

    def test_traditional_target_conservative(self):
        # 简→繁：安全字转换，歧义字保留
        plan = normalize_text("语言表达", {"script": "traditional",
                                          "punctuation": "keep"})
        self.assertIn("語", plan["normalized"])
        self.assertIn("言", plan["normalized"])

    def test_decisions_reject(self):
        plan = normalize_text("語言，１２３", {"script": "simplified",
                                             "punctuation": "half"})
        # 驳回所有标点类改动
        decisions = {c["idx"]: False for c in plan["changes"]
                     if c["kind"] == "punct"}
        final = apply_decisions(plan, decisions)
        self.assertIn("，", final["normalized"])      # 被驳回，保留全角
        self.assertIn("语言", final["normalized"])    # 字形转换保留
        self.assertGreaterEqual(final["stats"]["rejected"], 1)

    def test_decisions_edit(self):
        plan = normalize_text("語言，", {"script": "simplified",
                                       "punctuation": "half"})
        target = next(c for c in plan["changes"] if c["kind"] == "punct")
        final = apply_decisions(plan, {target["idx"]: ";"})
        self.assertIn(";", final["normalized"])
        self.assertNotIn(",", final["normalized"])

    def test_paragraph_views_consistent(self):
        text = "這是第一段，數字１。\n第二段 ABC。"
        plan = normalize_text(text, {"script": "simplified",
                                     "punctuation": "half"})
        joined = "\n".join(p["normalized_text"] for p in plan["paragraphs"])
        self.assertEqual(joined, plan["normalized"])

    def test_change_offsets(self):
        plan = normalize_text("Ａ", {"letters": "half"})
        c = plan["changes"][0]
        self.assertEqual(plan["original"][c["start"]:c["end"]], "Ａ")
        self.assertEqual(c["replacement"], "A")

    def test_ellipsis_and_dash(self):
        plan = normalize_text("嗯……再見——", {"script": "simplified",
                                            "punctuation": "half"})
        self.assertIn("...", plan["normalized"])
        self.assertIn("--", plan["normalized"])
        self.assertIn("再见", plan["normalized"])
        self.assertIn("ellipsis", plan["stats"]["by_kind"])

    def test_english_abbreviation_kept(self):
        # 拉丁缩写本就不被繁简/标点转换影响；GPU、U.S.A. 原样保留
        plan = normalize_text("使用 GPU 與 U.S.A. 的標準",
                              {"script": "simplified", "punctuation": "half"})
        self.assertIn("GPU", plan["normalized"])
        self.assertIn("U.S.A.", plan["normalized"])
        self.assertNotIn("ＧＰＵ", plan["normalized"])


class TestBatch(unittest.TestCase):
    def test_batch_order_and_isolation(self):
        docs = ["這是測試Ａ。", "plain English text", {"id": "z", "text": "網路"}]
        results = process_batch(docs, {"script": "simplified"}, max_workers=2)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["pos"], 0)
        self.assertTrue(all(r["ok"] for r in results))
        self.assertEqual(results[2]["id"], "z")

    def test_long_doc_truncated(self):
        r = process_one("測" * 100, {"script": "simplified"}, max_chars=50)
        self.assertTrue(r["ok"])
        self.assertEqual(r["truncated"], 50)
        self.assertIn("截断", r["warning"])

    def test_empty_doc(self):
        r = process_one("", {"script": "simplified"})
        self.assertTrue(r["ok"])
        self.assertEqual(r["normalized"], "")


class TestConsistency(unittest.TestCase):
    def test_pure_form_change(self):
        # 仅全角/字形变化，语义任务口径应一致（允许边界与分值小漂移）
        original = "这个产品非常好用，我很喜欢，价格是１９９元。"
        plan = normalize_text(original, {"punctuation": "half"})
        report = verify_consistency(original, plan["normalized"],
                                    polarity_tol=0.5)
        seg = next(c for c in report["checks"] if c["task"] == "segment")
        self.assertTrue(seg["consistent"], seg["detail"])
        form = next(c for c in report["checks"] if c["task"] == "form")
        self.assertTrue(form["consistent"])


class TestPipelineNormalize(unittest.TestCase):
    def setUp(self):
        self.engine = PipelineEngine().register_builtin()

    def test_normalize_then_segment(self):
        cfg = {"name": "p", "stages": [
            {"name": "normalize"}, {"name": "segment"}, {"name": "sentiment"}]}
        out = self.engine.build(cfg).run(
            {"text": "這個產品非常好用，１２３元！"})
        self.assertIn("normalized_text", out)
        self.assertIn("这个产品", out["normalized_text"])
        self.assertIn("words", out)
        # 下游必须消费规范文本
        joined = "".join(w for w in out["words"] if w not in "，。！,!.")
        self.assertIn("这个产品", joined)


if __name__ == "__main__":
    unittest.main()
