"""Automated tests cho pipeline truy xuat (khong can index, LLM hay torch)."""
import os
import unittest

# Ep reranker bo trong de su dung lexical rerank (khong tai torch/sentence-transformers).
os.environ.setdefault("RERANKER_MODEL", "")

from langchain_core.documents import Document  # noqa: E402

from rag.bm25_store import BM25Index, BM25Store, tokenize  # noqa: E402
from rag.chunking import split_parent_child  # noqa: E402
from rag.query_engine import (  # noqa: E402
    RankedDocument,
    boost_article_intent,
    expand_abbreviations,
    hybrid_search,
    _lexical_rerank_score,
    _merge_ranked_hits,
    _min_max,
)
from services.vector_rag_service.query_rewrite import _strip_marks, _word_change  # noqa: E402


def doc(code: str, title: str = "", content: str = "x") -> Document:
    return Document(
        page_content=content,
        metadata={"article_code": code, "article_title": title, "source_file": "x.html", "parent_id": code},
    )


def child(code: str, title: str = "", content: str = "x", index: int = 0) -> Document:
    d = Document(
        page_content=content,
        metadata={"article_code": code, "article_title": title, "source_file": "x.html", "parent_id": code},
    )
    d.metadata["child_index"] = index
    return d


class FakeVectorstore:
    def __init__(self, hits):
        self.hits = hits

    def similarity_search_with_score(self, question, k):
        return self.hits[:k]


class FakeBM25Store:
    def __init__(self, hits):
        self.hits = hits

    def search_with_scores(self, query, top_k):
        return self.hits[:top_k]


class FakeStore:
    def __init__(self, vector_hits, bm25_hits, parents, children):
        self.vectorstore = FakeVectorstore(vector_hits)
        self.bm25_store = FakeBM25Store(bm25_hits)
        self.parents = parents
        self.children = children
        self.manifest = {}


class TokenizeTest(unittest.TestCase):
    def test_tokenize_lowercases_and_strips_diacritics(self):
        assert tokenize("Chứng minh nhân dân CMND") == [
            "chứng",
            "minh",
            "nhân",
            "dân",
            "cmnd",
        ]

    def test_tokenize_empty(self):
        assert tokenize("   ") == []


class BM25Test(unittest.TestCase):
    def test_top_k_returns_best_matching_doc(self):
        index = BM25Index([["chứng", "minh", "nhân", "dân"], ["thuế", "sử dụng", "đất"]])
        res = index.top_k(["chứng", "minh"], 1)
        assert res[0][0] == 0
        assert res[0][1] >= 0.0

    def test_top_k_respects_k(self):
        index = BM25Index([["a"], ["b"], ["c"]])
        assert len(index.top_k(["a", "b", "c"], 2)) == 2


class RRFTest(unittest.TestCase):
    def test_fusion_sums_weighted_ranks(self):
        d1, d2 = doc("Điều 1"), doc("Điều 2")
        cands = {}
        _merge_ranked_hits(cands, [(d1, 0.1), (d2, 0.2)], "vector", 0.5)
        _merge_ranked_hits(cands, [(d1, 0.1)], "bm25", 0.5)
        assert cands["Điều 1"].score == 0.5 * (1 / (1 + 60)) + 0.5 * (1 / (1 + 60))
        assert cands["Điều 2"].score == 0.5 * (1 / (2 + 60))
        assert cands["Điều 1"].vector_rank == 1
        assert cands["Điều 1"].bm25_rank == 1


class AbbreviationsTest(unittest.TestCase):
    def test_expands_short_form(self):
        assert expand_abbreviations("CMND") == "CMND chứng minh nhân dân"
        assert expand_abbreviations("UBND") == "UBND ủy ban nhân dân"

    def test_keeps_sentence_when_no_abbreviation(self):
        assert expand_abbreviations("không có viết tắt") == "không có viết tắt"

    def test_expands_both_directions(self):
        assert expand_abbreviations("chứng minh nhân dân") == "chứng minh nhân dân CMND"


class IntentBoostTest(unittest.TestCase):
    def test_boosts_matching_intent(self):
        cands = [
            RankedDocument(document=doc("Điều 1", "Đối tượng áp dụng"), score=0.3),
            RankedDocument(document=doc("Điều 2", "Khoản đã qua"), score=0.4),
        ]
        out = boost_article_intent("những ai phải tuân theo", cands)
        scores = {r.document.metadata["article_code"]: r.score for r in out}
        assert abs(scores["Điều 1"] - 0.65) < 1e-9
        assert scores["Điều 2"] == 0.4

    def test_no_boost_when_no_intent(self):
        cands = [RankedDocument(document=doc("Điều 1", "Phạm vi điều chỉnh"), score=0.5)]
        out = boost_article_intent("xin chào", cands)
        assert out[0].score == 0.5


class MinMaxTest(unittest.TestCase):
    def test_normalizes_to_zero_one(self):
        assert _min_max([0, 10]) == [0.0, 1.0]

    def test_constant_values_all_zero(self):
        assert _min_max([5, 5]) == [0.0, 0.0]

    def test_single_value_zero(self):
        assert _min_max([42]) == [0.0]


class LexicalRerankTest(unittest.TestCase):
    def test_score_positive_when_query_in_text(self):
        assert _lexical_rerank_score("chứng minh nhân dân", "chứng minh nhân dân là gì", {}) > 0

    def test_score_zero_when_no_overlap(self):
        assert _lexical_rerank_score("xyz abc", "thuế sử dụng đất", {}) == 0.0

    def test_phrase_boost_adds_points(self):
        base = _lexical_rerank_score("hiệu lực", "hiệu lực", {})
        assert _lexical_rerank_score("hiệu lực", "có hiệu lực", {}) >= base


class RewriteNormalizeTest(unittest.TestCase):
    def test_strip_marks(self):
        assert _strip_marks("Thuế") == "thue"
        assert _strip_marks("Đông") == "dong"
        assert _strip_marks("CMND") == "cmnd"

    def test_word_change_unchanged_after_diacritics(self):
        assert _word_change("thue thue", "thue thue") == 0.0

    def test_word_change_detects_extra_words(self):
        assert _word_change("thue", "thue khác") > 0


class ChunkingTest(unittest.TestCase):
    def test_children_keep_header_and_parent_id(self):
        header = "Điều 1 Phạm vi điều chỉnh"
        parent = Document(
            page_content=f"{header}\n1. Nội dung khoá a)\n2. Nội dung khoá b)",
            metadata={
                "article_code": "Điều 1",
                "article_title": "Phạm vi điều chỉnh",
                "parent_id": "Điều 1",
                "source_file": "x.html",
            },
        )
        children = split_parent_child([parent], lambda t: len(t.split()), 50)
        assert len(children) >= 1
        assert all(c.metadata["parent_id"] == "Điều 1" for c in children)
        assert any(c.page_content.startswith(header) for c in children)
        assert all(len(c.page_content.split()) <= 60 for c in children)


class FakeRetrievalTest(unittest.TestCase):
    def test_hybrid_search_returns_both_sources(self):
        p1 = doc("Điều 1", "A", "A")
        p2 = doc("Điều 2", "B", "B")
        c1 = child("Điều 1", "A", "A child", 0)
        # FAISS tra ve child (co child_index); candidate_hits gom ve parent.
        store = FakeStore(
            [(c1, 0.1)],
            [(p1, 0.1), (p2, 0.2)],
            {"Điều 1": p1, "Điều 2": p2},
            {"Điều 1": [c1]},
        )
        res = hybrid_search("A", store, top_k=2, candidate_k=2)
        codes = [r.document.metadata["article_code"] for r in res]
        assert "Điều 1" in codes
        assert len(res) <= 2


if __name__ == "__main__":
    unittest.main()
