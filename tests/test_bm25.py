import math
import re

import numpy as np

from esci.retrieval.bm25 import (
    TokenizerSpec,
    build_index,
    index_size_bytes,
    load_index,
    save_index,
    search,
)

DOCS = [
    "red running shoes for men",
    "blue running shorts lightweight",
    "leather office chair with arms",
    "red leather wallet for men",
    "wireless mouse for office use",
    "running shoes red red sale",
    "steel water bottle blue",
    "gaming chair red black",
]
PATTERN = r"(?u)\b\w\w+\b"


def _tok(text):
    return re.findall(PATTERN, text.lower())


def _reference_scores(query, docs, k1, b):
    """Plain BM25 with the lucene idf, no stopwords, no stemming."""
    toks = [_tok(d) for d in docs]
    n = len(toks)
    avgdl = sum(len(t) for t in toks) / n
    scores = np.zeros(n)
    for term in set(_tok(query)):
        df = sum(1 for t in toks if term in t)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, t in enumerate(toks):
            tf = t.count(term)
            if tf:
                scores[i] += idf * tf / (tf + k1 * (1 - b + b * len(t) / avgdl))
    return scores


def test_matches_reference_ranking_and_scale():
    spec = TokenizerSpec()
    retriever, _ = build_index(DOCS, spec, k1=1.2, b=0.75)
    for query in ["red running shoes", "office chair", "blue bottle", "red men"]:
        docs, scores = search(retriever, spec, [query], k=len(DOCS))
        ref = _reference_scores(query, DOCS, 1.2, 0.75)
        got = np.zeros(len(DOCS))
        got[docs[0]] = scores[0]
        hit = ref > 0
        assert np.array_equal(hit, got > 0)
        ratio = got[hit] / ref[hit]
        assert np.allclose(ratio, ratio[0], rtol=1e-4)
        assert np.array_equal(
            np.argsort(-got[hit], kind="stable"), np.argsort(-ref[hit], kind="stable")
        )


def test_doc_ids_are_row_positions():
    spec = TokenizerSpec()
    retriever, _ = build_index(DOCS, spec)
    docs, _ = search(retriever, spec, ["wireless mouse"], k=1)
    assert docs[0][0] == 4


def test_unknown_query_terms_do_not_crash():
    spec = TokenizerSpec()
    retriever, _ = build_index(DOCS, spec)
    docs, scores = search(retriever, spec, ["zzzzqqq"], k=3)
    assert docs.shape == (1, 3)
    assert np.all(scores == 0)


def test_stemming_and_stopwords_change_tokens():
    plain = TokenizerSpec().tokenize_queries(["the running shoes"])[0]
    full = TokenizerSpec(stopwords=True, stem=True).tokenize_queries(["the running shoes"])[0]
    assert "the" in plain
    assert "the" not in full
    assert "run" in full


def test_save_load_round_trip(tmp_path):
    spec = TokenizerSpec(stopwords=True, stem=True)
    retriever, _ = build_index(DOCS, spec, k1=1.2, b=0.75)
    save_index(retriever, spec, tmp_path / "idx", {"k1": 1.2, "b": 0.75})
    loaded, loaded_spec, params = load_index(tmp_path / "idx")
    assert loaded_spec == spec
    assert params == {"k1": 1.2, "b": 0.75}
    before = search(retriever, spec, ["red running shoes"], k=5)
    after = search(loaded, loaded_spec, ["red running shoes"], k=5)
    assert np.array_equal(before[0], after[0])
    assert np.allclose(before[1], after[1])
    assert index_size_bytes(tmp_path / "idx") > 0
