import pandas as pd
import pytest

from esci.data import corpus as cp


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["B3", "B1", "B2", "B9", "B1"],
            "product_locale": ["us", "us", "us", "es", "us"],
            "product_title": ["Third", "First", "Second", "Spanish", "Duplicate"],
            "product_brand": ["Acme", "Acme", "Zeta", "Uno", "Acme"],
            "product_color": ["Red", "Blue", "", "Verde", "Blue"],
            "product_bullet_point": ["c", "a", "b", "d", "a"],
        }
    )


@pytest.fixture
def examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "product_id": ["B1", "B2", "B3", "B404"],
            "esci_label": ["E", "S", "E", "I"],
        }
    )


def test_corpus_contains_only_in_scope_us_products(products, examples):
    corpus = cp.build_corpus(products, examples, "us")
    assert set(corpus["product_id"]) == {"B1", "B2", "B3"}


def test_doc_ids_are_contiguous_and_sorted_by_product_id(products, examples):
    corpus = cp.build_corpus(products, examples, "us")
    assert list(corpus["doc_id"]) == [0, 1, 2]
    assert list(corpus["product_id"]) == ["B1", "B2", "B3"]


def test_doc_id_mapping_is_stable_across_row_order(products, examples):
    first = cp.build_corpus(products, examples, "us")
    shuffled = products.sample(frac=1.0, random_state=0)
    second = cp.build_corpus(shuffled, examples, "us")
    assert first.equals(second)


def test_duplicate_product_ids_are_dropped(products, examples):
    corpus = cp.build_corpus(products, examples, "us")
    assert corpus["product_id"].is_unique


def test_text_is_built_for_every_product(products, examples):
    corpus = cp.build_corpus(products, examples, "us")
    text = corpus.loc[corpus["product_id"] == "B1", "product_text"].iloc[0]
    assert text == "First | Acme | Blue | a"


def test_missing_catalogue_entries_are_logged_not_fatal(products, examples, caplog):
    with caplog.at_level("WARNING"):
        cp.build_corpus(products, examples, "us")
    assert "missing from the catalogue" in caplog.text


def test_empty_scope_raises(products):
    empty = pd.DataFrame({"product_id": ["NOPE"]})
    with pytest.raises(cp.CorpusError, match="no products matched"):
        cp.build_corpus(products, empty, "us")


def test_round_trip_through_disk(products, examples, tmp_path):
    corpus = cp.build_corpus(products, examples, "us")
    cp.write_corpus(corpus, tmp_path)
    loaded = cp.read_corpus(tmp_path)
    assert loaded.equals(corpus)


def test_read_missing_corpus_raises(tmp_path):
    with pytest.raises(cp.CorpusError, match="Run scripts/build_corpus.py"):
        cp.read_corpus(tmp_path)
