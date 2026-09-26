import pandas as pd
import pytest

from esci.data import text as tx


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain title", "plain title"),
        ("  padded  ", "padded"),
        ("line\nbreak\ttab", "line break tab"),
        ("<b>bold</b> text", "bold text"),
        ("caf&eacute; &amp; bar", "caf bar"),
        ("\u2022 bullet point", "bullet point"),
        ("ﬁle", "file"),  # NFKC folds the ligature
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_clean_field(raw, expected):
    assert tx.clean_field(raw) == expected


def test_build_text_orders_fields_and_skips_blanks():
    row = pd.Series(
        {
            "product_title": "Running Shoes",
            "product_brand": "Acme",
            "product_color": "",
            "product_bullet_point": "Lightweight",
        }
    )
    assert tx.build_product_text(row) == "Running Shoes | Acme | Lightweight"


def test_build_text_column_matches_row_version():
    products = pd.DataFrame(
        {
            "product_title": ["Shoes", "<i>Mug</i>", ""],
            "product_brand": ["Acme", None, ""],
            "product_color": ["Black", "White", ""],
            "product_bullet_point": ["Light", "", ""],
        }
    )
    column = tx.build_text_column(products)
    expected = products.apply(tx.build_product_text, axis=1)
    assert list(column) == list(expected)
    assert column.iloc[2] == ""


def test_build_text_column_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        tx.build_text_column(pd.DataFrame({"product_title": ["x"]}))


def test_diagnostics_counts_empty_text():
    stats = tx.text_diagnostics(pd.Series(["abc def", "", "one two three"]))
    assert stats["n_products"] == 3
    assert stats["n_empty"] == 1
    assert stats["words_mean"] == pytest.approx(1.7)
