"""Build the searchable text representation for each product.

Both BM25 and the bi-encoder search over one text field per product, so this
module decides what can be found at all. Fields are ordered by importance
(title, brand, colour, bullets) because the encoder truncates at a token limit:
if anything is cut off, it should be the least important part.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from esci.logging_utils import get_logger

logger = get_logger(__name__)

TEXT_FIELDS = ("product_title", "product_brand", "product_color", "product_bullet_point")
FIELD_SEPARATOR = " | "

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_WHITESPACE = re.compile(r"\s+")


def clean_field(value: object) -> str:
    """Normalise one raw field into plain single-spaced text.

    Product text in ESCI contains HTML tags, entities, bullet characters and
    inconsistent unicode. Left alone, these become tokens: BM25 would index
    'br' from '<br>' and the tokeniser would waste budget on markup.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if not text.strip():
        return ""

    # NFKC folds compatibility characters (full-width digits, ligatures) onto
    # their plain equivalents so the same word matches whichever form is used.
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    text = text.replace("\u2022", " ").replace("\xa0", " ")
    return _WHITESPACE.sub(" ", text).strip()


def build_product_text(row: pd.Series) -> str:
    """Join the cleaned fields of one product into a single searchable string."""
    parts = [clean_field(row.get(field)) for field in TEXT_FIELDS]
    return FIELD_SEPARATOR.join(part for part in parts if part)


def build_text_column(products: pd.DataFrame) -> pd.Series:
    """Vectorised version of build_product_text over a products frame.

    Applying clean_field per column is far faster than per row: 480k rows times
    four fields is 1.9M calls either way, but pandas string operations run in a
    single pass per column instead of one Python call per cell.
    """
    missing = [f for f in TEXT_FIELDS if f not in products.columns]
    if missing:
        raise ValueError(f"products frame is missing columns: {missing}")

    cleaned = {field: products[field].map(clean_field) for field in TEXT_FIELDS}
    frame = pd.DataFrame(cleaned, index=products.index)

    # Join non-empty parts only, so a product with no colour does not get a
    # dangling separator that would be tokenised as content.
    text = frame.apply(lambda r: FIELD_SEPARATOR.join(p for p in r if p), axis=1)
    return text.rename("product_text")


def text_diagnostics(text: pd.Series) -> dict[str, float]:
    """Summary statistics used in the EDA notes and logged during the build."""
    lengths = text.str.len()
    words = text.str.count(r"\S+")
    return {
        "n_products": int(len(text)),
        "n_empty": int((lengths == 0).sum()),
        "chars_mean": round(float(lengths.mean()), 1),
        "chars_p50": float(lengths.quantile(0.5)),
        "chars_p95": float(lengths.quantile(0.95)),
        "words_mean": round(float(words.mean()), 1),
        "words_p50": float(words.quantile(0.5)),
        "words_p95": float(words.quantile(0.95)),
    }
