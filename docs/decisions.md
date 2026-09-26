# Decisions

Each entry records a choice, the reason, what else was considered, and what it
costs. Changing any of these invalidates earlier results, so a change means
re-running the affected experiments.

## D1: Dataset scope, US locale and small version

Using `product_locale == "us"` and `small_version == 1` from the Shopping
Queries Dataset (upstream commit `7916cdf6ab75a462e77f20ab40428a10923998d5`,
recorded in `data/raw/esci_manifest.json`).

Reason: the small version is the Task 1 (query-product ranking) subset, which
matches this project. Restricting to US avoids mixing languages in one
retrieval index, which would need multilingual encoders and make results harder
to interpret.

Alternatives: all three locales, or the large version. Both increase compute
without changing the engineering story.

Cost: results are not comparable with published numbers that use all locales.

Scope: 20,888 train queries and 8,956 test queries.

## D2: Train, validation and test split

Test comes from the dataset's own `split` column. Validation is a random 10% of
the train queries, sampled with a fixed seed and grouped by `query_id`.

Reason: using the published test split keeps this comparable with other work on
the same dataset. Grouping by `query_id` means every judgement for a query stays
on one side, so a model cannot see part of a query's judgements during training
and then be scored on the rest.

Alternatives: stratifying validation by judgement count or label mix. At ~20k
queries a random split already matches the train distribution closely, and
stratification would add an arbitrary binning choice with no clear benefit.

Cost: none material. Split sizes vary slightly between seeds.

## D3: Relevance gains

E = 1.0, S = 0.1, C = 0.01, I = 0.0, as used in the KDD Cup 2022 ESCI task.

Reason: these gains keep NDCG comparable with published numbers, and they encode
the intended ordering. A substitute is worth something but much less than an
exact match, and a complement much less again.

Cost: the gaps are large, so NDCG@10 mostly comes down to whether exact matches
rank highly. Substitute ordering barely moves the metric.

## D4: LightGBM integer labels

`LGBMRanker` needs non-negative integer targets, so labels are I=0, C=1, S=2,
E=3 with `label_gain=[0.0, 0.01, 0.1, 1.0]`.

Reason: LightGBM looks up the gain by label index, so this makes the model
optimise exactly the gains in D3 rather than a different implied scale.

Cost: none. The config validates that the mapping stays consistent with D3.

## D5: Retrieval corpus

All US products that appear in the small-version examples, train and test
combined.

Reason: this gives a realistic pool of distractors, and it is small enough to
embed on free GPU quota and to hold in memory alongside FAISS.

Alternatives: the full US catalogue. That is closer to production and is still
possible as a later scale test, but it multiplies embedding time and memory for
a result that does not change the method comparison.

Cost: retrieval is easier than against the full catalogue, so absolute recall is
optimistic relative to a production system. Every method is measured against the
same corpus, so comparisons between them stay fair.

Including test products in the corpus is not leakage: the corpus is what gets
searched, not what the models train on. Models are trained on train queries only.

## D6: Recall definition

Recall@k counts exact (E) products only.

Reason: exact matches are what the search is meant to surface. Counting
substitutes as well would inflate recall and hide the difference between
retrieval methods.

Cost: the number is stricter than recall figures elsewhere that count all
judged products. Stated alongside every reported value.

## D7: Queries with no exact product

These are excluded from Recall and MRR but kept for NDCG. The count is reported
with every result.

Reason: recall and reciprocal rank are undefined when there is nothing to find.
Scoring them as zero would mix "the system failed" with "there was nothing to
retrieve". NDCG still means something, because substitutes and complements carry
gain.

Cost: Recall and MRR are computed over a subset of the test queries. The subset
is identical for every method being compared.

## D8: Unjudged products in end-to-end evaluation

Retrieved products with no judgement get gain 0.

Reason: ESCI judges only a small number of products per query, so retrieval
against the full corpus returns many products that were never assessed. Some of
them are relevant.

Cost: end-to-end scores are a lower bound on true relevance. They are reported
separately from judged-set evaluation, which re-ranks only the judged products
and compares directly with published ESCI results. Both are always reported.

## D9: Hard negatives in re-ranker training

Stage 1 candidates with no judgement are treated as label 0 during training.

Reason: hard negatives are what make a re-ranker useful. Without them the model
only ever sees products that were judged relevant enough to assess.

Cost: some of those negatives are actually relevant, so the training signal
contains false negatives. This is standard practice in learning to rank, and it
is recorded in the README limitations.

## D10: Text truncation limit

Product text is truncated at 256 tokens for the bi-encoder. BM25 indexes the
full text.

Measured on a 20,000 product sample with the MiniLM tokenizer: median 142
tokens, p95 401, p99 511. 23.4% of products exceed 256 tokens and 1.0% exceed 512.

Reason: fields are ordered title, brand, colour, bullets, so truncation removes
bullet points rather than identifying information. all-MiniLM-L6-v2 was trained
at 256 tokens, and encoding cost grows faster than linearly with length, which
matters when the corpus is embedded on free GPU quota.

Cost: for the 23.4% of products with longer text, part of the bullet content is
invisible to dense retrieval but BM25 can still find it. This asymmetry is a
candidate explanation if dense retrieval underperforms, so Phase 5 compares 256
against 384 on a subset and records recall and encoding time for each.
