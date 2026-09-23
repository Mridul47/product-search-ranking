# product-search-ranking

[![CI](https://github.com/Mridul47/product-search-ranking/actions/workflows/ci.yml/badge.svg)](https://github.com/Mridul47/product-search-ranking/actions/workflows/ci.yml)

Two-stage product search on the Amazon Shopping Queries (ESCI) dataset.

Stage 1 retrieves candidates with BM25, dense retrieval and a hybrid of both. Stage 2 re-ranks them with LightGBM (LambdaRank) and a cross-encoder. Each setup is compared on NDCG@10, Recall@100, MRR and latency.

**Status: work in progress.** Results will be added here once experiments have been run.

## Dataset

[Amazon Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data). The data is not included in this repo and has its own licence.

## Licence

Code is MIT licensed. See [LICENSE](LICENSE).
