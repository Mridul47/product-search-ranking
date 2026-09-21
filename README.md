# product-search-ranking
Built a two-stage product search system on the Amazon ESCI dataset. Stage one compares BM25, dense and hybrid retrieval. Stage two re-ranks candidates with LightGBM and a cross-encoder. Evaluated on NDCG and latency, and served through FastAPI in Docker.
