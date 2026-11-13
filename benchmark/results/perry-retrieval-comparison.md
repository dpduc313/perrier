# Perry retrieval: Readers and Chunking Strategies compared (no chat model)

Share of required source pages among the top 6 hits, over all 23 benchmark questions. Ingest time counts reading, chunking and embedding the Documents (- if they were already stored that way). Reading cost is what the vision model charged to read pages.

| Reader | Chunking | Chunks | Ingest time | Reading cost | Hybrid search: page recall | Hybrid search: all pages found | Reranked: page recall | Reranked: all pages found |
|---|---|---|---|---|---|---|---|---|
| docling | heading | 23 | 75 s | $0.0000 | 89% | 18/23 | 88% | 18/23 |
| docling | fixed | 23 | 71 s | $0.0000 | 89% | 18/23 | 88% | 18/23 |
| pymupdf | heading | 23 | 34 s | $0.0000 | 87% | 18/23 | 88% | 18/23 |
| pymupdf | fixed | 23 | 38 s | $0.0000 | 89% | 18/23 | 88% | 18/23 |
