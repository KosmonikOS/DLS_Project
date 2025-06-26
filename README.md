# NLP Anthology Search – Quick Start

This repository lets you build a **hybrid BM25 + dense-vector** search stack for large
BibTeX collections such as the ACL Anthology.

The workflow consists of three steps – *deploy*, *index*, *search* – each backed
by a small CLI or a Docker Compose service.

---

## 1  Deploy dependencies (Elasticsearch + Kibana)

Prerequisites
* Docker ≥ 20.10
* Docker-Compose plugin (`docker compose`)

```bash
# from the project root
cd deployment/

# start a single-node ES 9.0 + Kibana stack
docker compose up -d

# check that Elasticsearch responds
curl http://localhost:9200 | jq .cluster_name
```

The compose file disables X-Pack security and exposes

* Elasticsearch → `http://localhost:9200`
* Kibana → `http://localhost:5601` (optional GUI)

> Data are persisted in the `esdata` named volume – `docker compose down -v`
> will **wipe** the index.

---

## 2  Index the collection (pipeline in `src/indexing/`)

Create – or download – a BibTeX file (`.bib`) that contains a `url` field for
each entry pointing at a PDF. Example (ACL 2024):

```bash
curl -L https://aclanthology.org/anthology.bib.gz | gunzip > acl-2024.bib
```

Populate the required environment variables (or a `.env` file):

```
# .env
BIB_FILE=acl-2024.bib      # path to the BibTeX file
INDEX_NAME=acl-papers      # name of the ES index to create
ES_HOST=http://localhost:9200
# optional tuning parameters ↓
BATCH_SIZE=200
ACL_CONCURRENCY=4
CROSSREF_CONCURRENCY=40
PAGERANK_ALPHA=0.85           # damping factor for PageRank (optional)
FORCE_DELETE_INDEX=true
```

Run the pipeline:

```bash
python -m src.indexing.indexing_pipeline
```

Internally the pipeline will
1. Download every PDF in parallel (using Docling)
2. Extract Markdown, embed full text via `DenseEmbedder`
3. Create an index with a `dense_vector` field sized to the embedding dimension
4. Bulk-index all documents with metadata and `text_embedding`

Progress is shown via a TQDM bar. Once finished, the papers are searchable at
`http://localhost:9200/<INDEX_NAME>`.

---

## 3  Interactive search (CLI in `src/search/`)

Add search-specific knobs to your `.env` if you need to override defaults:

```
TOP_K=10        # how many results to display
KNN_K=50        # neighbours retrieved from vector search
KNN_CANDIDATES=100
```

Launch the CLI:

```bash
python -m src.search.search_cli
```

Features
* BM25 *and* k-NN vector search are executed per query.
* Results are fused locally via Reciprocal Rank Fusion (no commercial ES license required).
* Output shows rank, title, year, authors, and URL.

Example session
```
query> multilingual summarization
1. XL-Sum: Large-Scale Multilingual Abstractive Summarization (2024)
   ...
2. A Unified Model for ...
...
```

Stop with `Ctrl-C`, `Ctrl-D`, or by typing `exit`/`quit`.

Happy searching! 🎉
