# Indexing Pipeline

## 1. Where do the BibTeX records come from?
The pipeline was built for the [ACL Anthology](https://aclanthology.org). Every venue or year on the site exposes a *.bib file that lists all of its papers.  Example:

bash
# downloads all ACL 2024 entries
curl -o acl-2024.bib https://aclanthology.org/anthology.bib.gz


Any BibTeX file that contains a url field pointing at a PDF will work.

## 2. Command-line interface
All heavy lifting is hidden behind two small scripts.

### python -m src.indexing.index_cli
Ingests a BibTeX file and builds the index.

| Argument | Default | Description |
|----------|---------|-------------|
| --bib-file PATH | **required** | Path to the .bib file to ingest. |
| --index-name NAME | papers | Elasticsearch index to create / populate. |
| --batch-size N | 100 | BibTeX entries processed per batch. |
| --concurrency N | 4 | Worker processes used for PDF parsing. |
| --max-entries N | 0 | Stop after indexing at most *N* records (0 = no limit). |
| --force-delete-index | ‑ | Drop an existing index with the same name before ingesting. |
| --es-hosts HOST … | http://localhost:9200 | One or more Elasticsearch hosts. |

Example:
bash
python -m src.indexing.index_cli \
  --bib-file acl-2024.bib \
  --index-name acl-papers \
  --batch-size 200 \
  --concurrency 8 \
  --force-delete-index
