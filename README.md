# Production-Grade RAG over SEC 10-K Filings

A question-answering system over financial documents that **cites every claim to a filing and page, refuses when the documents don't support an answer, and is evaluated automatically on every pull request.**

Built over 15 fiscal-2026 Form 10-K filings (Cisco, FedEx, Microsoft, Oracle, Walmart and others). Ask a question in plain English; get an answer where every sentence links to the exact passage it came from, highlighted on the original PDF page.

## Results

Evaluated on a hand-verified golden dataset of 105 questions, with [RAGAS](https://docs.ragas.io/) as the faithfulness judge:

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | **0.976** | Share of the claims in an answer that are supported by the retrieved passages |
| Answer rate | **0.99** | Answerable questions that got an answer (99 of 100) |
| Refusal accuracy | **1.00** | Unanswerable questions correctly refused (5 of 5) |

Every pull request runs a 21-question subset and **fails the build** if faithfulness drops below 0.90, the answer rate below 0.90, or any unanswerable question is answered.

## How it works

```
                      question
                         │
         ┌───────────────┴───────────────┐
         │  detect companies named in it │   "Cisco and FedEx" -> search each company separately
         └───────────────┬───────────────┘
                         │
     ┌───────────────────┼───────────────────┐
     │  vector search (Chroma, embeddings)   │
     │  + BM25 keyword search                │   hybrid retrieval, merged 50/50
     └───────────────────┬───────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  cross-encoder reranker       │   scores each (question, chunk) pair together
         │  (Cohere Rerank or local)     │   -> top 10 chunks
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  Gate 1: evidence             │   refuse if no chunk is relevant enough
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  LLM answer with [n] citations│   versioned prompt from config/prompts.yaml
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │  Gate 2: citation checks      │   drop uncited claims and claims citing another
         │                               │   company's passage; refuse if most claims fail
         └───────────────┬───────────────┘
                         │
              answer + sources (file, company, pages)
```

### Ingestion

1. **Parse** each PDF to markdown with [Docling](https://github.com/docling-project/docling) (layout-aware, keeps tables).
2. **Extract the company name** from each filing's cover page (regex first, LLM fallback), so nothing is hard-coded per company.
3. **Chunk** into 800–1,000 tokens with 100 tokens of overlap (LangChain splitter, counted with `tiktoken`). 1,397 chunks in total.
4. **Map every chunk to its PDF pages** by phrase matching against the original PDF, so citations can name a page.
5. **Embed** with `nomic-ai/modernbert-embed-base` (reads the full 1,000-token chunk) and store in ChromaDB.

### Retrieval and answering

- **Company-aware search:** companies named in the question are detected, and each is searched only within its own filings, with the question rewritten to name only that company. Other companies' text never reaches the LLM.
- **Hybrid retrieval:** vector search catches meaning ("supplier shortages" ≈ "single-source vendors"); BM25 catches exact terms (tickers, "10-K", figures).
- **Reranking:** a cross-encoder reads the question and each chunk together and re-scores them. Cohere Rerank v3.5 by default, or a local `BAAI/bge-reranker-base` model.
- **Citation enforcement:** the LLM cites short excerpt numbers (`[2]`); the code, not the LLM, turns them into file, company and page labels, and removes any claim whose citation fails the checks.

## Project structure

```
app.py                        answer_question(): the full pipeline; also a command-line entry point
ui.py                         Streamlit web UI with highlighted PDF citations
config/
  prompts.yaml                versioned prompts (answer_v1, ...)
  settings.yaml               LLM, retrieval, thresholds, evaluation and CI settings
src/
  config.py                   loads the two YAML files
  highlight.py                finds the cited sentence and highlights it on the PDF page
  retrieval/hybrid_search.py  vector + BM25 + reranker
  ingestion/                  s3_sync, batch_parse, extract_companies, chunker,
                              map_pages, build_vectorstore, upload_ci_data
evaluation/
  generate_candidates.py      drafts Q&A candidates from sampled chunks
  build_golden.py             builds golden_dataset.jsonl from the rows verified by hand
  golden_candidates.csv       108 reviewed candidates with notes
  golden_dataset.jsonl        105 verified questions (95 single-company, 5 multi-company, 5 unanswerable)
  evaluate.py                 runs the questions, scores them with RAGAS, applies the CI thresholds
.github/workflows/evaluate.yml  CI: evaluation on every pull request
```

## Setup

Requires Python 3.12 (the pipeline also runs on 3.14).

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
COHERE_API_KEY=...          # LLM and reranker
RERANKER=cohere             # or sbert for the local cross-encoder
AWS_ACCESS_KEY_ID=...       # only for downloading PDFs / uploading CI data
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
S3_BUCKET_NAME=...
```

To run the LLM locally instead of through Cohere, set `llm.provider: ollama` in `config/settings.yaml` and run `ollama pull llama3.1`. It is much slower on a CPU.

## Usage

### Build the data (once)

```bash
python src/ingestion/s3_sync.py             # download the PDFs
python src/ingestion/batch_parse.py         # PDF -> markdown (slowest step on a CPU)
python src/ingestion/extract_companies.py   # company names
python src/ingestion/chunker.py             # 800-1,000-token chunks
python src/ingestion/map_pages.py           # chunk -> PDF pages
python src/ingestion/build_vectorstore.py   # embeddings -> ChromaDB
python src/ingestion/upload_ci_data.py      # package the data for CI (S3)
```

### Ask questions

```bash
streamlit run ui.py                                           # web UI at http://localhost:8501
python app.py "What was Cisco's total revenue for fiscal 2026?"   # command line
```

In the UI, each source opens to the original PDF page with the supporting sentence or table row highlighted.

## Evaluation

The golden dataset was drafted by an LLM from sampled chunks, then every pair was checked against the filing and kept only if correct. It includes multi-company questions and questions the filings cannot answer, which test the refusal behaviour.

The evaluation runs in two stages. Answers are generated with the main environment; scoring runs in a separate environment because RAGAS pins package versions that conflict with the pipeline:

```bash
python evaluation/evaluate.py --stage answers

python -m venv .venv-eval
.venv-eval/bin/pip install -r requirements-eval.txt      # Windows: .venv-eval\Scripts\pip
.venv-eval/bin/python evaluation/evaluate.py --stage score
```

Saved answers and unchanged scores are reused, so re-runs only pay for what changed. Add `--ci --check` to run the CI subset and exit with code 1 if a threshold is missed.

## CI

`.github/workflows/evaluate.yml` runs on every pull request:

1. installs the pinned packages (CPU-only PyTorch);
2. downloads the vector database and chunks from S3;
3. answers the 21 CI questions and scores them with RAGAS;
4. fails the build if a threshold in `config/settings.yaml` is not met.

With branch protection on `main`, a pull request that lowers answer quality cannot be merged. Required repository secrets: `COHERE_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `S3_BUCKET_NAME`.

## Known limitations

- **Faithfulness is not correctness.** It checks that answers are supported by the retrieved passages, not that they match the golden answer.
- **The same model generates and judges the answers** (Cohere `command-a`), which makes the faithfulness score somewhat optimistic.
- **Scores vary slightly between runs** (about ±0.015 on the CI subset), even at temperature 0.
- **Page numbers are PDF page numbers**, which differ from the page numbers printed in the filings.
- **The BM25 index is rebuilt at every startup**, which is fine for 15 filings but would need a persistent keyword index at scale.
- One answerable question in the golden set (a debt maturity date stored in a table) is currently refused.
