# MigraineGuidelineQA

A local-first medical guideline QA system that turns migraine PDFs into a structured, reviewable, cross-document knowledge base.

## Overview

MigraineGuidelineQA is designed for real-world guideline workflows where the user uploads one or many migraine-related PDF documents, parses them with MinerU, extracts structured clinical information, and routes questions to the correct retrieval path.

The system combines:

- PDF parsing and structure-preserving extraction
- Elasticsearch for exact medication and dose lookup
- Chroma for semantic retrieval
- GraphRAG for cross-guideline reasoning
- Tavily for latest public information
- Local Ollama models for privacy-friendly local inference

This project is intentionally split into two environments:

- Main app: `${PROJECT_ROOT}`
- GraphRAG runtime: `${GRAPH_RAG_ROOT}`

The split keeps GraphRAG isolated from the main app to avoid dependency conflicts while still enabling tight integration through controlled subprocess execution.

## Why this project exists

Medical guidance documents are often distributed as PDFs and tables, which are hard to query directly. In many clinical or research scenarios, users need to answer questions such as:

- What is the recommended dose of drug X?
- Which guideline recommends this treatment for acute migraine?
- Are there differences across multiple migraine guidelines for this therapy?
- Is there newer evidence or guidance beyond the uploaded documents?

This system addresses those needs with a complete knowledge pipeline:

- PDF ingestion and parsing
- Table extraction and normalization
- Searchable drug/dose index in Elasticsearch
- Human-in-the-loop review before final indexing
- Vector retrieval for general semantic questions
- Cross-guide reasoning through GraphRAG
- Optional latest information via Tavily web search

## System architecture

```text
User PDF Upload
   ↓
Streamlit Frontend
   ↓
Pipeline Orchestrator
   ├── MinerU PDF parsing → JSON + MD
   │   ├── JSON → ES table extraction
   │   │   └── approval/review workflow
   │   └── MD → Chroma vector retrieval
   │
   ├── GraphRAG integration (separate environment)
   │   ├── input/ historical MD set
   │   ├── output/ graph state
   │   └── standard / standard-update strategy
   │
   └── Query routing
       ├── ES precise retrieval
       ├── hybrid semantic retrieval
       ├── GraphRAG cross-guide reasoning
       └── Tavily for latest external knowledge
```

## Core capabilities

### 1. PDF ingestion and parsing

- Upload one or multiple PDF guideline files
- Parse with MinerU online API
- Generate both JSON and Markdown outputs
- Preserve real guideline names while storing historical results

### 2. Structured extraction for clinical content

- Extract treatment metadata and dose tables from parsed PDF pages
- Convert table rows into structured documents
- Filter and normalize malformed or inconsistent output
- Index the validated records into Elasticsearch for precise search

### 3. Human review and approval flow

- Created review records are shown in a dedicated review center
- Users can inspect extraction quality before final acceptance
- Approved or modified entries are saved into the approved knowledge pipeline
- This is critical in high-trust domains such as medical guidance

### 4. Local semantic retrieval

- Store the Markdown representation in a vector database
- Support general semantic QA for the uploaded guideline corpus
- Use local Ollama-hosted embeddings and models, keeping the system offline-friendly

### 5. GraphRAG for cross-guideline reasoning

- Build a knowledge graph across multiple uploaded guideline documents
- Support cross-document comparison and relationship reasoning
- Use incremental indexing whenever valid prior graph state exists
- Fall back to full rebuild when graph state is missing or invalid

### 6. Question intent routing

The system routes queries based on question type:

- Dose-specific or exact fact questions → Elasticsearch
- General guideline questions → local vector retrieval
- Multi-guideline comparison or relation reasoning → GraphRAG
- News or latest clinical updates → Tavily web search


## Technologies

### Frontend and orchestration
- Python
- Streamlit
- Project-level orchestration for upload and indexing flow

### AI and retrieval
- Ollama local models
- Qwen 2.5 family
- Local embeddings
- Bge-reranker-base
- Elasticsearch
- Chroma
- GraphRAG
- Tavily

### Data pipeline
- MinerU for PDF parsing
- JSON for tabular extraction
- Markdown for document retrieval and graph input
- Local storage for uploaded PDFs, processed outputs, and review records

## Runtime requirements

### Main project environment

- Python 3.11+
- Conda environment: `rag_env`
- Dependencies: `requirements.txt`
- Local services:
  - Elasticsearch: `${ES_HOST} (default: http://localhost:9200)`
  - Ollama: `${OLLAMA_BASE_URL} (default: http://localhost:11434)`

### GraphRAG environment

- Python 3.11
- Conda environment: `graphrag_env`
- GraphRAG Repository：<a href="https://github.com/charchar111720/Medical_migraine_graphrag" target="_blank">Medical_migraine_graphrag</a>
- GraphRAG root: `${GRAPH_RAG_ROOT}`
- Local models used through Ollama:
  - `qwen2.5:7b`
  - or `qwen2.5:14b-instruct-q4_K_M`
  - embedding model such as `bge-m3`

(Note: After deploying the GraphRAG repository, you must configure the input source to Markdown format in its settings.yaml, specifically by setting file_type: text and file_pattern: ".\.md$$", so that it can directly consume the MD files passed in from the main application.)*

## Quick start

### 1. Install dependencies

```bash
cd ${PROJECT_ROOT}
pip install -r requirements.txt
```

### 2. Pull local models

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull bge-m3
```

### 3. Start Elasticsearch

Make sure Elasticsearch is running locally at `http://localhost:9200`.


### 4. Start GraphRAG

Ensure GraphRAG indexing works properly.
```bash
graphrag index --root ./
```

### 5. Start the app

```bash
cd ${PROJECT_ROOT}
streamlit run app.py
```

### 6. Upload PDFs

The workflow will automatically:

- parse the PDF through MinerU
- generate JSON + Markdown outputs
- extract dose and treatment tables
- show review records for approval
- index into Elasticsearch
- build semantic retrieval data
- trigger the GraphRAG graph build in the separate environment

## Project structure

- [app.py](app.py) – frontend and user workflow
- [pipeline.py](pipeline.py) – orchestration pipeline
- [src/ingestion/](src/ingestion/) – MinerU, extraction, review, and GraphRAG preparation
- [src/retrieval/](src/retrieval/) – routing and retrieval logic
- [src/utils/](src/utils/) – metadata, environment, and path utilities
- [docs/](docs/) – deployment, demo, and project handoff materials

## Why this matters

This project is not just a chatbot layered on top of PDFs. It is a clinical knowledge system that emphasizes:

- accurate extraction of treatment and dose facts
- trustable review before indexing
- multi-document accumulation instead of blind overwrite
- local-first deployment with privacy awareness
- graph reasoning for cross-guide comparison

That makes it substantially more appropriate for medical guideline QA than a naive RAG-only implementation.

## License

This project is intended for research, internal experimentation, and demonstration purposes. If used in clinical, commercial, or production settings, please verify the licensing and compliance requirements of the underlying tools, including MinerU, GraphRAG, Ollama, Elasticsearch, and Tavily.

## Appendix: medical GraphRAG customization and public-safe deployment notes

This project does not use the default GraphRAG pipeline as a generic off-the-shelf graph builder. Instead, it treats GraphRAG as a medical-domain knowledge-graph engine and modifies it to fit the requirements of guideline QA.

Key domain-specific changes:

- Medical entity schema: Disease, Symptom, Drug, Diagnostic Criteria, Treatment, Evidence Level, Contraindication
- Controlled relation schema based on diagnosis, therapy, contraindication, and evidence logic
- Manual Prompt engineering across indexing and query stages
- Explicit restrictions to prevent the LLM from inventing unsupported clinical entities
- PDF-first processing with MinerU and domain-specific chunking for guideline text
- Multi-document graph accumulation instead of blind overwriting

This is important because the default Microsoft GraphRAG design is general-purpose and uses generic labels such as PERSON and ORGANIZATION, which are not well aligned with clinical evidence tasks. For guideline-based QA, a medical graph must be semantically precise and clinically grounded.

## Public deployment note

Do not hardcode a personal root path in public-facing documentation. The project should use environment variables or a local configuration file instead of exposing a machine-specific directory.

Recommended pattern:

- Main app root: `${PROJECT_ROOT}` or a local path configured in `.env`
- GraphRAG root: `${GRAPH_RAG_ROOT}`
- Model endpoints: `${OLLAMA_BASE_URL}`
- MinerU endpoint and token: environment variables, not hardcoded absolute paths

Example configuration pattern:

```bash
export PROJECT_ROOT="${HOME}/projects/migraine-guideline-qa"
export GRAPH_RAG_ROOT="${HOME}/projects/graphrag-main"
export OLLAMA_BASE_URL="http://localhost:11434"
export ES_HOST="http://localhost:9200"
```

This keeps project documentation portable across machines and avoids leaking a user’s local filesystem structure.
