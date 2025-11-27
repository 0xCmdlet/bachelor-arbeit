# RAG Pipeline Evaluation System

**Bachelorarbeit**: Systematische Evaluation von Retrieval-Augmented Generation (RAG) Pipelines für deutschsprachige technische Dokumentation

## �berblick

Dieses Repository implementiert ein vollständiges RAG-System zur Beantwortung von Fragen aus technischen Dokumenten. Das System verarbeitet PDF-Dokumente, erstellt Vektorembeddings und beantwortet Fragen mithilfe eines lokalen LLMs (Llama 3.1). Der Fokus liegt auf der systematischen Evaluation verschiedener Pipeline-Konfigurationen.

## Architektur

Das System besteht aus mehreren Komponenten:

### 1. **Ingestion Worker** (`ingestion-worker/`)

Verarbeitet Dokumente mit GPU-Beschleunigung:

- **Textextraktion**: Docling, Unstructured.io, LlamaParse (mit OCR, Layout-Erkennung, Tabellen)
- **Chunking**: LangChain, Chonkie, Semantic Chunking
- **Embeddings**: SentenceTransformers (Nomic), OpenAI
- **Speicherung**: Qdrant (Vektoren), PostgreSQL (Metadaten), MinIO (Dokumente)

### 2. **RAG API** (`rag-api/`)

FastAPI-Backend f�r Anfragen:

- **Retrieval**: Dense, Hybrid (Dense + Sparse mit RRF), optional Reranking
- **Generation**: Llama 3.1 (8B Instruct) via Ollama
- **Endpoints**: `/query` (einmalig), `/chat` (konversational mit Historie)
- **Features**: Cross-Encoder Reranking, konversationelle Agenten mit LangGraph

### 3. **Frontend** (`frontend/`)

React-basierte Chat-Oberfl�che:

- Konversationelle Schnittstelle mit Nachrichtenverlauf
- Quellenangaben mit Relevanz-Scores
- Konversationsverwaltung (Liste, L�schen, Fortsetzen)

### 4. **Database** (`Database/`)

PostgreSQL-Schema:

- `files`: Dokumentmetadaten (MIME-Type, Gr��e, Extraktionsstatus)
- `chunks`: Textchunks mit Markdown-Headern, Token-Counts, Chunk-Typen

### 5. **Automated Pipeline Evaluation** (`automated-pipeline-evaluation/`)

Automatisiertes Testing aller Pipeline-Kombinationen:

- **56 Pipeline-Varianten** �ber 4 Dimensionen (Extraktor, Chunking, Embeddings, Retrieval)
- **46 deutsche Testfragen** zu C-Programmierung
- Siehe [Pipeline Testing Usage](automated-pipeline-evaluation/usage.md) für Details

### 6. **Evaluation** (`evaluation/`)

RAGAS-basierte Qualit�tsmetriken:

- Context Precision, Faithfulness, Factual Correctness, Semantic Similarity
- Vergleich von Pipeline-Varianten mit Ground Truth
- Ausgabe: JSON, CSV, TXT

## Schnellstart

### Voraussetzungen

- Docker & Docker Compose
- NVIDIA GPU mit 8-24 GB VRAM (für Worker & Ollama)
- API-Keys (optional): OpenAI, Unstructured.io, LlamaParse

### 1. System starten

```bash
# .env konfigurieren (siehe .env f�r Optionen)
cp .env .env.local
# Editiere .env.local nach Bedarf

# Alle Services starten
docker-compose up -d

# Logs �berwachen
docker-compose logs -f worker rag-api
```

### 2. Dokumente hochladen

```bash
# Via MinIO Web UI (http://localhost:9001)
# Login: minioadmin / minioadmin123
# Bucket: study
```

Der Worker verarbeitet automatisch neue Dokumente im `study` Bucket.

### 3. System nutzen

- **Frontend**: http://localhost
- **API Docs**: http://localhost:8080/docs
- **MinIO Console**: http://localhost:9001

### 4. Pipeline Testing (Optional)

```bash
cd automated-pipeline-evaluation

# Quick-Test (13 Pipelines, ~2h)
python validate_pipeline_dimensions.py

# Vollst�ndige Evaluation (56 Pipelines, ~12-24h)
python run_full_evaluation.py --results-file ../results/pipeline_test_results.json
```

Siehe [Pipeline Testing Usage](automated-pipeline-evaluation/usage.md) für Details.

### 5. Evaluation mit RAGAS

```bash
cd evaluation

# .env mit OpenAI API Key erstellen
echo "OPENAI_API_KEY=sk-..." > .env

# Pipelines evaluieren
python evaluate_with_ragas.py --ground-truth ground_truth.json

# Ergebnisse analysieren
python analyze_pipeline_statistics.py ../results/pipeline_test_results.json
```

## Konfiguration

### Pipeline-Dimensionen

Die `.env` Datei steuert die Pipeline-Konfiguration:

```bash
# Extraktion
EXTRACTOR_TYPE=docling              # docling, unstructured, llamaparse

# Chunking
CHUNKING_STRATEGY=langchain         # langchain, chonkie, semantic
CHUNK_TOKENS=750
CHUNK_OVERLAP=150

# Embeddings
EMBEDDING_PROVIDER=sentence-transformers  # sentence-transformers, openai
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Retrieval
RETRIEVAL_STRATEGY=hybrid           # dense, hybrid
RERANKING_STRATEGY=cross-encoder    # none, cross-encoder
```

### Qdrant Collections

Collection-Namen folgen der Konvention:

```
docs_{extractor}_{chunking}_{embedding}_{retrieval}[_rerank]
```

Beispiel: `docs_docling_langchain_nomic_hybrid_rerank`

## Services

| Service    | Port       | Beschreibung            |
| ---------- | ---------- | ----------------------- |
| Frontend   | 80         | React UI                |
| RAG API    | 8080       | FastAPI Backend         |
| Ollama     | 11434      | LLM Inference           |
| Qdrant     | 6333       | Vector Database         |
| MinIO      | 9000, 9001 | Object Storage + Web UI |
| PostgreSQL | 5432       | Metadata Database       |

## GPU-Anforderungen

- **8 GB VRAM**: Docling, Embeddings, Inference (sequenziell)
- **24 GB VRAM**: Alle Modelle gleichzeitig + Reranking auf GPU

Konfiguration via `.env`:

```bash
GPU_CACHE_CLEAR_BETWEEN_PHASES=true  # F�r 8GB VRAM
RERANKER_DEVICE=cuda                  # Nur f�r 24GB VRAM
```

## Projekt-Struktur

```
.
ingestion-worker/      # Dokumentenverarbeitung
rag-api/               # Query & Chat API
frontend/              # React UI
Database/              # PostgreSQL Schema
automated-pipeline-evaluation/  # Pipeline Testing
evaluation/            # RAGAS Evaluation
config/                # Konfigurationsdateien
results/               # Test-Ergebnisse
docker-compose.yaml    # Service-Orchestrierung
.env                   # Umgebungsvariablen
```

## Forschungsfokus

Diese Bachelorarbeit untersucht den Einfluss verschiedener Pipeline-Komponenten auf die Qualit�t der RAG-Ausgabe:

- **Extraktoren**: OCR-Genauigkeit, Layout-Erkennung, Tabellenextraktion
- **Chunking**: Semantische Koh�renz vs. feste Token-Grenzen
- **Embeddings**: Lokale Modelle vs. API-basiert
- **Retrieval**: Dense vs. Hybrid Search, Reranking-Effekte

Systematische Evaluation mit RAGAS-Metriken auf 46 deutschen Testfragen zur C-Programmierung.

## Lizenz

Bachelorarbeit-Projekt - Alle Rechte vorbehalten.
