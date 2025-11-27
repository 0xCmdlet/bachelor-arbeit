# Pipeline Testing

Automated testing for all RAG pipeline combinations.

## Two Scripts

### 1. `smoke-test.py` - Quick Validation (Run First)

Tests 13 strategic pipelines to verify all dimensions work.

```bash
python smoke-test.py
```

- **Time:** ~2 hours
- **Tests:** 3 extractors, 4 chunking, 2 embeddings, 4 retrieval strategies
- **Output:** Pass/fail report
- **Purpose:** Catch issues before running full 24-hour test

---

### 2. `test_pipelines.py` - Full Evaluation (Run After Smoke Test)

Tests all 56 valid pipeline combinations with 46 queries each.

```bash
python test_pipelines.py --results-file ../results/pipeline_test_results.json
```

- **Time:** ~12-24 hours
- **Tests:** All 56 combinations
- **Output:** `pipeline_test_results.json` (for RAGAS evaluation)
- **Purpose:** Comprehensive performance measurement

---

## Usage

```bash
# Step 1: Always run smoke test first
python smoke-test.py

# Step 2: If smoke test passes, run full test
python test_pipelines.py

# Step 3: Analyze results
python ../evaluation/analyze_results.py ../results/pipeline_test_results.json
```

---

## Common Options

```bash
# Dry run (see what will be tested)
python test_pipelines.py --dry-run

# Resume from pipeline N
python smoke-test.py --start-from 7
python test_pipelines.py --start-from 30

# Test only N pipelines
python test_pipelines.py --limit 5
```

---

## What Gets Tested

**Pipeline Dimensions:**

- **Extractors:** Docling, Unstructured, LlamaParse
- **Chunking:** LangChain, Chonkie, Semantic, Native
- **Embeddings:** Nomic (local), OpenAI (API)
- **Retrieval:** Dense, Hybrid, Dense+Rerank, Hybrid+Rerank

**Results:**

- Ingestion metrics (time, chunks, errors)
- Retrieval metrics (scores, timing, answers, contexts)

---

## Helper Files

- **`env_updater.py`** - Updates .env with pipeline config
- **`log_monitor.py`** - Waits for Docker worker completion
- **`validators.py`** - Validates smoke test results
