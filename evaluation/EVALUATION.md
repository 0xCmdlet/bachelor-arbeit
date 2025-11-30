# RAG Pipeline Evaluation

Evaluation Framework zum Vergleich von RAG-Pipeline-Varianten mit RAGAS-Metriken.

## Überblick

Dieses Evaluationssystem analysiert gespeicherte Pipeline-Testergebnisse, um verschiedene RAG-Konfigurationen (Extraktoren, Chunking-Strategien, Embeddings, Retrieval-Methoden) systematisch anhand standardisierter Qualitätsmetriken zu vergleichen.

**Hauptmerkmale:**
- Offline-Evaluation (kein erneutes Ausführen der Pipelines notwendig)
- RAGAS-Metriken: Context Precision, Faithfulness, Factual Correctness, Semantic Similarity
- Unterstützt 46 deutsche Testfragen zur C-Programmierung
- Ausgaben: JSON (detailliert), CSV (Vergleich), TXT (Zusammenfassung)

## Dateien

```
evaluation/
├── evaluate_pipeline_results.py    # Haupt-Evaluationsskript
├── ground_truth.json                # Referenzantworten für 46 Testfragen
├── requirements.txt                 # Python-Abhängigkeiten
├── .env                             # API-Keys (erstellen)
└── results/                         # Evaluationsergebnisse
    ├── ragas_evaluation_*.json      # Detaillierte Ergebnisse
    ├── ragas_evaluation_*.csv       # Vergleichstabelle
    └── ragas_evaluation_*.txt       # Zusammenfassungsbericht
```

## Voraussetzungen

### 1. Abhängigkeiten installieren

```bash
cd evaluation
pip install -r requirements.txt
```

**Benötigte Pakete:**
- `ragas` - Evaluation Framework
- `langchain-openai` - LLM Integration
- `langchain-huggingface` - Embeddings
- `pandas` - Datenverarbeitung
- `python-dotenv` - Umgebungsvariablen

### 2. API-Key einrichten

`.env`-Datei im `evaluation/`-Ordner erstellen:

```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

**Warum benötigt:** RAGAS verwendet GPT-4o-mini zur Evaluation von Metriken wie Context Precision und Faithfulness.

### 3. Pipeline-Testergebnisse

Sicherstellen, dass Pipeline-Testergebnisse in `../results/` vorhanden sind:
- `pipeline_test_results.json` (vollständige Ergebnisse, 6.1 MB)
- `small-sample.json` (Test-Teilmenge, 438 KB)

Diese werden von `../automated-pipeline-evaluation/run_full_evaluation.py` generiert.

## Verwendung

### Basis-Evaluation

Alle Pipelines mit Ground Truth evaluieren:

```bash
python evaluate_pipeline_results.py --ground-truth ground_truth.json
```

Dies wird:
1. `../results/pipeline_test_results.json` laden
2. Alle erfolgreichen Pipelines evaluieren
3. Ergebnisse in `results/ragas_evaluation_YYYYMMDD_HHMMSS.*` speichern

### Teilmenge evaluieren (Quick-Test)

Kleine Stichprobe für schnellere Tests verwenden:

```bash
python evaluate_pipeline_results.py \
  --results-file ../results/small-sample.json \
  --ground-truth ground_truth.json
```

### Top N Pipelines evaluieren

Nur die Top-5-Pipelines nach Retrieval-Score evaluieren:

```bash
python evaluate_pipeline_results.py \
  --ground-truth ground_truth.json \
  --top-n 5
```

### Spezifische Pipelines evaluieren

Nur bestimmte Pipeline-IDs evaluieren:

```bash
python evaluate_pipeline_results.py \
  --ground-truth ground_truth.json \
  --pipeline-ids "doc_lan_nomi_den,uns_sem_open_hyb,lla_cho_nomi_den_rnk"
```

### Benutzerdefiniertes Ausgabeverzeichnis

```bash
python evaluate_pipeline_results.py \
  --ground-truth ground_truth.json \
  --output-dir custom_results/
```

## Kommandozeilen-Optionen

| Option | Standard | Beschreibung |
|--------|---------|-------------|
| `--results-file` | `../results/pipeline_test_results.json` | Eingabe-Pipeline-Testergebnisse |
| `--ground-truth` | None | Ground Truth Referenzdatei (empfohlen) |
| `--pipeline-ids` | None | Kommagetrennte Pipeline-IDs zur Evaluation |
| `--top-n` | None | Nur Top N Pipelines nach Score evaluieren |
| `--output-dir` | `results` | Ausgabeverzeichnis für Evaluationsergebnisse |

## Evaluationsmetriken

### 1. Context Precision
- **Misst:** Relevanz der abgerufenen Kontexte zur Anfrage
- **Bereich:** 0.0 bis 1.0 (höher ist besser)
- **Benötigt:** Ground Truth Referenz
- **Interpretation:** Wie gut das Retrieval-System relevante Informationen findet

### 2. Faithfulness
- **Misst:** Ob die Antwort in den abgerufenen Kontexten begründet ist
- **Bereich:** 0.0 bis 1.0 (höher ist besser)
- **Erkennt:** Halluzinationen und unbelegte Behauptungen
- **Interpretation:** Wie treu das LLM dem Quellmaterial bleibt

### 3. Factual Correctness
- **Misst:** Genauigkeit der Antwort im Vergleich zur Ground Truth
- **Bereich:** 0.0 bis 1.0 (höher ist besser)
- **Benötigt:** Ground Truth Referenz
- **Interpretation:** Gesamte Antwortqualität und Korrektheit

### 4. Semantic Similarity
- **Misst:** Semantische Ähnlichkeit zwischen generierter und Referenzantwort
- **Bereich:** 0.0 bis 1.0 (höher ist besser)
- **Verwendet:** Embedding-basierte Ähnlichkeit
- **Interpretation:** Wie nah die Antwort an der erwarteten Antwort ist

### Composite Score
Durchschnitt aller vier Metriken, verwendet zum Ranking der Pipelines.

## Ausgabedateien

Nach der Evaluation werden drei Dateien in `results/` erstellt:

### 1. JSON-Datei (`ragas_evaluation_YYYYMMDD_HHMMSS.json`)

Detaillierte Ergebnisse einschließlich:
- Metadaten (Zeitstempel, Quelldatei, verwendete Metriken)
- Pro-Pipeline-Scores (gesamt + pro Anfrage)
- Vollständige Anfragedetails (Anfragetext, Antwort, Kontexte, individuelle Metrik-Scores)

**Verwendung für:** Tiefgehende Analyse, Debugging spezifischer Anfragen

### 2. CSV-Datei (`ragas_evaluation_YYYYMMDD_HHMMSS_comparison.csv`)

Vergleichstabelle mit Spalten:
- `pipeline_id`
- `extractor`, `chunking`, `embedding`, `retrieval`
- `avg_retrieval_score`, `avg_retrieval_time_ms`
- `ragas_faithfulness`, `ragas_context_precision`, `ragas_factual_correctness`, `ragas_semantic_similarity`
- `ragas_composite` (Durchschnitt der 4 Metriken)

**Verwendung für:** Excel/Pandas-Analyse, Sortieren, Filtern, Visualisierung

### 3. Zusammenfassungstext (`ragas_evaluation_YYYYMMDD_HHMMSS_summary.txt`)

Menschenlesbare Zusammenfassung:
- Metadaten (Zeitstempel, evaluierte Pipelines, verwendete Metriken)
- Top-10-Pipelines nach Composite Score gerankt
- Individuelle Metrik-Scores für jede Pipeline
- Retrieval-Performance (Score, Zeit)

**Verwendung für:** Schneller Überblick, Präsentationen, Berichte

## Kostenschätzung

**Pro Anfrage:**
- 4 Metriken × ~500 Tokens/Metrik = ~2.000 Tokens
- Mit `gpt-4o-mini`: ~$0.001 pro Anfrage

**Beispiele:**
- 3 Pipelines × 46 Anfragen = 138 Evaluationen → ~$0.14
- 10 Pipelines × 46 Anfragen = 460 Evaluationen → ~$0.46
- 72 Pipelines × 46 Anfragen = 3.312 Evaluationen → ~$3.30

## Beispiel-Workflows

### 1. Quick-Test (3 Pipelines)
```bash
# Kleine Stichprobe für schnelle Tests verwenden
python evaluate_pipeline_results.py \
  --results-file ../results/small-sample.json \
  --ground-truth ground_truth.json

# Kosten: ~$0.14, Zeit: ~5 Minuten
```

### 2. Top-Performer (Top 10)
```bash
# Nur beste Pipelines evaluieren
python evaluate_pipeline_results.py \
  --ground-truth ground_truth.json \
  --top-n 10

# Kosten: ~$0.46, Zeit: ~15 Minuten
```

### 3. Vollständige Evaluation (alle 72)
```bash
# Umfassende Evaluation
python evaluate_pipeline_results.py \
  --ground-truth ground_truth.json

# Kosten: ~$3.30, Zeit: ~45 Minuten
```

## Troubleshooting

### Fehler: "OPENAI_API_KEY not set"
**Lösung:** `.env`-Datei mit API-Key erstellen:
```bash
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### Fehler: "Results file not found"
**Lösung:** Pipeline-Ergebnisse zuerst generieren:
```bash
cd ../automated-pipeline-evaluation
python run_full_evaluation.py --results-file ../results/pipeline_test_results.json
```

### Fehler: "No successful pipeline results to evaluate"
**Lösung:** Prüfen, dass `pipeline_test_results.json` enthält:
- `"phase": "retrieval"`
- `"status": "success"`

### Evaluation ist langsam
**Tipps:**
- Mit `--top-n 3` für Tests beginnen
- `small-sample.json` statt vollständiger Ergebnisse verwenden
- RAGAS-Metriken benötigen API-Aufrufe (unvermeidbar)
- Während Nebenzeiten ausführen für bessere API-Antwortzeiten

### Import-Fehler (ragas, langchain, etc.)
**Lösung:** Sicherstellen, dass die Evaluation-Virtual-Environment verwendet wird:
```bash
cd evaluation
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Ground Truth Format

Die `ground_truth.json`-Datei enthält Referenzantworten für 46 Anfragen:

```json
[
  {
    "query_id": "q1",
    "reference": "C wurde 1974 von Dennis M. Ritchie entwickelt."
  },
  {
    "query_id": "q2",
    "reference": "1988: Normierung durch ANSI, 'ANSI-C'."
  }
]
```

**Felder:**
- `query_id`: Eindeutiger Identifikator (entspricht Pipeline-Testergebnissen)
- `reference`: Erwartete/korrekte Antwort auf Deutsch

## Pipeline-IDs

Pipeline-IDs folgen dem Format: `{extractor}_{chunking}_{embedding}_{retrieval}[_reranking]`

**Beispiele:**
- `doc_lan_nomi_den` = Docling + LangChain + Nomic + Dense
- `uns_sem_open_hyb` = Unstructured + Semantic + OpenAI + Hybrid
- `lla_cho_nomi_den_rnk` = LlamaParse + Chonkie + Nomic + Dense + Reranking

**Komponenten:**
- Extractor: `doc` (Docling), `uns` (Unstructured), `lla` (LlamaParse)
- Chunking: `lan` (LangChain), `cho` (Chonkie), `sem` (Semantic)
- Embedding: `nomi` (Nomic), `open` (OpenAI)
- Retrieval: `den` (Dense), `hyb` (Hybrid)
- Reranking: `rnk` (Cross-encoder Reranking, optional)

## Nächste Schritte

1. **Ergebnisse analysieren:** CSV-Datei in Excel/Pandas zum Sortieren und Filtern öffnen
2. **Beste Pipeline identifizieren:** Nach höchstem Composite Score suchen
3. **Schwache Performer debuggen:** Pro-Anfrage-JSON-Details für spezifische Fehler prüfen
4. **Iterieren:** Pipeline-Konfigurationen basierend auf Erkenntnissen anpassen
5. **Ergebnisse dokumentieren:** Zusammenfassungs-TXT für Berichte und Präsentationen verwenden

## Referenzen

- [RAGAS Dokumentation](https://docs.ragas.io/)
- [RAGAS Metrics Guide](https://docs.ragas.io/en/latest/concepts/metrics/)
- [LangChain Integration](https://python.langchain.com/)
- Pipeline Testing: `../automated-pipeline-evaluation/`
- Testergebnisse: `../results/`
