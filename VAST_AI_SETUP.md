# Vast.ai Setup Anleitung

Diese Anleitung erklärt, wie das RAG Pipeline Evaluation System auf einer Vast.ai GPU-Instanz bereitgestellt und ausgeführt wird.

## Voraussetzungen

- Vast.ai Account mit ausreichend Credits
- GPU-Instanz mit NVIDIA GPU (empfohlen: 8-24 GB VRAM)
- Grundlegende Linux- und Docker-Kenntnisse

## Überblick

Vast.ai bietet kostengünstige GPU-Instanzen für Machine Learning Workloads. Diese Anleitung deckt den kompletten Setup-Prozess von der Instanz-Erstellung bis zum Ausführen der RAG-Pipeline ab.

## 1. Instanz-Auswahl

Bei der Auswahl einer Vast.ai Instanz sicherstellen:

- **GPU**: NVIDIA GPU mit 8+ GB VRAM (RTX 3060, RTX 3090 oder besser)
- **RAM**: Minimum 16 GB System-RAM
- **Disk**: Mindestens 50 GB verfügbarer Speicher
- **Docker**: Vorinstalliert (bei den meisten Instanzen enthalten)

## 2. Initiales Setup

### 2.1 NVIDIA Container Toolkit konfigurieren

Das NVIDIA Container Toolkit wird benötigt, damit Docker-Container auf die GPU zugreifen können. Ohne dieses schlagen Ollama und andere GPU-abhängige Services fehl.

```bash
# Version setzen
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1

# Toolkit und Abhängigkeiten installieren
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

Docker für die Verwendung der NVIDIA Runtime konfigurieren:

```bash
nvidia-ctk runtime configure --runtime=docker
```

Docker neu starten, um Änderungen zu übernehmen:

```bash
# Diese Befehle der Reihe nach ausprobieren, bis einer funktioniert
systemctl restart docker
# oder
service docker restart
# oder, falls keiner existiert (häufig in minimalen Images):
dockerd &
```

GPU-Zugriff verifizieren:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## 3. Projekt starten

Umgebungsvariablen konfigurieren:

```bash
cp .env .env.local
# .env.local mit den gewünschten Einstellungen bearbeiten
nano .env.local
```

Alle Services starten:

```bash
docker-compose up -d
```

Logs überwachen:

```bash
docker-compose logs -f worker rag-api
```

## 4. MinIO Konfiguration

### 4.1 MinIO Client installieren

MinIO Client herunterladen und installieren:

```bash
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

### 4.2 MinIO Alias konfigurieren

MinIO Alias einrichten, um zur lokalen Instanz zu verbinden:

```bash
mc alias set myminio http://localhost:9000 minioadmin minioadmin123
```

Verbindung verifizieren:

```bash
mc admin info myminio
```

### 4.3 Storage Bucket erstellen

`study` Bucket für Dokumentenspeicherung erstellen:

```bash
mc mb myminio/study
```

Bucket-Erstellung verifizieren:

```bash
mc ls myminio
```

## 5. Dokumente hochladen

PDF-Dokumente in den MinIO Bucket hochladen:

```bash
# Einzelne Datei hochladen
mc cp ~/AP1.pdf myminio/study/

# Mehrere Dateien hochladen
mc cp ~/documents/*.pdf myminio/study/

# Verzeichnis rekursiv hochladen
mc cp --recursive ~/documents/ myminio/study/
```

Upload-Fortschritt überwachen:

```bash
mc ls myminio/study/
```

Der Ingestion Worker erkennt und verarbeitet neue Dateien im Bucket automatisch.

## 6. Zugriff auf Services

Sobald alle Services laufen, kann über Vast.ai Port Forwarding oder SSH-Tunneling darauf zugegriffen werden.

### Via SSH Tunnel (Empfohlen)

SSH-Tunnel für jeden Service erstellen:

```bash
# Auf dem lokalen Rechner
ssh -L 80:localhost:80 \
    -L 8080:localhost:8080 \
    -L 9001:localhost:9001 \
    -L 6333:localhost:6333 \
    root@ihre-vast-instanz-ip
```

Dann lokal auf die Services zugreifen:

- **Frontend**: http://localhost
- **API Docs**: http://localhost:8080/docs
- **MinIO Console**: http://localhost:9001
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### Via Vast.ai Port Forwarding

Alternativ kann das eingebaute Port Forwarding Feature von Vast.ai genutzt werden (siehe Instanz-Details für freigegebene Ports).

## 7. Monitoring und Troubleshooting

### Service-Status prüfen

```bash
docker-compose ps
```

### Logs anzeigen

```bash
# Alle Services
docker-compose logs -f

# Spezifischer Service
docker-compose logs -f worker
docker-compose logs -f rag-api
docker-compose logs -f ollama
```

### GPU-Nutzung prüfen

```bash
nvidia-smi
```

### Speicherplatz prüfen

```bash
df -h
```

### Services neu starten

```bash
# Alle Services neu starten
docker-compose restart

# Spezifischen Service neu starten
docker-compose restart worker
```

## 8. Pipeline Evaluation ausführen

Nachdem Dokumente verarbeitet wurden, automatisierte Evaluation ausführen:

```bash
cd automated-pipeline-evaluation

# Quick-Test (13 Pipelines)
python validate_pipeline_dimensions.py

# Vollständige Evaluation (56 Pipelines)
python run_full_evaluation.py --results-file ../results/pipeline_test_results.json
```

## 9. Aufräumen

Nach Abschluss alle Services stoppen:

```bash
docker-compose down
```

Um alle Daten (Volumes) zu entfernen:

```bash
docker-compose down -v
```

## Häufige Probleme

### Problem: Docker kann nicht auf GPU zugreifen

**Lösung**: Sicherstellen, dass NVIDIA Container Toolkit korrekt installiert und Docker neu gestartet wurde.

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Problem: MinIO Bucket nicht erreichbar

**Lösung**: Sicherstellen, dass MinIO Service läuft und Alias korrekt konfiguriert ist.

```bash
docker-compose ps minio
mc alias set myminio http://localhost:9000 minioadmin minioadmin123
```

### Problem: Kein Speicherplatz mehr

**Lösung**: Docker-Ressourcen aufräumen.

```bash
docker system prune -a
```

### Problem: Ollama startet nicht

**Lösung**: GPU-Verfügbarkeit und VRAM prüfen. Bei Systemen mit wenig VRAM GPU-Cache-Clearing verwenden.

```bash
# In .env.local
GPU_CACHE_CLEAR_BETWEEN_PHASES=true
```

## Tipps zur Kostenoptimierung

1. **Instanz pausieren, wenn nicht in Verwendung**: Vast.ai berechnet stundenweise
2. **Spot-Instanzen nutzen**: Günstiger, können aber unterbrochen werden
3. **GPU-Nutzung überwachen**: Sicherstellen, dass Workload tatsächlich die GPU nutzt
4. **Dokumentenverarbeitung batchen**: Mehrere Dokumente auf einmal hochladen
5. **Regelmäßig aufräumen**: Alte Docker Images und Container entfernen

## Sicherheitshinweise

1. **Standard-Passwörter ändern**: MinIO Credentials in `.env.local` aktualisieren
2. **SSH-Tunneling nutzen**: Services nicht direkt ins Internet exponieren
3. **Zugriffslogs überwachen**: Auf unautorisierte Zugriffsversuche prüfen
4. **System aktuell halten**: Packages und Docker Images regelmäßig updaten

## Nächste Schritte

- Siehe [README.md](README.md) für allgemeine Systemdokumentation
- Siehe [automated-pipeline-evaluation/usage.md](automated-pipeline-evaluation/usage.md) für Evaluation-Details
- Siehe [evaluation/README.md](evaluation/README.md) für RAGAS-Metriken

## Support

Bei Problemen zu:
- **Vast.ai Plattform**: Vast.ai Support kontaktieren
- **Diesem Projekt**: Issue auf GitHub öffnen
- **Docker/GPU-Probleme**: NVIDIA Container Toolkit Dokumentation prüfen
