FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    libpq5 ca-certificates \
    tesseract-ocr poppler-utils libmagic1 \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Use faster pip settings and parallel downloads
RUN pip install --timeout 300 --retries 5 --no-cache-dir -r requirements.txt

COPY extractor.py .
COPY worker.py .
COPY embedding.py .
COPY chunking.py .

CMD ["python", "worker.py"]