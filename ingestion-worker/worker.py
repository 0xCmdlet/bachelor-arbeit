import os, json, structlog, traceback, gc
from datetime import datetime, timezone
from typing import List, Tuple
from contextlib import closing

from minio import Minio
from minio.error import S3Error
from tenacity import retry, stop_after_attempt, wait_exponential
import psycopg2

# Enhanced extractors and chunking
from extractor import detect_mime, extract_text, extract_text_with_strategy  # Docling, Unstructured, or LlamaParse
from chunking import (
    chunk_text,
    create_chunks_from_llamaparse_with_chonkie,
    create_chunks_from_docling_with_chonkie,
    create_chunks_from_docling_with_semantic,
    create_chunks_from_llamaparse_with_semantic
)
from embedding import (                                       # Phase 3: embeddings -> Qdrant
    get_embedding_provider,
    ensure_collection,
    embed_chunks,
    generate_sparse_vectors,
    upsert_to_qdrant,
    QDRANT_COLLECTION,
    RETRIEVAL_STRATEGY,
)

log = structlog.get_logger()

# ---------- ENV ----------
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes", "on")

PG_DSN = os.getenv("POSTGRES_DSN", "")

REQUIRED_ENVS = {
    "MINIO_ENDPOINT": MINIO_ENDPOINT,
    "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
    "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
    "MINIO_BUCKET": MINIO_BUCKET,
    "POSTGRES_DSN": PG_DSN,
}

# Chunking strategy configuration
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "langchain").lower()

# LangChain chunking parameters
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Chonkie chunking parameters
CHONKIE_CHUNK_SIZE = int(os.getenv("CHONKIE_CHUNK_SIZE", "512"))
CHONKIE_OVERLAP = int(os.getenv("CHONKIE_OVERLAP", "128"))

# Chunk validation limits
MAX_CHUNK_SIZE_BYTES = int(os.getenv("MAX_CHUNK_SIZE_BYTES", "50000"))  # 50KB max
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "2000"))  # Token safety limit

# ---------- DB ----------
def pg():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn

def upsert_file(conn, filename, size_bytes, etag=None, mime=None, meta=None, processed=None):
    """Insert or update file record in PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO files (filename, size_bytes, etag, mime, meta, processed)
            VALUES (%s, %s, %s, %s, %s::jsonb, COALESCE(%s, FALSE))
            ON CONFLICT (filename) DO UPDATE SET
                size_bytes = EXCLUDED.size_bytes,
                etag       = EXCLUDED.etag,
                mime       = COALESCE(EXCLUDED.mime, files.mime),
                -- merge existing meta with new (new keys override)
                meta       = COALESCE(files.meta, '{}'::jsonb) || COALESCE(EXCLUDED.meta, '{}'::jsonb),
                -- only change processed if caller provided a value
                processed  = COALESCE(EXCLUDED.processed, files.processed),
                updated_at = now()
            RETURNING id
            """,
            (filename, size_bytes, etag, mime, json.dumps(meta or {}), processed),
        )
        return cur.fetchone()[0]  # Return the file ID

def validate_chunk_size(chunk, max_size_bytes: int, max_tokens: int) -> Tuple[bool, str]:
    """Validate chunk size and token limits."""
    # Check byte size
    text_size = len(chunk.text.encode('utf-8'))
    if text_size > max_size_bytes:
        return False, f"Chunk size {text_size} bytes exceeds limit {max_size_bytes}"

    # Check token count
    if chunk.token_count > max_tokens:
        return False, f"Chunk tokens {chunk.token_count} exceeds limit {max_tokens}"

    return True, ""

def upsert_chunks_to_postgres(conn, file_id: int, chunks: List, collection: str):
    """Insert chunk metadata into PostgreSQL chunks table with enhanced metadata."""
    if not chunks:
        return

    with conn.cursor() as cur:
        # Clear existing chunks for this file (for reprocessing scenarios)
        cur.execute("DELETE FROM chunks WHERE file_id = %s", (file_id,))

        # Prepare chunk data for batch insert with enhanced metadata
        chunk_data = []
        valid_chunks = []
        invalid_count = 0

        for chunk in chunks:
            # Validate chunk size
            is_valid, error_msg = validate_chunk_size(chunk, MAX_CHUNK_SIZE_BYTES, MAX_CHUNK_TOKENS)

            if not is_valid:
                log.warning(
                    "chunk_validation_failed",
                    chunk_id=chunk.chunk_id,
                    file_id=file_id,
                    error=error_msg,
                    chunk_size_bytes=len(chunk.text.encode('utf-8')),
                    chunk_tokens=chunk.token_count
                )
                invalid_count += 1
                continue

            valid_chunks.append(chunk)

            # Store metadata as JSON in the existing meta field
            chunk_meta = {
                "chunk_type": getattr(chunk, 'chunk_type', 'unknown'),
                "token_count": chunk.token_count,
                "size_bytes": len(chunk.text.encode('utf-8'))
            }

            # Conditional storage based on chunk type to avoid PostgreSQL index size limits
            if chunk.chunk_type.startswith("unstructured_"):
                # Unstructured: store element type only (avoid large nested metadata)
                chunk_meta["element_type"] = chunk.metadata.get("element_type", "")
                chunk_meta["extractor"] = "unstructured"
            else:
                # Docling/LlamaParse: store markdown headers (small, semantic structure)
                chunk_meta["headers"] = chunk.metadata if chunk.metadata else {}

            chunk_data.append((
                chunk.chunk_id,
                file_id,
                chunk.index,
                chunk.start_token,
                chunk.end_token,
                collection,
                chunk.text,
                json.dumps(chunk_meta)  # Store enhanced metadata
            ))

        # Batch insert all chunks with metadata
        cur.executemany(
            """
            INSERT INTO chunks (chunk_id, file_id, chunk_index, start_token, end_token, collection, text, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                meta = EXCLUDED.meta,
                start_token = EXCLUDED.start_token,
                end_token = EXCLUDED.end_token
            """,
            chunk_data
        )

        log.info(
            "enhanced_chunks_stored_postgres",
            file_id=file_id,
            total_chunks=len(chunks),
            valid_chunks=len(valid_chunks),
            invalid_chunks=invalid_count,
            collection=collection,
            chunks_with_headers=sum(1 for c in valid_chunks if c.metadata and any(k.startswith("Header") for k in c.metadata.keys()))
        )

        return valid_chunks  # Return only valid chunks for further processing

def get_file_id(conn, filename: str) -> int:
    """Get file ID for a given filename."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM files WHERE filename = %s", (filename,))
        result = cur.fetchone()
        if result is None:
            raise ValueError(f"File not found: {filename}")
        return result[0]

# ---------- MinIO ----------
def minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
def ensure_bucket(mc: Minio):
    if not mc.bucket_exists(MINIO_BUCKET):
        raise RuntimeError(f"Bucket {MINIO_BUCKET} does not exist")

# ---------- Helpers ----------
def validate_env():
    missing = [k for k, v in REQUIRED_ENVS.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

def _clear_gpu_memory():
    """Clear GPU memory cache to free memory between processing phases."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log.info("gpu_cache_cleared")  # Fixed: removed duplicate 'event' parameter
    except ImportError:
        pass  # PyTorch not available
    except Exception as e:
        log.warning("gpu_clear_failed", error=str(e))

# ---------- Enhanced Core processing ----------
def process_object(conn, mc: Minio, obj) -> None:
    """Enhanced processing: configurable extraction + chunking + embeddings."""
    if getattr(obj, "is_dir", False):
        return

    key = obj.object_name
    etag = getattr(obj, "etag", None)
    size = getattr(obj, "size", None)

    log.info("Key + Etag + Size", key=key, size=size, etag=etag)

    # Download bytes (ensure connection closes on all paths)
    try:
        with closing(mc.get_object(MINIO_BUCKET, key)) as resp:
            raw_bytes = resp.read()
    except S3Error as e:
        log.error("minio_get_error", key=key, error=str(e))
        return
    except Exception as e:
        log.error("download_error", key=key, error=str(e), tb=traceback.format_exc())
        return

    # Phase 1: Enhanced text extraction (Docling, Unstructured, or LlamaParse based on EXTRACTOR_TYPE)
    try:
        mime = detect_mime(key, raw_bytes)
        extractor_type = os.getenv("EXTRACTOR_TYPE", "docling").lower()

        # Unstructured returns chunks directly, others return text
        if extractor_type == "unstructured":
            from extractor import UnstructuredExtractor
            extractor = UnstructuredExtractor()
            chunks, meta = extractor.extract(key, raw_bytes, mime)
            text = None  # No combined text for Unstructured
        else:
            text, meta = extract_text_with_strategy(key, raw_bytes, mime)
            chunks = None  # Will be created in chunking phase

        # Enhance metadata with extraction info
        meta.update({
            "etag": etag,
            "size_bytes": size,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        })

        log.info(
            "extraction_complete",
            key=key,
            mime=mime,
            text_length=len(text) if text else sum(len(c.text) for c in chunks) if chunks else 0,
            extractor=extractor_type,
            ocr_used=meta.get("ocr_used", False),
            layout_detection=meta.get("layout_detection", False),
            table_extraction=meta.get("table_extraction", False),
            formula_detection=meta.get("formula_detection", False)
        )

        # Clear GPU cache after extraction to free memory for embedding phase
        # Can be disabled for 24GB VRAM to reduce overhead
        if os.getenv("GPU_CACHE_CLEAR_BETWEEN_PHASES", "true").lower() == "true":
            _clear_gpu_memory()

    except Exception as e:
        log.error("extraction_error", key=key, error=str(e), tb=traceback.format_exc())
        upsert_file(conn, key, size, etag=etag, mime=None,
                    meta={"extract_error": str(e)}, processed=False)
        return

    # Phase 2: Chunking (strategy depends on extractor and CHUNKING_STRATEGY)
    try:
        if extractor_type == "unstructured":
            # Chunks already created by Unstructured API - skip chunking phase entirely
            chunks_with_headers = 0  # Not applicable for Unstructured
            log.info(
                "unstructured_chunks_direct",
                key=key,
                total_chunks=len(chunks),
                element_types=meta.get("element_types", [])
            )
        elif extractor_type == "llamaparse" and CHUNKING_STRATEGY == "chonkie":
            # LlamaParse + Chonkie RecursiveChunker
            chunks = list(create_chunks_from_llamaparse_with_chonkie(
                text,
                meta,
                chunk_size=CHONKIE_CHUNK_SIZE,
                overlap=CHONKIE_OVERLAP,
                max_tokens=MAX_CHUNK_TOKENS
            ))
            chunks_with_headers = 0  # Not applicable for Chonkie

            log.info(
                "chonkie_chunking_complete",
                key=key,
                total_chunks=len(chunks),
                chunk_size=CHONKIE_CHUNK_SIZE,
                overlap=CHONKIE_OVERLAP,
                avg_tokens_per_chunk=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            )
        elif extractor_type == "docling" and CHUNKING_STRATEGY == "chonkie":
            # Docling + Chonkie RecursiveChunker
            chunks = list(create_chunks_from_docling_with_chonkie(
                text,
                meta,
                chunk_size=CHONKIE_CHUNK_SIZE,
                overlap=CHONKIE_OVERLAP,
                max_tokens=MAX_CHUNK_TOKENS
            ))
            chunks_with_headers = 0  # Not applicable for Chonkie

            log.info(
                "chonkie_chunking_complete",
                key=key,
                extractor="docling",
                total_chunks=len(chunks),
                chunk_size=CHONKIE_CHUNK_SIZE,
                overlap=CHONKIE_OVERLAP,
                avg_tokens_per_chunk=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            )
        elif extractor_type == "docling" and CHUNKING_STRATEGY == "semantic":
            # Docling + Semantic Chunker (embeddings-based)
            chunks = list(create_chunks_from_docling_with_semantic(
                text,
                meta,
                model_name=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
                breakpoint_threshold_type="percentile",
                max_tokens=MAX_CHUNK_TOKENS
            ))
            chunks_with_headers = 0  # Not applicable for semantic chunking

            log.info(
                "semantic_chunking_complete",
                key=key,
                extractor="docling",
                total_chunks=len(chunks),
                embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
                avg_tokens_per_chunk=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            )
        elif extractor_type == "llamaparse" and CHUNKING_STRATEGY == "semantic":
            # LlamaParse + Semantic Chunker (embeddings-based)
            chunks = list(create_chunks_from_llamaparse_with_semantic(
                text,
                meta,
                model_name=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
                breakpoint_threshold_type="percentile",
                max_tokens=MAX_CHUNK_TOKENS
            ))
            chunks_with_headers = 0  # Not applicable for semantic chunking

            log.info(
                "semantic_chunking_complete",
                key=key,
                extractor="llamaparse",
                total_chunks=len(chunks),
                embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
                avg_tokens_per_chunk=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            )
        else:
            # Docling or LlamaParse + LangChain markdown-aware chunking
            chunks = list(chunk_text(text, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP))

            # Analyze chunking quality
            chunks_with_headers = sum(1 for c in chunks if c.metadata and any(k.startswith("Header") for k in c.metadata.keys()))

            log.info(
                "langchain_chunking_complete",
                key=key,
                total_chunks=len(chunks),
                chunks_with_headers=chunks_with_headers,
                avg_tokens_per_chunk=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
            )

    except Exception as e:
        log.error("chunking_error", key=key, error=str(e), tb=traceback.format_exc())
        # Fallback: still save file metadata but mark as chunking failed
        upsert_file(conn, key, size, etag=etag, mime=mime,
                   meta={**meta, "chunking_error": str(e)}, processed=False)
        return

    # Persist file record as processed (text extracted; chunks created)
    file_id = upsert_file(conn, key, size, etag=etag, mime=mime, meta=meta, processed=True)

    # Phase 3: embeddings -> Qdrant (same as before)
    try:
        # Ensure collection with correct vector size from provider
        provider = get_embedding_provider()
        vec_dim = provider.get_dimensions()
        ensure_collection(vector_size=vec_dim)

        # Embed and upsert to Qdrant
        chunk_list, embs = embed_chunks(chunks)

        # Generate sparse vectors if hybrid mode enabled
        sparse_embs = None
        if "hybrid" in RETRIEVAL_STRATEGY.lower():
            print(f"Generating sparse vectors for hybrid search...")
            sparse_embs = generate_sparse_vectors(chunk_list)

        upsert_to_qdrant(
            filename=key,
            chunks=chunk_list,
            embeddings=embs,
            common_meta={
                "mime": mime,
                "etag": etag,
                "ocr_used": bool(meta.get("ocr_used")),
                "size_bytes": size,
                "extracted_at": meta.get("extracted_at"),
                "layout_detection": bool(meta.get("layout_detection")),
                "table_extraction": bool(meta.get("table_extraction")),
                "chunks_with_headers": chunks_with_headers
            },
            sparse_embeddings=sparse_embs,
        )

        # Clean up embedding model to free GPU memory
        if hasattr(provider, 'cleanup'):
            provider.cleanup()

        # Phase 4: Store enhanced chunk metadata in PostgreSQL (with validation)
        valid_chunk_list = upsert_chunks_to_postgres(conn, file_id, chunk_list, QDRANT_COLLECTION)

        log.info(
            "enhanced_file_processed_complete",
            key=key,
            file_id=file_id,
            mime=mime,
            size=size,
            etag=etag,
            characters=len(text) if text else sum(len(c.text) for c in valid_chunk_list),
            chunk_count=len(valid_chunk_list),
            chunks_with_headers=chunks_with_headers,
            vector_dim=int(embs.shape[1]) if embs.size else 0,
            extraction_features={
                "ocr_used": bool(meta.get("ocr_used")),
                "layout_detection": bool(meta.get("layout_detection")),
                "table_extraction": bool(meta.get("table_extraction")),
                "formula_detection": bool(meta.get("formula_detection"))
            },
            preview=text[:200] if text else (valid_chunk_list[0].text[:200] if valid_chunk_list else ""),
        )

    except Exception as e:
        log.error("embedding_or_qdrant_error", key=key, error=str(e), tb=traceback.format_exc())
        # Mark file as unprocessed if embedding fails
        try:
            upsert_file(conn, key, size, etag=etag, mime=mime,
                       meta={**meta, "embed_error": str(e)}, processed=False)
        except Exception as db_e:
            log.error("failed_to_update_file_error_status", key=key, error=str(db_e))

def main():
    """Enhanced main worker function: configurable extraction + chunking + embeddings."""
    structlog.configure(processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ])

    validate_env()

    conn = pg()
    mc = minio_client()
    ensure_bucket(mc)

    log.info(
        "enhanced_worker_started",
        bucket=MINIO_BUCKET,
        minio_secure=MINIO_SECURE,
        chunk_tokens=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP
    )

    processed_count = 0
    error_count = 0

    for obj in mc.list_objects(MINIO_BUCKET, recursive=True):
        try:
            process_object(conn, mc, obj)
            processed_count += 1
        except Exception as e:
            error_count += 1
            log.error(
                "object_process_failed",
                key=getattr(obj, "object_name", None),
                error=str(e),
                tb=traceback.format_exc()
            )

    log.info(
        "worker_complete",
        bucket=MINIO_BUCKET,
        processed_count=processed_count,
        error_count=error_count
    )

    # Clean up GPU memory before exit, so when the next pipeline runs, it has max memory available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            log.info("gpu_cleanup_on_exit", status="success")
    except Exception as e:
        log.warning("gpu_cleanup_on_exit", status="failed", error=str(e))

if __name__ == "__main__":
    main()