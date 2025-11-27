# embeddings.py
from __future__ import annotations
from typing import Iterable, List, Tuple
from abc import ABC, abstractmethod
import os
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseIndexParams, SparseVector
from chunking import MarkdownChunk

# ==========================
# Config (env-overridable)
# ==========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

# Embedding provider configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")  # "sentence-transformers" or "openai"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

# OpenAI specific config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# GPU cache clearing counter (module-level)
_gpu_cache_clear_counter = 0

# For retrieval, most instruction-tuned embedders expect a task prefix.
# Note: OpenAI embeddings don't use prefixes
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: ")

# Retrieval strategy for hybrid search
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "dense").lower()
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

# Lazy singletons
_provider = None
_client: QdrantClient | None = None


# ==========================
# Provider Architecture
# ==========================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of texts and return normalized embeddings.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            np.ndarray of shape (len(texts), dimensions) with normalized embeddings
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the embedding dimension size"""
        pass

    @abstractmethod
    def supports_prefix(self) -> bool:
        """Whether this provider supports document/query prefixes"""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Provider for SentenceTransformer (HuggingFace) models"""

    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size

        # Configurable device selection based on VRAM availability
        embedding_device = os.getenv("EMBEDDING_DEVICE", "auto").lower()
        chunking_strategy = os.getenv("CHUNKING_STRATEGY", "langchain").lower()
        extractor_type = os.getenv("EXTRACTOR_TYPE", "docling").lower()

        if embedding_device == "cuda":
            device = "cuda"
            print(f"Using GPU for embeddings (forced via EMBEDDING_DEVICE=cuda)")
        elif embedding_device == "cpu":
            device = "cpu"
            print(f"Using CPU for embeddings (forced via EMBEDDING_DEVICE=cpu)")
        elif embedding_device == "auto":
            # Smart selection for 8GB VRAM: Use CPU for Docling + Semantic to avoid GPU OOM
            # GPU is free for LlamaParse/Unstructured + Semantic (cloud extraction)
            if chunking_strategy == "semantic" and extractor_type == "docling":
                device = "cpu"
                print(f"Using CPU for embeddings (Docling + semantic chunking, auto mode)")
            else:
                device = None  # Auto-detect (use GPU if available)
                print(f"Using auto-detect for embeddings (will use GPU if available)")
        else:
            device = None  # Fallback to auto-detect
            print(f"Unknown EMBEDDING_DEVICE value '{embedding_device}', using auto-detect")

        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        print(f"Loaded SentenceTransformer: {model_name}")

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)

    def get_dimensions(self) -> int:
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback for common models
            if "384" in self.model_name or "MiniLM" in self.model_name:
                return 384
            return 768

    def supports_prefix(self) -> bool:
        return True  # SentenceTransformers support prefixes

    def cleanup(self):
        """Clean up GPU memory by deleting model and clearing cache"""
        try:
            import gc
            import torch

            # Delete model
            if hasattr(self, 'model'):
                del self.model

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()
        except Exception:
            pass  # Silently ignore cleanup errors


class OpenAIProvider(EmbeddingProvider):
    """Provider for OpenAI embedding models"""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

        # Dimension mapping for OpenAI models
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        print(f"Initialized OpenAI embeddings: {model_name}")

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Embed texts using OpenAI API.
        OpenAI handles batching internally, but we'll batch for cost tracking.
        """
        # OpenAI can handle up to 2048 texts per request, we'll use smaller batches
        batch_size = 100
        all_embeddings = []

        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            if show_progress and total_batches > 1:
                print(f"OpenAI embedding batch {i // batch_size + 1}/{total_batches}")

            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        # Convert to numpy and normalize
        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def get_dimensions(self) -> int:
        return self.dimension_map.get(self.model_name, 1536)

    def supports_prefix(self) -> bool:
        return False  # OpenAI doesn't use prefixes


# ==========================
# Sparse Vectorizer (TF-IDF)
# ==========================

class SimpleTFIDFVectorizer:
    """
    Simple TF-IDF sparse vectorizer for hybrid search.

    Uses term frequency normalized by document length (no global IDF stats).
    This provides lexical matching without requiring a global corpus vocabulary.
    """

    def __init__(self):
        """Initialize the vectorizer with an empty vocabulary."""
        self.vocab = {}  # Maps token -> index
        self.next_idx = 0

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, remove punctuation, split on whitespace.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        import re
        # Lowercase
        text = text.lower()
        # Remove punctuation except hyphens and apostrophes
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        # Split on whitespace
        tokens = text.split()
        # Remove very short tokens (single chars)
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def vectorize_batch(self, texts: List[str]) -> List[dict]:
        """
        Vectorize a batch of texts, building vocabulary from the batch.

        Args:
            texts: List of text strings

        Returns:
            List of sparse vectors in Qdrant format:
            {"indices": [int, ...], "values": [float, ...]}
        """
        # Reset vocabulary for this batch
        self.vocab = {}
        self.next_idx = 0

        # First pass: build vocabulary from all texts
        all_tokens_list = []
        for text in texts:
            tokens = self._tokenize(text)
            all_tokens_list.append(tokens)

            # Add to vocabulary
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.next_idx
                    self.next_idx += 1

        # Second pass: create sparse vectors
        sparse_vectors = []
        for tokens in all_tokens_list:
            if not tokens:
                # Empty document - return empty sparse vector
                sparse_vectors.append({"indices": [], "values": []})
                continue

            # Count term frequencies
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            # Normalize by document length
            doc_length = len(tokens)

            # Build sparse vector
            indices = []
            values = []
            for token, freq in term_freq.items():
                if token in self.vocab:
                    indices.append(self.vocab[token])
                    # Simple TF normalization
                    values.append(float(freq) / doc_length)

            # Sort by indices (required by some vector DBs)
            if indices:
                sorted_pairs = sorted(zip(indices, values))
                indices, values = zip(*sorted_pairs)
                indices = list(indices)
                values = list(values)

            sparse_vectors.append({
                "indices": indices,
                "values": values
            })

        return sparse_vectors


def get_embedding_provider() -> EmbeddingProvider:
    """
    Factory function to get the configured embedding provider.
    Returns a singleton instance.
    """
    global _provider

    if _provider is None:
        provider_type = EMBEDDING_PROVIDER.lower()

        if provider_type == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI provider")
            _provider = OpenAIProvider(
                api_key=OPENAI_API_KEY,
                model_name=OPENAI_EMBEDDING_MODEL
            )
        else:
            # Default to SentenceTransformer
            _provider = SentenceTransformerProvider(
                model_name=EMBEDDING_MODEL,
                batch_size=EMBED_BATCH
            )

    return _provider


def get_qdrant() -> QdrantClient:
    """Load (once) and return the Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


# --------------------
# Embedding
# --------------------
def _format_doc_inputs(texts: List[str]) -> List[str]:
    """
    Apply the document prompt prefix if the provider supports it.
    E.g., 'search_document: <chunk text>'
    """
    provider = get_embedding_provider()

    if not provider.supports_prefix() or not EMBED_DOC_PREFIX:
        return texts

    pref = EMBED_DOC_PREFIX
    return [f"{pref}{t}" for t in texts]


def _clear_gpu_cache():
    """
    Clear GPU cache based on configured interval.
    Safe to call even if GPU is not available.

    Set GPU_CACHE_CLEAR_INTERVAL=1 for aggressive clearing (8GB VRAM)
    Set GPU_CACHE_CLEAR_INTERVAL=10 for less overhead (24GB VRAM)
    """
    global _gpu_cache_clear_counter

    interval = int(os.getenv("GPU_CACHE_CLEAR_INTERVAL", "1"))
    _gpu_cache_clear_counter += 1

    # Only clear if we've reached the interval
    if _gpu_cache_clear_counter >= interval:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            _gpu_cache_clear_counter = 0  # Reset counter
        except ImportError:
            pass
        except Exception:
            pass


def embed_chunks(chunks: Iterable[MarkdownChunk]) -> Tuple[List[MarkdownChunk], np.ndarray]:
    """
    Embed chunk texts using the configured provider.

    Processes chunks in groups to avoid memory issues.

    Returns:
        (chunks_list, embeddings) where embeddings is shape (N, D) float32 (normalized).
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return [], np.zeros((0, 0), dtype=np.float32)

    # Get provider
    provider = get_embedding_provider()

    # Process in batches to avoid memory issues
    MAX_CHUNKS_PER_BATCH = int(os.getenv("MAX_CHUNKS_PER_BATCH", "25"))
    all_embeddings = []
    total_batches = (len(chunks_list) + MAX_CHUNKS_PER_BATCH - 1) // MAX_CHUNKS_PER_BATCH

    # Process chunks in manageable batches
    for batch_num, batch_start in enumerate(range(0, len(chunks_list), MAX_CHUNKS_PER_BATCH), 1):
        batch_end = min(batch_start + MAX_CHUNKS_PER_BATCH, len(chunks_list))
        batch_chunks = chunks_list[batch_start:batch_end]

        # Log progress for large documents
        if total_batches > 1:
            print(f"Embedding batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")

        # Extract texts and format inputs
        texts = [c.text for c in batch_chunks]
        inputs = _format_doc_inputs(texts)

        # Embed this batch using provider
        batch_embs = provider.embed(inputs, show_progress=False)
        all_embeddings.append(batch_embs)

        # Clear GPU cache after each batch (only matters for SentenceTransformers)
        _clear_gpu_cache()

    # Combine all batch embeddings
    if len(all_embeddings) == 1:
        embs = all_embeddings[0]
    else:
        embs = np.vstack(all_embeddings)

    return chunks_list, embs.astype(np.float32, copy=False)


def generate_sparse_vectors(chunks: Iterable[MarkdownChunk]) -> List[dict]:
    """
    Generate sparse TF-IDF vectors for chunks (for hybrid search).

    Args:
        chunks: Iterable of MarkdownChunk objects

    Returns:
        List of sparse vectors in Qdrant format:
        [{"indices": [int, ...], "values": [float, ...]}, ...]
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return []

    # Extract texts
    texts = [c.text for c in chunks_list]

    # Generate sparse vectors
    vectorizer = SimpleTFIDFVectorizer()
    sparse_vectors = vectorizer.vectorize_batch(texts)

    print(f"Generated {len(sparse_vectors)} sparse vectors (vocab size: {vectorizer.next_idx})")

    return sparse_vectors


# --------------------
# Qdrant
# --------------------
def ensure_collection(vector_size: int | None = None, distance: Distance = Distance.COSINE) -> None:
    """
    Ensure the Qdrant collection exists with the expected vector size.
    If not provided, get vector size from the provider.
    If existing collection size differs, it will be recreated.

    Supports hybrid mode (dense + sparse vectors) based on RETRIEVAL_STRATEGY.
    """

    client = get_qdrant()

    # Determine if hybrid mode is enabled
    use_hybrid = "hybrid" in RETRIEVAL_STRATEGY

    # Try to read existing collection info
    exists = False
    current_size = None
    is_hybrid = False
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        exists = True
        try:
            # Check if it's a hybrid collection (named vectors)
            if hasattr(info.config.params.vectors, '__getitem__'):
                is_hybrid = True
                current_size = info.config.params.vectors.get('dense', {}).get('size')
            else:
                current_size = info.config.params.vectors.size
        except Exception:
            current_size = None
    except Exception:
        exists = False

    # Recreate if:
    # 1. Collection doesn't exist
    # 2. Vector size changed
    # 3. Hybrid mode changed (dense-only <-> hybrid)
    needs_recreate = (
        not exists or
        (current_size and current_size != vector_size) or
        (is_hybrid != use_hybrid)
    )

    if needs_recreate:
        if use_hybrid:
            # Create hybrid collection with named vectors
            print(f"Creating hybrid collection (dense + sparse): {QDRANT_COLLECTION}")
            client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=int(vector_size), distance=distance),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        )
                    ),
                },
            )
        else:
            # Create dense-only collection (backward compatible)
            print(f"Creating dense-only collection: {QDRANT_COLLECTION}")
            client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=int(vector_size), distance=distance),
            )


def upsert_to_qdrant(
    filename: str,
    chunks: List[MarkdownChunk],
    embeddings: np.ndarray,
    common_meta: dict | None = None,
    sparse_embeddings: List[dict] | None = None,
) -> None:
    """
    Upsert chunk embeddings into Qdrant.

    - Uses stable chunk_id (32-hex) → UUID for point ids.
    - Stores useful metadata (filename, index, token offsets, preview, model).
    - Supports hybrid mode (dense + sparse vectors) if sparse_embeddings provided.
    """
    if not chunks:
        return
    if embeddings.shape[0] != len(chunks):
        raise ValueError("embeddings count does not match chunks")
    if sparse_embeddings and len(sparse_embeddings) != len(chunks):
        raise ValueError("sparse_embeddings count does not match chunks")

    client = get_qdrant()
    provider = get_embedding_provider()

    # Determine if hybrid mode
    use_hybrid = "hybrid" in RETRIEVAL_STRATEGY and sparse_embeddings is not None

    payload_common = dict(common_meta or {})

    # Store embedding model info based on provider
    if isinstance(provider, OpenAIProvider):
        embed_model_name = f"openai:{provider.model_name}"
    else:
        embed_model_name = provider.model_name

    payload_common.update({
        "filename": filename,
        "embed_model": embed_model_name,
        "embed_prefix": EMBED_DOC_PREFIX if provider.supports_prefix() else None,
        "embed_provider": EMBEDDING_PROVIDER,
        "retrieval_strategy": RETRIEVAL_STRATEGY,
    })

    points: List[PointStruct] = []

    if use_hybrid:
        # Hybrid mode: upload both dense and sparse vectors
        for ch, dense_vec, sparse_vec in zip(chunks, embeddings, sparse_embeddings):
            point_id = uuid.UUID(hex=ch.chunk_id)
            payload = {
                **payload_common,
                "chunk_id": ch.chunk_id,
                "index": ch.index,
                "token_count": ch.token_count,
                "start_token": ch.start_token,
                "end_token": ch.end_token,
                "preview": ch.text[:240],
            }

            # Create hybrid vector (named vectors + sparse)
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"]
                        )
                    },
                    payload=payload,
                )
            )
    else:
        # Dense-only mode (backward compatible)
        for ch, vec in zip(chunks, embeddings):
            point_id = uuid.UUID(hex=ch.chunk_id)
            payload = {
                **payload_common,
                "chunk_id": ch.chunk_id,
                "index": ch.index,
                "token_count": ch.token_count,
                "start_token": ch.start_token,
                "end_token": ch.end_token,
                "preview": ch.text[:240],
            }
            points.append(
                PointStruct(
                    id=str(point_id),
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
