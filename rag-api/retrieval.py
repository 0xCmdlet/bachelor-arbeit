import os
import structlog
from typing import List, Dict, Any
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue, SparseVector, Prefetch, FusionQuery, Fusion
import psycopg2
import json
import torch  # For sigmoid activation in reranking

logger = structlog.get_logger()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://admin:admin123@postgres:5432/mydb")

# Embedding provider configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")

# OpenAI specific config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Reranking configuration
RERANKING_STRATEGY = os.getenv("RERANKING_STRATEGY", "none").lower()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K_MULTIPLIER = int(os.getenv("RERANK_TOP_K_MULTIPLIER", "3"))

# Hybrid search configuration
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "dense").lower()
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))


class SimpleTokenizer:
    """
    Simple tokenizer for hybrid search queries.
    Matches the tokenization logic from ingestion worker.
    """

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text: lowercase, remove punctuation, split on whitespace.

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

    def vectorize(self, text: str, vocab: Dict[str, int]) -> Dict:
        """
        Convert text to sparse vector format using provided vocabulary.

        For query-time, we don't have a vocabulary built from the corpus.
        Instead, we'll create a simple TF-based sparse vector.

        Args:
            text: Query text
            vocab: Not used at query time (placeholder for API compatibility)

        Returns:
            Sparse vector: {"indices": [...], "values": [...]}
        """
        tokens = self.tokenize(text)

        if not tokens:
            return {"indices": [], "values": []}

        # Count term frequencies
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Normalize by query length
        query_length = len(tokens)

        # Build vocabulary on-the-fly from query tokens
        query_vocab = {token: idx for idx, token in enumerate(sorted(term_freq.keys()))}

        # Build sparse vector
        indices = []
        values = []
        for token, freq in term_freq.items():
            indices.append(query_vocab[token])
            values.append(float(freq) / query_length)

        # Sort by indices
        if indices:
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs)
            indices = list(indices)
            values = list(values)

        return {
            "indices": indices,
            "values": values
        }


class RAGRetriever:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.provider_type = EMBEDDING_PROVIDER.lower()

        # Initialize appropriate embedding model based on provider
        if self.provider_type == "openai":
            from openai import OpenAI
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI provider")
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.embedding_model_name = OPENAI_EMBEDDING_MODEL
            logger.info("rag_retriever_initialized", provider="openai", model=OPENAI_EMBEDDING_MODEL)
        else:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
            self.embedding_model_name = EMBEDDING_MODEL
            logger.info("rag_retriever_initialized", provider="sentence-transformers", model=EMBEDDING_MODEL)

        # Initialize reranker if enabled
        self.reranking_enabled = RERANKING_STRATEGY == "cross-encoder"
        self.reranker = None
        if self.reranking_enabled:
            from sentence_transformers import CrossEncoder

            # Configurable device for reranker (GPU recommended for 24GB VRAM)
            reranker_device = os.getenv("RERANKER_DEVICE", "auto").lower()
            if reranker_device == "cuda":
                self.reranker = CrossEncoder(RERANKER_MODEL, device='cuda')
                logger.info("reranker_initialized", model=RERANKER_MODEL, device="cuda", multiplier=RERANK_TOP_K_MULTIPLIER)
            elif reranker_device == "cpu":
                self.reranker = CrossEncoder(RERANKER_MODEL, device='cpu')
                logger.info("reranker_initialized", model=RERANKER_MODEL, device="cpu", multiplier=RERANK_TOP_K_MULTIPLIER)
            else:
                # Auto mode defaults to CPU (safe for 8GB VRAM)
                self.reranker = CrossEncoder(RERANKER_MODEL)
                logger.info("reranker_initialized", model=RERANKER_MODEL, device="auto (CPU)", multiplier=RERANK_TOP_K_MULTIPLIER)

        # Initialize hybrid search if enabled
        self.use_hybrid = "hybrid" in RETRIEVAL_STRATEGY
        self.tokenizer = None
        if self.use_hybrid:
            self.tokenizer = SimpleTokenizer()
            logger.info("hybrid_search_initialized", alpha=HYBRID_ALPHA, strategy=RETRIEVAL_STRATEGY)

        # Detect if collection uses named vectors (hybrid) or unnamed default vector (pure dense)
        self.collection_has_named_vectors = self._detect_collection_type()
        logger.info(
            "collection_type_detected",
            collection=QDRANT_COLLECTION,
            has_named_vectors=self.collection_has_named_vectors
        )

    def _detect_collection_type(self) -> bool:
        """
        Detect if collection uses named vectors (hybrid) or unnamed default vector (pure dense).

        Returns:
            True if collection uses named vectors (hybrid), False if unnamed (pure dense)
        """
        try:
            collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
            # Check if vectors config is a dictionary (named vectors)
            has_named = hasattr(collection_info.config.params.vectors, '__getitem__')
            return has_named
        except Exception as e:
            logger.warning(
                "collection_type_detection_failed",
                collection=QDRANT_COLLECTION,
                error=str(e),
                defaulting_to="named vectors"
            )
            # Default to named vectors (safer for hybrid collections)
            return True

    def _generate_sparse_query(self, query: str) -> Dict:
        """
        Generate sparse vector from query text.
        Uses same tokenization as ingestion (on-the-fly vocabulary).

        Args:
            query: Query text to tokenize

        Returns:
            Sparse vector dict with 'indices' and 'values' keys
        """
        if not self.tokenizer:
            return {"indices": [], "values": []}

        # Use existing tokenizer's vectorize method
        sparse_vec = self.tokenizer.vectorize(query, vocab={})
        return sparse_vec

    async def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.3, collection: str = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query

        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            collection: Override QDRANT_COLLECTION env var (useful for testing multiple collections)
        """
        try:
            # Use provided collection or fall back to environment variable
            collection_name = collection or QDRANT_COLLECTION

            # Step 1: Generate query embedding
            query_embedding = await self._embed_query(query)

            # Step 2: Determine how many candidates to retrieve
            # If reranking, retrieve more candidates; otherwise just get what we need
            retrieval_limit = top_k * RERANK_TOP_K_MULTIPLIER if self.reranking_enabled else top_k * 2

            # Step 3: Search Qdrant for similar vectors
            if self.use_hybrid:
                # True hybrid search: query both dense and sparse vectors
                query_sparse = self._generate_sparse_query(query)

                search_response = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        Prefetch(
                            query=query_embedding,
                            using="dense",
                            limit=retrieval_limit * 2,  # Get more candidates for fusion
                        ),
                        Prefetch(
                            query=SparseVector(
                                indices=query_sparse["indices"],
                                values=query_sparse["values"]
                            ),
                            using="sparse",
                            limit=retrieval_limit * 2,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
                    limit=retrieval_limit,
                    with_payload=True,
                )

                # Extract points from query_points response
                search_results = search_response.points

                logger.info(
                    "hybrid_search_completed",
                    query=query,
                    dense_query_dim=len(query_embedding),
                    sparse_query_terms=len(query_sparse["indices"]),
                    fusion_method="RRF"
                )
            else:
                # Dense-only search using query_points (qdrant-client 1.16+)
                # Conditionally use "dense" only for hybrid collections with named vectors
                query_params = {
                    "collection_name": collection_name,
                    "query": query_embedding,
                    "limit": retrieval_limit,
                    "with_payload": True
                }
                # Only add 'using' parameter for collections with named vectors (hybrid)
                if self.collection_has_named_vectors:
                    query_params["using"] = "dense"

                search_response = self.qdrant_client.query_points(**query_params)
                search_results = search_response.points

            # Step 4: Filter and enrich results
            enriched_results = []
            for result in search_results:
                # In dense-only mode, use min_score as a cosine similarity threshold
                if not self.use_hybrid and result.score < min_score:
                    continue

                # In hybrid (RRF) mode, DO NOT filter by score here,
                # because scores are rank-based and very small (e.g. 0.01)
                enriched_doc = await self._enrich_result(result)
                if enriched_doc:
                    enriched_results.append(enriched_doc)

            logger.info(
                "initial_retrieval_completed",
                query=query,
                candidates_found=len(enriched_results)
            )

            # Step 4.5: Hybrid fusion now handled by Qdrant server-side (RRF)
            # No manual fusion needed

            # Step 5: Apply reranking if enabled
            if self.reranking_enabled and enriched_results:
                enriched_results = await self._rerank(query, enriched_results, top_k)
                logger.info(
                    "reranking_completed",
                    query=query,
                    final_results=len(enriched_results)
                )
            else:
                # No reranking - just take top_k
                enriched_results = enriched_results[:top_k]

            logger.info(
                "retrieval_completed",
                query=query,
                results_found=len(enriched_results),
                avg_score=sum(r['score'] for r in enriched_results) / len(enriched_results) if enriched_results else 0,
                reranked=self.reranking_enabled
            )

            return enriched_results

        except Exception as e:
            logger.error("retrieval_failed", query=query, error=str(e))
            raise

    async def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for query using the configured provider"""
        loop = asyncio.get_event_loop()

        if self.provider_type == "openai":
            # OpenAI embedding (no prefix needed)
            embedding = await loop.run_in_executor(
                None,
                lambda: self._embed_with_openai(query)
            )
        else:
            # SentenceTransformer embedding (with prefix)
            prefixed_query = f"search_query: {query}"
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embedding_model.encode(prefixed_query, normalize_embeddings=True)
            )
            embedding = embedding.tolist()

        return embedding

    def _embed_with_openai(self, text: str) -> List[float]:
        """Embed text using OpenAI API"""
        response = self.openai_client.embeddings.create(
            input=[text],
            model=self.embedding_model_name
        )

        embedding = response.data[0].embedding

        # Normalize for cosine similarity
        import numpy as np
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm

        return embedding.tolist()

    async def _rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        pairs = [[query, doc['text']] for doc in documents]

        loop = asyncio.get_event_loop()
        raw_scores = await loop.run_in_executor(
            None,
            lambda: self.reranker.predict(pairs)  # logits from cross-encoder
        )

        scores = torch.sigmoid(torch.tensor(raw_scores)).tolist()

        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            doc['original_score'] = doc['score']
            doc['score'] = float(score)

        reranked_docs = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )[:top_k]

        logger.info(
            "documents_reranked",
            input_count=len(documents),
            output_count=len(reranked_docs),
            top_score=reranked_docs[0]['rerank_score'] if reranked_docs else 0
        )

        return reranked_docs

    async def _enrich_result(self, result) -> Dict[str, Any]:
        """Enrich search result with additional metadata from PostgreSQL"""
        try:
            chunk_id = result.payload.get('chunk_id')

            # Get additional chunk data from PostgreSQL
            conn = psycopg2.connect(POSTGRES_DSN)
            cur = conn.cursor()

            cur.execute("""
                SELECT c.text, c.meta, f.filename, f.mime, f.size_bytes
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE c.chunk_id = %s
            """, (chunk_id,))

            row = cur.fetchone()
            conn.close()

            if not row:
                return None

            full_text, chunk_meta, filename, mime, size_bytes = row

            # Parse chunk metadata
            if isinstance(chunk_meta, str):
                chunk_meta = json.loads(chunk_meta)
            elif chunk_meta is None:
                chunk_meta = {}

            return {
                'score': float(result.score),
                'text': full_text,
                'chunk_id': chunk_id,
                'filename': filename,
                'mime': mime,
                'size_bytes': size_bytes,
                'chunk_index': result.payload.get('index', 0),
                'token_count': chunk_meta.get('token_count', 0),
                'chunk_type': chunk_meta.get('chunk_type', 'unknown'),
                'headers': chunk_meta.get('headers', {}),
                'preview': result.payload.get('preview', ''),
                'embed_model': result.payload.get('embed_model', ''),
            }

        except Exception as e:
            logger.error("result_enrichment_failed", chunk_id=result.payload.get('chunk_id'), error=str(e))
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        try:
            # Get Qdrant collection info
            collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)

            # Get PostgreSQL stats
            conn = psycopg2.connect(POSTGRES_DSN)
            cur = conn.cursor()

            cur.execute("""
                SELECT
                    COUNT(*) as total_files,
                    COUNT(CASE WHEN processed = true THEN 1 END) as processed_files,
                    SUM(size_bytes) as total_size_bytes
                FROM files
            """)
            file_stats = cur.fetchone()

            cur.execute("""
                SELECT
                    COUNT(*) as total_chunks,
                    AVG(CASE WHEN meta->>'token_count' ~ '^[0-9]+$'
                        THEN (meta->>'token_count')::integer
                        ELSE NULL END) as avg_tokens_per_chunk
                FROM chunks
            """)
            chunk_stats = cur.fetchone()

            conn.close()

            return {
                'qdrant': {
                    'collection': QDRANT_COLLECTION,
                    'points_count': collection_info.points_count,
                    'vector_size': collection_info.config.params.vectors.size,
                    'status': collection_info.status.value
                },
                'postgres': {
                    'total_files': file_stats[0] if file_stats else 0,
                    'processed_files': file_stats[1] if file_stats else 0,
                    'total_size_bytes': file_stats[2] if file_stats else 0,
                    'total_chunks': chunk_stats[0] if chunk_stats else 0,
                    'avg_tokens_per_chunk': float(chunk_stats[1]) if chunk_stats and chunk_stats[1] else 0
                },
                'embedding_model': self.embedding_model_name,
                'embedding_provider': EMBEDDING_PROVIDER
            }

        except Exception as e:
            logger.error("stats_collection_failed", error=str(e))
            raise
