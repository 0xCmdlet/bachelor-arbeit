# langchain_chunking.py
"""
Enhanced chunking using LangChain's MarkdownHeaderTextSplitter
Preserves semantic boundaries and document structure from Docling output

Also supports converting pre-chunked Unstructured.io output to MarkdownChunk format.
Also supports Chonkie RecursiveChunker for alternative chunking strategy.
"""
from __future__ import annotations
import os
import re
import uuid
from typing import Iterator, List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import AutoTokenizer

# Chonkie imports (optional - only used if CHUNKING_STRATEGY=chonkie)
try:
    from chonkie import RecursiveChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False

# Semantic chunking imports (optional - only used if CHUNKING_STRATEGY=semantic)
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

# OpenAI embeddings for LangChain (optional)
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False

# Default tokenizer for token counting
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Shared GPT-2 tokenizer for accurate token counting (lazy-loaded)
_gpt2_tokenizer = None

def _get_gpt2_tokenizer():
    """Get or initialize the shared GPT-2 tokenizer for accurate token counting."""
    global _gpt2_tokenizer
    if _gpt2_tokenizer is None:
        from transformers import AutoTokenizer
        _gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _gpt2_tokenizer

def _get_langchain_embeddings(model_name: str, device: str = "cpu"):
    """
    Get LangChain embeddings based on EMBEDDING_PROVIDER environment variable.

    Args:
        model_name: Model name for HuggingFace embeddings
        device: Device to use for HuggingFace embeddings ("cpu" or "cuda")

    Returns:
        LangChain embeddings instance (HuggingFaceEmbeddings or OpenAIEmbeddings)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower()

    if provider == "openai":
        if not OPENAI_EMBEDDINGS_AVAILABLE:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI provider")

        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"Using OpenAI embeddings for semantic chunking: {openai_model}")

        return OpenAIEmbeddings(
            model=openai_model,
            openai_api_key=openai_api_key
        )
    else:
        # Default to HuggingFace
        if not SEMANTIC_CHUNKING_AVAILABLE:
            raise ImportError("langchain-huggingface not installed. Run: pip install langchain-huggingface")

        print(f"Using HuggingFace embeddings for semantic chunking: {model_name} on {device}")

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True, "device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

@dataclass
class MarkdownChunk:
    """Enhanced chunk with metadata from LangChain processing."""
    chunk_id: str
    text: str
    index: int
    start_token: int
    end_token: int
    token_count: int
    metadata: Dict[str, Any]
    chunk_type: str = "markdown"

# ---------- Public API ----------

def chunk_text(text: str, max_tokens: int = 750, overlap: int = 150, model_name: str = DEFAULT_MODEL) -> Iterator[MarkdownChunk]:
    """
    Enhanced chunking that preserves markdown structure using LangChain.

    Args:
        text: Input text (preferably markdown from Docling)
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        model_name: Model for tokenization

    Yields:
        MarkdownChunk objects with preserved structure and metadata
    """
    if not text or not text.strip():
        return

    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        # Fallback to basic word counting
        tokenizer = None

    # Use LangChain's MarkdownHeaderTextSplitter for semantic boundaries
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers in chunks for context
    )

    # First pass: split by headers to preserve document structure
    try:
        header_sections = markdown_splitter.split_text(text)
    except Exception:
        # Fallback to basic processing if LangChain fails
        header_sections = [{"page_content": text, "metadata": {}}]

    # Second pass: further split large sections by token count
    chunk_index = 0
    current_token_position = 0

    for section in header_sections:
        section_text = getattr(section, 'page_content', '')
        section_metadata = getattr(section, 'metadata', {})

        if not section_text.strip():
            continue

        # Calculate tokens for this section
        section_tokens = _count_tokens(section_text, tokenizer)

        if section_tokens <= max_tokens:
            # Section fits in one chunk
            chunk_id = str(uuid.uuid4())
            yield MarkdownChunk(
                chunk_id=chunk_id,
                text=section_text,
                index=chunk_index,
                start_token=current_token_position,
                end_token=current_token_position + section_tokens,
                token_count=section_tokens,
                metadata=section_metadata,
                chunk_type="header_section"
            )
            chunk_index += 1
            current_token_position += section_tokens
        else:
            # Split large section recursively
            for chunk in _split_large_section(
                section_text,
                section_metadata,
                max_tokens,
                overlap,
                tokenizer,
                chunk_index,
                current_token_position
            ):
                yield chunk
                chunk_index += 1
                current_token_position = chunk.end_token


def _rechunk_oversized_chonkie_chunk(text: str, max_tokens: int = 2000) -> List[str]:
    """
    Re-chunk an oversized Chonkie chunk using a smaller RecursiveChunker.

    Args:
        text: The oversized chunk text
        max_tokens: Maximum tokens allowed per chunk

    Returns:
        List of smaller text chunks that fit within max_tokens
    """
    if not CHONKIE_AVAILABLE:
        # Fallback to simple splitting if Chonkie unavailable
        return [text]

    # Use smaller chunk size to ensure we fit within max_tokens
    # Set to ~half of max_tokens to have safety margin
    smaller_chunk_size = max(100, max_tokens // 2)

    try:
        # Create a new chunker with smaller size
        rechunker = RecursiveChunker(
            tokenizer="gpt2",
            chunk_size=smaller_chunk_size,
            min_characters_per_chunk=24
        )

        # Re-chunk the text
        sub_chunks = rechunker.chunk(text)

        # Extract text from each sub-chunk
        return [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in sub_chunks]

    except Exception as e:
        # If re-chunking fails, return original text (will be caught by validation)
        return [text]


def _rechunk_oversized_semantic_chunk(text: str, max_tokens: int = 2000, tokenizer=None) -> List[str]:
    """
    Re-chunk an oversized semantic chunk by splitting on sentence boundaries.

    Args:
        text: The oversized chunk text
        max_tokens: Maximum tokens allowed per chunk
        tokenizer: Tokenizer for accurate token counting (recommended: GPT-2)

    Returns:
        List of smaller text chunks that fit within max_tokens
    """
    # Split by sentence boundaries
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = _count_tokens(sentence, tokenizer)

        # If single sentence exceeds max_tokens, split by words
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            # Split sentence by words
            words = sentence.split()
            temp_chunk = ""
            temp_tokens = 0

            for word in words:
                word_tokens = _count_tokens(word, tokenizer)

                if temp_tokens + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = word
                    temp_tokens = word_tokens
                else:
                    temp_chunk += (" " + word) if temp_chunk else word
                    temp_tokens += word_tokens

            if temp_chunk:
                chunks.append(temp_chunk.strip())

            continue

        # Try adding sentence to current chunk
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def create_chunks_from_llamaparse_with_chonkie(
    text: str,
    extraction_meta: Dict[str, Any],
    chunk_size: int = 512,
    overlap: int = 128,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 2000
) -> Iterator[MarkdownChunk]:
    """
    Chunk LlamaParse markdown output using Chonkie's RecursiveChunker.

    This function uses Chonkie's markdown recipe which is optimized for markdown structure,
    providing an alternative to LangChain's MarkdownHeaderTextSplitter.

    Args:
        text: Markdown text from LlamaParse
        extraction_meta: Metadata from LlamaParseExtractor
        chunk_size: Maximum tokens per chunk (Chonkie parameter)
        overlap: Token overlap between chunks
        model_name: Model for tokenization (for compatibility)

    Yields:
        MarkdownChunk objects with Chonkie-specific metadata
    """
    if not CHONKIE_AVAILABLE:
        raise ImportError(
            "Chonkie not installed. Install with: pip install chonkie"
        )

    if not text or not text.strip():
        return

    # Initialize Chonkie's RecursiveChunker with markdown recipe
    # Note: OSS version uses from_recipe() method, not constructor parameters
    chunker = RecursiveChunker.from_recipe("markdown", lang="en")

    # Chunk the text with Chonkie (first pass)
    try:
        chonkie_chunks = chunker.chunk(text)
    except Exception as e:
        raise RuntimeError(f"Chonkie chunking failed: {e}")

    # Two-pass chunking: check sizes and re-chunk if needed
    output_index = 0

    for chonkie_chunk in chonkie_chunks:
        # Extract text and token count from Chonkie chunk
        chunk_text = chonkie_chunk.text if hasattr(chonkie_chunk, 'text') else str(chonkie_chunk)
        token_count = chonkie_chunk.token_count if hasattr(chonkie_chunk, 'token_count') else _count_tokens(chunk_text, None)

        # Check if chunk is too large
        if token_count > max_tokens:
            # Second pass: re-chunk oversized chunk
            sub_chunks = _rechunk_oversized_chonkie_chunk(chunk_text, max_tokens)

            for sub_chunk_text in sub_chunks:
                sub_token_count = _count_tokens(sub_chunk_text, None)

                # Build metadata for re-chunked sub-chunk
                metadata = {
                    "chunker": "chonkie",
                    "chunker_type": "RecursiveChunker",
                    "recipe": "markdown",
                    "rechunked": True,  # Mark as re-chunked
                    "original_tokens": token_count,
                    "extractor": extraction_meta.get("extractor", "llamaparse"),
                    "parse_mode": extraction_meta.get("parse_mode", ""),
                    "model": extraction_meta.get("model", ""),
                }

                yield MarkdownChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub_chunk_text,
                    index=output_index,
                    start_token=0,
                    end_token=sub_token_count,
                    token_count=sub_token_count,
                    metadata=metadata,
                    chunk_type="chonkie_rechunked"
                )
                output_index += 1
        else:
            # Chunk is fine, yield as-is
            metadata = {
                "chunker": "chonkie",
                "chunker_type": "RecursiveChunker",
                "recipe": "markdown",
                "rechunked": False,
                "extractor": extraction_meta.get("extractor", "llamaparse"),
                "parse_mode": extraction_meta.get("parse_mode", ""),
                "model": extraction_meta.get("model", ""),
            }

            # Add any additional Chonkie chunk metadata
            if hasattr(chonkie_chunk, 'metadata') and chonkie_chunk.metadata:
                metadata["chonkie_metadata"] = chonkie_chunk.metadata

            yield MarkdownChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                index=output_index,
                start_token=0,
                end_token=token_count,
                token_count=token_count,
                metadata=metadata,
                chunk_type="chonkie_recursive"
            )
            output_index += 1


def create_chunks_from_docling_with_chonkie(
    text: str,
    extraction_meta: Dict[str, Any],
    chunk_size: int = 512,
    overlap: int = 128,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = 2000
) -> Iterator[MarkdownChunk]:
    """
    Chunk Docling markdown output using Chonkie's RecursiveChunker.

    This provides an alternative to LangChain chunking for Docling output,
    using the same two-pass approach as LlamaParse + Chonkie.

    Args:
        text: Markdown text from Docling
        extraction_meta: Metadata from DoclingExtractor
        chunk_size: Maximum tokens per chunk (Chonkie parameter)
        overlap: Token overlap between chunks
        model_name: Model for tokenization (for compatibility)

    Yields:
        MarkdownChunk objects with Chonkie-specific metadata
    """
    if not CHONKIE_AVAILABLE:
        raise ImportError(
            "Chonkie not installed. Install with: pip install chonkie"
        )

    if not text or not text.strip():
        return

    # Initialize Chonkie's RecursiveChunker with markdown recipe
    # Note: OSS version uses from_recipe() method, not constructor parameters
    chunker = RecursiveChunker.from_recipe("markdown", lang="en")

    # Chunk the text with Chonkie (first pass)
    try:
        chonkie_chunks = chunker.chunk(text)
    except Exception as e:
        raise RuntimeError(f"Chonkie chunking failed: {e}")

    # Two-pass chunking: check sizes and re-chunk if needed
    output_index = 0

    for chonkie_chunk in chonkie_chunks:
        # Extract text and token count from Chonkie chunk
        chunk_text = chonkie_chunk.text if hasattr(chonkie_chunk, 'text') else str(chonkie_chunk)
        token_count = chonkie_chunk.token_count if hasattr(chonkie_chunk, 'token_count') else _count_tokens(chunk_text, None)

        # Check if chunk is too large
        if token_count > max_tokens:
            # Second pass: re-chunk oversized chunk
            sub_chunks = _rechunk_oversized_chonkie_chunk(chunk_text, max_tokens)

            for sub_chunk_text in sub_chunks:
                sub_token_count = _count_tokens(sub_chunk_text, None)

                # Build metadata for re-chunked sub-chunk
                metadata = {
                    "chunker": "chonkie",
                    "chunker_type": "RecursiveChunker",
                    "recipe": "markdown",
                    "rechunked": True,  # Mark as re-chunked
                    "original_tokens": token_count,
                    "extractor": extraction_meta.get("extractor", "docling"),
                    "ocr_used": extraction_meta.get("ocr_used", False),
                    "layout_detection": extraction_meta.get("layout_detection", False),
                    "table_extraction": extraction_meta.get("table_extraction", False),
                }

                yield MarkdownChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub_chunk_text,
                    index=output_index,
                    start_token=0,
                    end_token=sub_token_count,
                    token_count=sub_token_count,
                    metadata=metadata,
                    chunk_type="chonkie_rechunked"
                )
                output_index += 1
        else:
            # Chunk is fine, yield as-is
            metadata = {
                "chunker": "chonkie",
                "chunker_type": "RecursiveChunker",
                "recipe": "markdown",
                "rechunked": False,
                "extractor": extraction_meta.get("extractor", "docling"),
                "ocr_used": extraction_meta.get("ocr_used", False),
                "layout_detection": extraction_meta.get("layout_detection", False),
                "table_extraction": extraction_meta.get("table_extraction", False),
            }

            # Add any additional Chonkie chunk metadata
            if hasattr(chonkie_chunk, 'metadata') and chonkie_chunk.metadata:
                metadata["chonkie_metadata"] = chonkie_chunk.metadata

            yield MarkdownChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                index=output_index,
                start_token=0,
                end_token=token_count,
                token_count=token_count,
                metadata=metadata,
                chunk_type="chonkie_recursive"
            )
            output_index += 1


# ---------- Private Helpers ----------

def _split_oversized_paragraph(
    text: str,
    max_tokens: int,
    overlap: int,
    tokenizer
) -> List[str]:
    """
    Split a paragraph that exceeds max_tokens into smaller chunks.
    Uses sentence boundaries first, then fixed-size chunks if needed.

    Args:
        text: The oversized paragraph text
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        tokenizer: Tokenizer for counting tokens

    Returns:
        List of text chunks, each <= max_tokens
    """
    # First, try splitting by sentences
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = _count_tokens(sentence, tokenizer)

        # If a single sentence exceeds max_tokens, split it by fixed size
        if sentence_tokens > max_tokens:
            # Yield current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            # Split sentence by character boundaries
            words = sentence.split()
            temp_chunk = ""
            temp_tokens = 0

            for word in words:
                word_tokens = _count_tokens(word, tokenizer)

                if temp_tokens + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = word
                    temp_tokens = word_tokens
                else:
                    temp_chunk += (" " + word) if temp_chunk else word
                    temp_tokens += word_tokens

            if temp_chunk:
                chunks.append(temp_chunk.strip())

            continue

        # Try adding sentence to current chunk
        if current_tokens + sentence_tokens > max_tokens:
            # Current chunk is full, yield it
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk with this sentence
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _split_large_section(
    text: str,
    base_metadata: Dict[str, Any],
    max_tokens: int,
    overlap: int,
    tokenizer,
    start_index: int,
    start_token_pos: int
) -> Iterator[MarkdownChunk]:
    """Split a large section into smaller token-bounded chunks."""

    # Try to split on paragraph boundaries first
    paragraphs = re.split(r'\n\s*\n', text)

    current_chunk_text = ""
    current_chunk_tokens = 0
    chunk_index = start_index
    token_position = start_token_pos

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = _count_tokens(para, tokenizer)

        # If a single paragraph exceeds max_tokens, split it recursively
        if para_tokens > max_tokens:
            # Yield current chunk if it has content
            if current_chunk_text:
                chunk_id = str(uuid.uuid4())
                yield MarkdownChunk(
                    chunk_id=chunk_id,
                    text=current_chunk_text.strip(),
                    index=chunk_index,
                    start_token=token_position,
                    end_token=token_position + current_chunk_tokens,
                    token_count=current_chunk_tokens,
                    metadata={**base_metadata, "split_type": "paragraph_boundary"},
                    chunk_type="paragraph_split"
                )
                token_position += current_chunk_tokens
                chunk_index += 1
                current_chunk_text = ""
                current_chunk_tokens = 0

            # Split oversized paragraph into smaller chunks
            para_chunks = _split_oversized_paragraph(para, max_tokens, overlap, tokenizer)

            for sub_para in para_chunks:
                sub_tokens = _count_tokens(sub_para, tokenizer)
                chunk_id = str(uuid.uuid4())
                yield MarkdownChunk(
                    chunk_id=chunk_id,
                    text=sub_para,
                    index=chunk_index,
                    start_token=token_position,
                    end_token=token_position + sub_tokens,
                    token_count=sub_tokens,
                    metadata={**base_metadata, "split_type": "oversized_paragraph"},
                    chunk_type="sentence_split"
                )
                token_position += sub_tokens
                chunk_index += 1

            continue

        # If adding this paragraph exceeds max_tokens, yield current chunk
        if current_chunk_tokens + para_tokens > max_tokens and current_chunk_text:
            chunk_id = str(uuid.uuid4())
            yield MarkdownChunk(
                chunk_id=chunk_id,
                text=current_chunk_text.strip(),
                index=chunk_index,
                start_token=token_position,
                end_token=token_position + current_chunk_tokens,
                token_count=current_chunk_tokens,
                metadata={**base_metadata, "split_type": "paragraph_boundary"},
                chunk_type="paragraph_split"
            )

            # Start new chunk with overlap
            overlap_text = _get_overlap_text(current_chunk_text, overlap, tokenizer)
            current_chunk_text = overlap_text + "\n\n" + para if overlap_text else para
            current_chunk_tokens = _count_tokens(current_chunk_text, tokenizer)
            token_position += (current_chunk_tokens - _count_tokens(overlap_text, tokenizer) if overlap_text else current_chunk_tokens)
            chunk_index += 1
        else:
            # Add paragraph to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n" + para
            else:
                current_chunk_text = para
            current_chunk_tokens += para_tokens

    # Yield final chunk if any content remains
    if current_chunk_text.strip():
        chunk_id = str(uuid.uuid4())
        yield MarkdownChunk(
            chunk_id=chunk_id,
            text=current_chunk_text.strip(),
            index=chunk_index,
            start_token=token_position,
            end_token=token_position + current_chunk_tokens,
            token_count=current_chunk_tokens,
            metadata={**base_metadata, "split_type": "final_chunk"},
            chunk_type="paragraph_split"
        )

def _count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using provided tokenizer or word approximation."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    # Fallback: approximate 1 token ≈ 0.75 words
    words = len(text.split())
    return int(words / 0.75)

def _get_overlap_text(text: str, overlap_tokens: int, tokenizer) -> str:
    """Extract the last N tokens worth of text for overlap."""
    if overlap_tokens <= 0:
        return ""

    words = text.split()
    if not words:
        return ""

    # Approximate words needed for overlap_tokens
    overlap_words = int(overlap_tokens * 0.75)
    overlap_words = min(overlap_words, len(words))

    if overlap_words <= 0:
        return ""

    return " ".join(words[-overlap_words:])

# ---------- Utility Functions ----------

def get_chunk_statistics(chunks: List[MarkdownChunk]) -> Dict[str, Any]:
    """Get statistics about the chunking results."""
    if not chunks:
        return {"total_chunks": 0}

    token_counts = [chunk.token_count for chunk in chunks]
    chunks_with_headers = sum(1 for chunk in chunks if chunk.metadata and
                            any(k.startswith("Header") for k in chunk.metadata.keys()))

    return {
        "total_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "chunks_with_headers": chunks_with_headers,
        "header_preservation_rate": chunks_with_headers / len(chunks),
        "chunk_types": {chunk.chunk_type: sum(1 for c in chunks if c.chunk_type == chunk.chunk_type)
                       for chunk in chunks}
    }

def validate_chunking(text: str, chunks: List[MarkdownChunk]) -> Dict[str, Any]:
    """Validate that chunking preserved all content."""
    original_length = len(text)
    reconstructed_text = "\n\n".join(chunk.text for chunk in chunks)
    reconstructed_length = len(reconstructed_text)

    # Check for significant content loss (allowing for some formatting differences)
    content_preservation = reconstructed_length / original_length if original_length > 0 else 0

    return {
        "original_length": original_length,
        "reconstructed_length": reconstructed_length,
        "content_preservation_ratio": content_preservation,
        "chunks_have_ids": all(chunk.chunk_id for chunk in chunks),
        "chunks_ordered": all(chunks[i].index < chunks[i+1].index for i in range(len(chunks)-1)),
        "token_positions_valid": all(chunk.start_token < chunk.end_token for chunk in chunks)
    }

# ---------- Testing and Debugging ----------

def test_chunking(text: str, max_tokens: int = 750, overlap: int = 150) -> Dict[str, Any]:
    """Test chunking on a sample text and return detailed results."""
    try:
        chunks = list(chunk_text(text, max_tokens=max_tokens, overlap=overlap))
        stats = get_chunk_statistics(chunks)
        validation = validate_chunking(text, chunks)

        return {
            "success": True,
            "chunks": len(chunks),
            "statistics": stats,
            "validation": validation,
            "sample_chunks": [
                {
                    "index": chunk.index,
                    "tokens": chunk.token_count,
                    "metadata": chunk.metadata,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                }
                for chunk in chunks[:3]  # Show first 3 chunks
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


# ============================================================================
# Semantic Chunking (LangChain Experimental)
# ============================================================================

def create_chunks_from_docling_with_semantic(
    text: str,
    extraction_meta: Dict[str, Any],
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    breakpoint_threshold_type: str = "percentile",
    max_tokens: int = 2000
) -> Iterator[MarkdownChunk]:
    """
    Chunk Docling markdown output using LangChain's SemanticChunker.

    This uses embeddings-based chunking that splits text based on semantic similarity
    rather than structural markers (headers) or fixed token counts.

    Args:
        text: Markdown text from Docling extraction
        extraction_meta: Metadata from extraction (OCR, layout detection, etc.)
        model_name: HuggingFace model for embeddings during chunking
        breakpoint_threshold_type: How to determine split points
            - "percentile" (default): split at top X percentile of similarity drops
            - "standard_deviation": split when similarity drops by X std devs
            - "interquartile": split based on interquartile range

    Yields:
        MarkdownChunk objects with semantic boundaries
    """
    if not SEMANTIC_CHUNKING_AVAILABLE:
        raise ImportError(
            "Semantic chunking not available. Install with: "
            "pip install langchain-experimental langchain-huggingface"
        )

    if not text or not text.strip():
        return

    # Initialize GPT-2 tokenizer for accurate token counting
    tokenizer = _get_gpt2_tokenizer()

    # Get embeddings provider for semantic chunking
    # Configurable device based on VRAM availability
    semantic_device = os.getenv("SEMANTIC_CHUNKING_DEVICE", "auto").lower()
    if semantic_device == "auto":
        device = "cpu"  # Safe default for Docling+Semantic on 8GB VRAM
    elif semantic_device == "cuda":
        device = "cuda"  # Use GPU for 24GB VRAM
    elif semantic_device == "cpu":
        device = "cpu"
    else:
        device = "cpu"  # Fallback to CPU

    embeddings = _get_langchain_embeddings(model_name, device=device)

    # Create semantic chunker
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type
    )

    # Split text semantically
    try:
        langchain_docs = text_splitter.create_documents([text])
    except Exception as e:
        raise RuntimeError(f"Semantic chunking failed: {e}")

    # Two-pass chunking: check sizes and re-chunk if needed
    output_index = 0

    for doc in langchain_docs:
        chunk_text = doc.page_content
        token_count = _count_tokens(chunk_text, tokenizer)

        # Check if chunk is too large
        if token_count > max_tokens:
            # Second pass: re-chunk oversized chunk
            sub_chunks = _rechunk_oversized_semantic_chunk(chunk_text, max_tokens, tokenizer)

            for sub_chunk_text in sub_chunks:
                sub_token_count = _count_tokens(sub_chunk_text, tokenizer)

                metadata = {
                    "chunker": "semantic",
                    "chunker_type": "SemanticChunker",
                    "breakpoint_threshold_type": breakpoint_threshold_type,
                    "embedding_model": model_name,
                    "rechunked": True,
                    "original_tokens": token_count,
                    "extractor": extraction_meta.get("extractor", "docling"),
                    "ocr_used": extraction_meta.get("ocr_used", False),
                    "layout_detection": extraction_meta.get("layout_detection", False),
                    "table_extraction": extraction_meta.get("table_extraction", False),
                }

                yield MarkdownChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub_chunk_text,
                    index=output_index,
                    start_token=0,
                    end_token=sub_token_count,
                    token_count=sub_token_count,
                    metadata=metadata,
                    chunk_type="semantic_rechunked"
                )
                output_index += 1
        else:
            # Chunk is fine, yield as-is
            metadata = {
                "chunker": "semantic",
                "chunker_type": "SemanticChunker",
                "breakpoint_threshold_type": breakpoint_threshold_type,
                "embedding_model": model_name,
                "rechunked": False,
                "extractor": extraction_meta.get("extractor", "docling"),
                "ocr_used": extraction_meta.get("ocr_used", False),
                "layout_detection": extraction_meta.get("layout_detection", False),
                "table_extraction": extraction_meta.get("table_extraction", False),
            }

            # Include any metadata from the LangChain document
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata["langchain_metadata"] = doc.metadata

            yield MarkdownChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                index=output_index,
                start_token=0,
                end_token=token_count,
                token_count=token_count,
                metadata=metadata,
                chunk_type="semantic"
            )
            output_index += 1


def create_chunks_from_llamaparse_with_semantic(
    text: str,
    extraction_meta: Dict[str, Any],
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    breakpoint_threshold_type: str = "percentile",
    max_tokens: int = 2000
) -> Iterator[MarkdownChunk]:
    """
    Chunk LlamaParse markdown output using LangChain's SemanticChunker.

    This uses embeddings-based chunking that splits text based on semantic similarity
    rather than structural markers (headers) or fixed token counts.

    Args:
        text: Markdown text from LlamaParse extraction
        extraction_meta: Metadata from extraction
        model_name: HuggingFace model for embeddings during chunking
        breakpoint_threshold_type: How to determine split points

    Yields:
        MarkdownChunk objects with semantic boundaries
    """
    if not SEMANTIC_CHUNKING_AVAILABLE:
        raise ImportError(
            "Semantic chunking not available. Install with: "
            "pip install langchain-experimental langchain-huggingface"
        )

    if not text or not text.strip():
        return

    # Initialize GPT-2 tokenizer for accurate token counting
    tokenizer = _get_gpt2_tokenizer()

    # Get embeddings provider for semantic chunking
    # Can use GPU for LlamaParse since it doesn't use GPU for extraction
    embeddings = _get_langchain_embeddings(model_name, device="cuda")

    # Create semantic chunker
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type
    )

    # Split text semantically
    try:
        langchain_docs = text_splitter.create_documents([text])
    except Exception as e:
        raise RuntimeError(f"Semantic chunking failed: {e}")

    # Two-pass chunking: check sizes and re-chunk if needed
    output_index = 0

    for doc in langchain_docs:
        chunk_text = doc.page_content
        token_count = _count_tokens(chunk_text, tokenizer)

        # Check if chunk is too large
        if token_count > max_tokens:
            # Second pass: re-chunk oversized chunk
            sub_chunks = _rechunk_oversized_semantic_chunk(chunk_text, max_tokens, tokenizer)

            for sub_chunk_text in sub_chunks:
                sub_token_count = _count_tokens(sub_chunk_text, tokenizer)

                metadata = {
                    "chunker": "semantic",
                    "chunker_type": "SemanticChunker",
                    "breakpoint_threshold_type": breakpoint_threshold_type,
                    "embedding_model": model_name,
                    "rechunked": True,
                    "original_tokens": token_count,
                    "extractor": extraction_meta.get("extractor", "llamaparse"),
                    "parse_mode": extraction_meta.get("parse_mode", ""),
                    "model": extraction_meta.get("model", ""),
                }

                yield MarkdownChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub_chunk_text,
                    index=output_index,
                    start_token=0,
                    end_token=sub_token_count,
                    token_count=sub_token_count,
                    metadata=metadata,
                    chunk_type="semantic_rechunked"
                )
                output_index += 1
        else:
            # Chunk is fine, yield as-is
            metadata = {
                "chunker": "semantic",
                "chunker_type": "SemanticChunker",
                "breakpoint_threshold_type": breakpoint_threshold_type,
                "embedding_model": model_name,
                "rechunked": False,
                "extractor": extraction_meta.get("extractor", "llamaparse"),
                "parse_mode": extraction_meta.get("parse_mode", ""),
                "model": extraction_meta.get("model", ""),
            }

            # Include any metadata from the LangChain document
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata["langchain_metadata"] = doc.metadata

            yield MarkdownChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                index=output_index,
                start_token=0,
                end_token=token_count,
                token_count=token_count,
                metadata=metadata,
                chunk_type="semantic"
            )
            output_index += 1