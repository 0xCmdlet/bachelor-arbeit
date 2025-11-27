# docling_extractors.py
from __future__ import annotations
import tempfile
import os
import uuid
from typing import Tuple, Dict, Optional, List
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

import mimetypes
import json

from chunking import MarkdownChunk

# Unstructured imports (optional - only used if EXTRACTOR_TYPE=unstructured)
try:
    import unstructured_client
    from unstructured_client.models import shared, errors
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# LlamaParse imports (optional - only used if EXTRACTOR_TYPE=llamaparse)
try:
    from llama_cloud_services import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

# Debug: Check if tesserocr can be imported
try:
    import tesserocr
    print(f"[DEBUG] tesserocr successfully imported! Version: {tesserocr.tesseract_version()}")
except ImportError as e:
    print(f"[DEBUG] tesserocr import failed: {e}")
except Exception as e:
    print(f"[DEBUG] tesserocr import error: {type(e).__name__}: {e}")


# -----------------------------
# Public API - Compatible with extractors.py
# -----------------------------

def detect_mime(filename: str, raw_bytes: bytes) -> str:
    """MIME detection using filename extension (same interface as original)."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def extract_text(filename: str, raw_bytes: bytes, mime: Optional[str] = None) -> Tuple[str, Dict]:
    """Extract text using Docling.

    Returns (text, metadata) tuple compatible with the original extractor interface.
    """
    mime = mime or detect_mime(filename, raw_bytes)
    meta: Dict[str, object] = {
        "filename": filename,
        "mime": mime,
        "ocr_used": False,
        "pages": None,
        "warnings": [],
        "layout_detection": False,
        "table_extraction": False,
        "formula_detection": False,
    }

    try:
        # Create temporary file for Docling processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=_get_file_extension(filename)) as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_file.flush()
            tmp_path = tmp_file.name

        try:
            # Configure Docling converter with advanced options
            converter = _create_docling_converter(mime)

            # Convert document
            result = converter.convert(tmp_path)

            # Extract text and metadata
            text = result.document.export_to_markdown()

            # Enhance metadata with Docling-specific information
            _update_metadata_from_docling(meta, result)

            return text, _finalize_meta(meta, text)

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        meta["warnings"].append(f"Docling extraction failed: {type(e).__name__}: {e}")
        return "", _finalize_meta(meta, "")


# -----------------------------
# Unstructured.io Extractor
# -----------------------------

class UnstructuredExtractor:
    """Extract and chunk text using Unstructured.io API (partition + chunking in single call)."""

    def __init__(self, api_key: Optional[str] = None):
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("unstructured-client not installed. Install with: pip install unstructured-client")

        self.api_key = api_key or os.getenv("UNSTRUCTURED_API_KEY")
        if not self.api_key:
            raise ValueError("UNSTRUCTURED_API_KEY environment variable not set")

        self.client = unstructured_client.UnstructuredClient(api_key_auth=self.api_key)

        # Configuration from environment
        self.partition_strategy = os.getenv("UNSTRUCTURED_PARTITION_STRATEGY", "auto")
        self.chunking_strategy = os.getenv("UNSTRUCTURED_CHUNKING_STRATEGY", "by_title")
        self.max_characters = int(os.getenv("UNSTRUCTURED_MAX_CHARACTERS", "1200"))
        self.overlap = int(os.getenv("UNSTRUCTURED_OVERLAP", "150"))

    def extract(self, filename: str, raw_bytes: bytes, mime: Optional[str] = None) -> Tuple[List[MarkdownChunk], Dict]:
        """Extract chunks using Unstructured API, returning MarkdownChunk objects.

        Returns (chunks, metadata) where:
        - chunks: List of MarkdownChunk objects converted from Unstructured elements
        - metadata: Unstructured-specific metadata
        """
        mime = mime or detect_mime(filename, raw_bytes)
        meta: Dict[str, object] = {
            "filename": filename,
            "mime": mime,
            "extractor": "unstructured",
            "partition_strategy": self.partition_strategy,
            "chunking_strategy": self.chunking_strategy,
            "max_characters": self.max_characters,
            "overlap": self.overlap,
            "warnings": [],
        }

        try:
            # Create temporary file for API upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=_get_file_extension(filename)) as tmp_file:
                tmp_file.write(raw_bytes)
                tmp_file.flush()
                tmp_path = tmp_file.name

            try:
                # Call Unstructured API with partition + chunking
                with open(tmp_path, "rb") as f:
                    request = {
                        "partition_parameters": {
                            "files": {
                                "content": f,
                                "file_name": os.path.basename(filename),
                            },
                            "strategy": self._get_partition_strategy_enum(),
                            "languages": ["eng"],
                            "split_pdf_page": True,
                            "split_pdf_allow_failed": True,
                            "split_pdf_concurrency_level": 15,

                            # Chunking parameters (strings, not enums)
                            "chunking_strategy": self.chunking_strategy,
                            "max_characters": self.max_characters,
                            "overlap": self.overlap,
                        }
                    }

                    result = self.client.general.partition(request=request)

                # Process elements (chunks)
                elements = result.elements
                if not elements:
                    meta["warnings"].append("No elements returned from Unstructured API")
                    return [], _finalize_meta(meta, "")

                # Convert elements to dict if needed
                if hasattr(elements[0], "to_dict"):
                    elements = [e.to_dict() for e in elements]

                # Convert elements to MarkdownChunk objects
                chunks = []
                for idx, element in enumerate(elements):
                    chunk_text = element.get("text", "")
                    if not chunk_text.strip():
                        continue  # Skip empty chunks

                    # Simple token count (word split as fallback)
                    token_count = len(chunk_text.split())

                    chunk = MarkdownChunk(
                        chunk_id=str(uuid.uuid4()).replace("-", ""),  # 32-char hex
                        text=chunk_text,  # EXACT SAME TEXT - no modification
                        index=idx,
                        start_token=0,  # Placeholder - Unstructured doesn't provide global positions
                        end_token=token_count,
                        token_count=token_count,
                        metadata={
                            "element_type": element.get("type", "Unknown"),
                            "element_id": element.get("element_id", ""),
                            "unstructured_metadata": element.get("metadata", {})
                        },
                        chunk_type=f"unstructured_{element.get('type', 'unknown').lower()}"
                    )
                    chunks.append(chunk)

                # Add chunks info to metadata
                meta["chunks_count"] = len(chunks)
                meta["element_types"] = list(set(e.get("type", "Unknown") for e in elements))

                # Return MarkdownChunk objects
                return chunks, _finalize_meta(meta, "")

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except errors.SDKError as e:
            meta["warnings"].append(f"Unstructured API error: {e.message if hasattr(e, 'message') else str(e)}")
            return [], _finalize_meta(meta, "")
        except Exception as e:
            meta["warnings"].append(f"Unstructured extraction failed: {type(e).__name__}: {e}")
            return [], _finalize_meta(meta, "")

    def _get_partition_strategy_enum(self) -> shared.Strategy:
        """Convert string strategy to Unstructured SDK enum."""
        strategy_map = {
            "auto": shared.Strategy.AUTO,
            "fast": shared.Strategy.FAST,
            "hi_res": shared.Strategy.HI_RES,
            "ocr_only": shared.Strategy.OCR_ONLY,
        }
        return strategy_map.get(self.partition_strategy.lower(), shared.Strategy.AUTO)


# -----------------------------
# LlamaParse Extractor
# -----------------------------

class LlamaParseExtractor:
    """Extract text using LlamaParse API with Agentic mode."""

    def __init__(self, api_key: Optional[str] = None):
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError("llama-cloud-services not installed. Install with: pip install llama-cloud-services")

        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable not set")

        # Configuration - Agentic mode with recommended settings
        self.parse_mode = os.getenv("LLAMAPARSE_PARSE_MODE", "parse_page_with_agent")
        self.model = os.getenv("LLAMAPARSE_MODEL", "openai-gpt-4-1-mini")

        # Initialize LlamaParse with Agentic configuration
        self.parser = LlamaParse(
            api_key=self.api_key,
            parse_mode=self.parse_mode,
            model=self.model,
            high_res_ocr=True,  # Better for technical diagrams
            adaptive_long_table=True,  # Important for config tables
            outlined_table_extraction=True,  # Extract table structures
            output_tables_as_HTML=True,  # Better for embedding
            verbose=True,
        )

    def extract(self, filename: str, raw_bytes: bytes, mime: Optional[str] = None) -> Tuple[str, Dict]:
        """Extract text using LlamaParse API.

        Returns (markdown_text, metadata) where:
        - markdown_text: Extracted markdown with HTML tables and Mermaid diagrams
        - metadata: Contains LlamaParse-specific metadata
        """
        mime = mime or detect_mime(filename, raw_bytes)
        meta: Dict[str, object] = {
            "filename": filename,
            "mime": mime,
            "extractor": "llamaparse",
            "parse_mode": self.parse_mode,
            "model": self.model,
            "warnings": [],
        }

        try:
            # LlamaParse requires file path, create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=_get_file_extension(filename)) as tmp_file:
                tmp_file.write(raw_bytes)
                tmp_file.flush()
                tmp_path = tmp_file.name

            try:
                # Parse with LlamaParse (synchronous)
                result = self.parser.parse(tmp_path)

                # Get markdown documents
                markdown_documents = result.get_markdown_documents(split_by_page=False)

                if not markdown_documents:
                    meta["warnings"].append("No markdown documents returned from LlamaParse")
                    return "", _finalize_meta(meta, "")

                # Combine all markdown documents into single text
                # (usually just one document for split_by_page=False)
                markdown_text = "\n\n".join([doc.text for doc in markdown_documents])

                # Extract metadata from result
                if hasattr(result, 'pages') and result.pages:
                    meta["pages"] = len(result.pages)

                    # Check for images in pages
                    image_count = sum(len(page.images) for page in result.pages if hasattr(page, 'images') and page.images)
                    if image_count > 0:
                        meta["images_found"] = image_count

                # Add document-level metadata
                meta["ocr_used"] = True  # LlamaParse always uses OCR/vision
                meta["layout_detection"] = True
                meta["table_extraction"] = True  # Enabled in config

                return markdown_text, _finalize_meta(meta, markdown_text)

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            meta["warnings"].append(f"LlamaParse extraction failed: {type(e).__name__}: {e}")
            return "", _finalize_meta(meta, "")


# -----------------------------
# Factory Pattern - Switch between extractors
# -----------------------------

def extract_text_with_strategy(filename: str, raw_bytes: bytes, mime: Optional[str] = None) -> Tuple[str, Dict]:
    """Factory function to extract text using configured strategy.

    Strategy is selected via EXTRACTOR_TYPE environment variable:
    - "docling" (default): Use Docling extractor
    - "unstructured": Use Unstructured.io API extractor
    - "llamaparse": Use LlamaParse API extractor

    Returns (text, metadata) tuple.
    """
    extractor_type = os.getenv("EXTRACTOR_TYPE", "docling").lower()

    if extractor_type == "unstructured":
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "Unstructured extractor requested but unstructured-client not installed. "
                "Install with: pip install unstructured-client"
            )

        extractor = UnstructuredExtractor()
        return extractor.extract(filename, raw_bytes, mime)

    elif extractor_type == "llamaparse":
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError(
                "LlamaParse extractor requested but llama-cloud-services not installed. "
                "Install with: pip install llama-cloud-services"
            )

        extractor = LlamaParseExtractor()
        return extractor.extract(filename, raw_bytes, mime)

    elif extractor_type == "docling":
        return extract_text(filename, raw_bytes, mime)

    else:
        raise ValueError(
            f"Unknown EXTRACTOR_TYPE: {extractor_type}. "
            f"Valid options: 'docling', 'unstructured', 'llamaparse'"
        )


# -----------------------------
# Docling-specific helpers
# -----------------------------

def _create_docling_converter(mime: str) -> DocumentConverter:
    """Create a DocumentConverter with optimized settings based on MIME type."""

    # Configure PDF pipeline with advanced features
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = True  # Enable OCR for scanned documents
    pdf_options.do_table_structure = True  # Extract table structure
    pdf_options.table_structure_options.do_cell_matching = True

    # Create converter with format-specific options
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
    }

    converter = DocumentConverter(
        format_options=format_options,
        # Enable parallel processing if available
        # num_workers=2  # Uncomment if you want parallel processing
    )

    return converter


def _get_file_extension(filename: str) -> str:
    """Get file extension, defaulting to .bin if none found."""
    ext = Path(filename).suffix
    return ext if ext else '.bin'


def _update_metadata_from_docling(meta: Dict, result) -> None:
    """Update metadata dictionary with information from Docling result."""
    try:
        doc = result.document

        # Basic document info
        if hasattr(doc, 'page_count') and doc.page_count:
            meta["pages"] = doc.page_count

        # Check for advanced features used
        meta["layout_detection"] = True  # Docling always does layout detection

        # Check if tables were found
        if hasattr(doc, 'tables') and doc.tables:
            meta["table_extraction"] = True
            meta["tables_found"] = len(doc.tables)

        # Check if images were processed
        if hasattr(doc, 'pictures') and doc.pictures:
            meta["images_found"] = len(doc.pictures)

        # Check if formulas were detected
        if hasattr(doc, 'equations') and doc.equations:
            meta["formula_detection"] = True
            meta["formulas_found"] = len(doc.equations)

        # OCR information (for PDFs and images)
        # Note: Docling handles OCR internally, we assume it was used for scanned content
        if meta["mime"] in ["application/pdf", "image/png", "image/jpeg", "image/tiff"]:
            meta["ocr_used"] = True

    except Exception as e:
        meta["warnings"].append(f"Failed to extract Docling metadata: {e}")


def _finalize_meta(meta: Dict, text: str) -> Dict:
    """Finalize metadata with text statistics."""
    meta = dict(meta)
    meta["characters"] = len(text)
    meta["empty"] = (len(text.strip()) == 0)
    return meta


# -----------------------------
# Utility functions for testing and validation
# -----------------------------

def get_supported_formats() -> Dict[str, list]:
    """Get information about supported formats."""
    return {
        "extensions": [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
                      ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
                      ".wav", ".mp3", ".vtt"],
        "mime_types": ["application/pdf",
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                      "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                      "text/html", "image/png", "image/jpeg", "image/tiff",
                      "image/bmp", "image/gif", "audio/wav", "audio/mpeg", "text/vtt"]
    }


def test_docling_extraction(test_file_path: str) -> Dict:
    """Test Docling extraction on a specific file (for debugging)."""
    try:
        with open(test_file_path, 'rb') as f:
            raw_bytes = f.read()

        filename = os.path.basename(test_file_path)
        text, meta = extract_text(filename, raw_bytes)

        return {
            "success": True,
            "text_length": len(text),
            "metadata": meta,
            "text_preview": text[:500] if text else "No text extracted"
        }

    except Exception as e:
        return {"error": str(e)}