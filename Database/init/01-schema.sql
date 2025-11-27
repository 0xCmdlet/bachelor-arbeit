-- ============================================================================
-- PostgreSQL Schema Initialization for RAG Document Processing Pipeline
-- Auto-run by Docker on first container start
-- ============================================================================

-- ============================================================================
-- TABLES
-- ============================================================================

-- Table for files (one per object in MinIO)
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,       -- "notes.txt" or "bucket/key"
    mime TEXT,                           -- detected MIME type
    size_bytes BIGINT,
    etag TEXT,                           -- optional MinIO object version/hash
    meta JSONB DEFAULT '{}'::jsonb,      -- enhanced: extraction metadata (ocr, layout, tables, formulas)
    processed BOOLEAN DEFAULT FALSE,     -- ingestion status
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enhanced table for text chunks with LangChain metadata
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,           -- stable hash-based id
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER,                 -- order within file
    start_token INTEGER,                 -- where in token stream this chunk starts
    end_token INTEGER,                   -- where it ends
    collection TEXT,                     -- Qdrant collection name
    text TEXT,                           -- the actual chunk text
    meta JSONB DEFAULT '{}'::jsonb,      -- NEW: LangChain metadata (headers, chunk_type, etc.)
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT valid_token_range CHECK (start_token IS NULL OR end_token IS NULL OR start_token <= end_token)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Basic indexes
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_files_processed ON files(processed);

-- Enhanced indexes for LangChain metadata (GIN for JSONB)
CREATE INDEX IF NOT EXISTS idx_chunks_meta ON chunks USING GIN(meta);
CREATE INDEX IF NOT EXISTS idx_files_meta ON files USING GIN(meta);

-- Index for chunk types (for efficient filtering)
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks((meta->>'chunk_type'));

-- Index for chunks with headers (for section-based queries)
CREATE INDEX IF NOT EXISTS idx_chunks_headers ON chunks((meta->'headers')) WHERE meta->'headers' IS NOT NULL;

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_file_index ON chunks(file_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Chunks with their header hierarchy
CREATE OR REPLACE VIEW chunk_sections AS
SELECT
    c.chunk_id,
    c.file_id,
    c.chunk_index,
    c.text,
    f.filename,
    c.meta->>'chunk_type' as chunk_type,
    c.meta->'headers' as headers,
    CASE
        WHEN c.meta->'headers'->>'Header 1' IS NOT NULL THEN c.meta->'headers'->>'Header 1'
        ELSE 'No Section'
    END as main_section,
    CASE
        WHEN c.meta->'headers'->>'Header 2' IS NOT NULL THEN c.meta->'headers'->>'Header 2'
        ELSE NULL
    END as subsection,
    c.created_at
FROM chunks c
JOIN files f ON c.file_id = f.id;

-- View: File processing summary with enhanced metadata
CREATE OR REPLACE VIEW file_processing_summary AS
SELECT
    f.id,
    f.filename,
    f.mime,
    f.size_bytes,
    f.processed,
    COALESCE((f.meta->'ocr_used')::boolean, false) as ocr_used,
    COALESCE((f.meta->'layout_detection')::boolean, false) as layout_detection,
    COALESCE((f.meta->'table_extraction')::boolean, false) as table_extraction,
    COALESCE((f.meta->'formula_detection')::boolean, false) as formula_detection,
    COUNT(c.chunk_id) as total_chunks,
    COUNT(CASE WHEN c.meta->'headers' IS NOT NULL THEN 1 END) as chunks_with_headers,
    COUNT(CASE WHEN c.meta->>'chunk_type' = 'table' THEN 1 END) as table_chunks,
    COUNT(CASE WHEN c.meta->>'chunk_type' = 'code' THEN 1 END) as code_chunks,
    AVG(CASE WHEN c.meta->>'token_count' ~ '^[0-9]+$' THEN (c.meta->>'token_count')::integer ELSE NULL END) as avg_tokens_per_chunk,
    f.created_at,
    f.updated_at
FROM files f
LEFT JOIN chunks c ON f.id = c.file_id
GROUP BY f.id, f.filename, f.mime, f.size_bytes, f.processed, f.meta, f.created_at, f.updated_at;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Get chunks by section hierarchy
CREATE OR REPLACE FUNCTION get_chunks_by_section(
    section_name TEXT,
    file_pattern TEXT DEFAULT '%'
)
RETURNS TABLE(
    chunk_id TEXT,
    filename TEXT,
    chunk_index INTEGER,
    text TEXT,
    full_hierarchy TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_id,
        f.filename,
        c.chunk_index,
        c.text,
        CONCAT_WS(' > ',
            c.meta->'headers'->>'Header 1',
            c.meta->'headers'->>'Header 2',
            c.meta->'headers'->>'Header 3',
            c.meta->'headers'->>'Header 4',
            c.meta->'headers'->>'Header 5',
            c.meta->'headers'->>'Header 6'
        ) as full_hierarchy
    FROM chunks c
    JOIN files f ON c.file_id = f.id
    WHERE f.filename ILIKE file_pattern
    AND (
        c.meta->'headers'->>'Header 1' ILIKE '%' || section_name || '%' OR
        c.meta->'headers'->>'Header 2' ILIKE '%' || section_name || '%' OR
        c.meta->'headers'->>'Header 3' ILIKE '%' || section_name || '%' OR
        c.meta->'headers'->>'Header 4' ILIKE '%' || section_name || '%' OR
        c.meta->'headers'->>'Header 5' ILIKE '%' || section_name || '%' OR
        c.meta->'headers'->>'Header 6' ILIKE '%' || section_name || '%'
    )
    ORDER BY f.filename, c.chunk_index;
END;
$$ LANGUAGE plpgsql;

-- Function: Get processing statistics
CREATE OR REPLACE FUNCTION get_processing_stats()
RETURNS TABLE(
    metric TEXT,
    value TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'Total Files'::TEXT, COUNT(*)::TEXT FROM files
    UNION ALL
    SELECT 'Processed Files'::TEXT, COUNT(*)::TEXT FROM files WHERE processed = true
    UNION ALL
    SELECT 'Total Chunks'::TEXT, COUNT(*)::TEXT FROM chunks
    UNION ALL
    SELECT 'Chunks with Headers'::TEXT, COUNT(*)::TEXT FROM chunks WHERE meta->'headers' IS NOT NULL
    UNION ALL
    SELECT 'Files with OCR'::TEXT, COUNT(*)::TEXT FROM files WHERE COALESCE((meta->'ocr_used')::boolean, false) = true
    UNION ALL
    SELECT 'Files with Tables'::TEXT, COUNT(*)::TEXT FROM files WHERE COALESCE((meta->'table_extraction')::boolean, false) = true
    UNION ALL
    SELECT 'Files with Formulas'::TEXT, COUNT(*)::TEXT FROM files WHERE COALESCE((meta->'formula_detection')::boolean, false) = true
    UNION ALL
    SELECT 'Avg Tokens per Chunk'::TEXT, ROUND(AVG(CASE WHEN meta->>'token_count' ~ '^[0-9]+$' THEN (meta->>'token_count')::integer ELSE NULL END), 2)::TEXT FROM chunks WHERE meta->>'token_count' IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- LANGCHAIN CHECKPOINT TABLES (for conversational memory)
-- ============================================================================

-- LangGraph checkpoint storage
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Checkpoint writes (for async operations)
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Indexes for checkpoints
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id);

-- ============================================================================
-- CONVERSATION METADATA (optional, for UI/listing)
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversations (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);

-- ============================================================================
-- INITIALIZATION COMPLETE
-- ============================================================================
