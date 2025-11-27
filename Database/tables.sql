-- Enhanced schema for Docling + LangChain chunking metadata

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
    created_at TIMESTAMPTZ DEFAULT now()
);
