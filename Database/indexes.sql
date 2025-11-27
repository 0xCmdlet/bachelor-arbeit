
-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_files_processed ON files(processed);

-- NEW: Enhanced indexes for LangChain metadata
CREATE INDEX IF NOT EXISTS idx_chunks_meta ON chunks USING GIN(meta);
CREATE INDEX IF NOT EXISTS idx_files_meta ON files USING GIN(meta);

-- NEW: Index for chunk types (for efficient filtering)
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks((meta->>'chunk_type'));

-- NEW: Index for chunks with headers (for section-based queries)
CREATE INDEX IF NOT EXISTS idx_chunks_headers ON chunks((meta->'headers')) WHERE meta->'headers' IS NOT NULL;
