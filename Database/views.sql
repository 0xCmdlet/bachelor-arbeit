
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

