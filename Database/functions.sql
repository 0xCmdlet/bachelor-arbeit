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