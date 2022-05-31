MERGE INTO `{project_name}.{dataset}.{table}` t
USING UNNEST(
    [struct<call_convrstn_id STRING, chunk_id INTEGER, text STRING>
     {new_values}]
) s
ON t.call_convrstn_id = s.call_convrstn_id and t.chunk_id = s.chunk_id
WHEN MATCHED THEN
    UPDATE SET t.chunk_id = s.chunk_id, t.text = s.text
WHEN NOT MATCHED THEN
    INSERT (call_convrstn_id, chunk_id, text) 
        VALUES(s.call_convrstn_id, s.chunk_id, s.text)