MERGE INTO `{project_name}.{dataset}.mover_detection` t
USING UNNEST(
    [struct<convrstn_id STRING, call_date DATE, gcp_refresh_date DATE, 
     ban STRING, src_id INTEGER, telephone STRING, topic STRING, 
     detection_score FLOAT64>
     {new_values}]
) s
ON t.convrstn_id = s.convrstn_id
WHEN MATCHED THEN
    UPDATE SET t.detection_score = s.detection_score, t.call_date = s.call_date, 
        t.gcp_refresh_date = s.gcp_refresh_date, t.ban = s.ban, 
        t.src_id = s.src_id, t.telephone = s.telephone, 
        t.topic = s.topic
WHEN NOT MATCHED THEN
    INSERT (convrstn_id, call_date, gcp_refresh_date, ban, 
            src_id, telephone, topic, detection_score) 
        VALUES(s.convrstn_id, s.call_date, s.gcp_refresh_date, s.ban, 
               s.src_id, s.telephone, s.topic, s.detection_score)