CREATE TABLE IF NOT EXISTS `{project_name}.{dataset}.{table}`
 (
   call_convrstn_id STRING(30) NOT NULL OPTIONS(
       description="""ID of the conversation. It is a refernce to field `call_convrstn_id` of table 
       `cio-datahub-enterprise-pr-183a.ent_cust_intractn_ccai.bq_voice_call_insights`""" 
   ),  
   chunk_id INTEGER OPTIONS(
       description="Chunk number of the conversation"
   ), 
   text STRING OPTIONS(
       description="Text to display and label"
   )
 )
 OPTIONS(
   description="""Table gathering the conversation chunk to be labelled 
     for competitor's detection."""
 );