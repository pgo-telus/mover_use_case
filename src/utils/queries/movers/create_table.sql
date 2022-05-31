CREATE TABLE IF NOT EXISTS `{project_name}.{dataset}.mover_detection`
 (
   convrstn_id STRING(30) NOT NULL OPTIONS(
       description="""ID of the conversation. It is a refernce to field `call_convrstn_id` of table 
       `cio-datahub-enterprise-pr-183a.ent_cust_intractn_ccai.bq_voice_call_insights`""" 
   ),  
   call_date DATE OPTIONS(
       description="Date at which the call took place"
   ),
   gcp_refresh_date DATE NOT NULL OPTIONS(
       description="Date at which the call was added to the table"
   ), 
   ban STRING OPTIONS(
       description="BAN number associated with the person calling"
   ), 
   src_id INTEGER OPTIONS(
       description="This is the source system identifier."
   ),
   telephone STRING(20) OPTIONS(
       description="Subscription No"
   ),
   topic STRING(30) OPTIONS(
       description="topic of the detection = 'moves',"
   ),
   detection_score FLOAT64 NOT NULL OPTIONS(
       description="The probability of detection of competitor durng the conversation"
   )
 )
 OPTIONS(
   description="""Table gathering the result of the pipeline of mobe intent detection 
     from customer_call, the table contains only conversation ids for which there was 
     a probability of more than 50% that a moving intention was mentionned during the call."""
 );