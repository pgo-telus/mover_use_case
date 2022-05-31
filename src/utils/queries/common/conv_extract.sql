declare min_date date;
declare max_date date;

-- Analysis period run for the month of January 2022
set max_date = DATE('{max_date}');
set min_date = DATE('{min_date}');

with conv_direct_field as (
  select
    call_convrstn_id, -- Conversation id
    call_convrstn_dt as call_convrstn_date, -- Conversation date
    bus_bacct_num_src_id,
  from `cio-datahub-enterprise-pr-183a.ent_cust_intractn_ccai.bq_voice_call_insights`
  where call_convrstn_dt between min_date and max_date 
),
conv_with_label as (
  SELECT
    a.call_convrstn_id,   
    if (b.label_key IN (
          'INITIAL_VIRTUAL_QUEUE', 
          'RVirtualQueue', 
          'VIRTUAL_QUEUE', 
          'VQ', 
          'outboundVQ', 
          'virtualQueue'
          ), 
        nullif(CAST(b.label_val as STRING),''),
        NULL
        ) AS virtualqueue,
    if (lower(b.label_key) IN (
          'ban', 
          'cm_ban', 
          'account_no', 
          'accountnumber', 
          'cust_acct_no', 
          'rsv4nl_cust_accountno'
          ), 
        if(CAST(b.label_val as STRING) in ('', '0000000000'), NULL, CAST(b.label_val as STRING)), 
        NULL
        ) AS ban,
    if (b.label_key ='RTargetAgentSelected',
        nullif(CAST(b.label_val as STRING), ''), 
        NULL
        ) AS agnt_id,
    if (upper(b.label_key) IN (
          'ANI', 
          'BILLING_PHONE_NUMBER',
          'CALLBACK_TN',
          'CM_BILLING_PHONE_NUMBER',
          'CUST_BILLING_PHONE',
          'LSC_ANI', 
          'PHONENUMBER', 
          'RSV4NL_USERS_TELEPHONENO',
          'TCCS_ANI',
          'YCCS_UTN',
          'UTN',
          'XIRELINE_CONTACT_TN', 
          'CALLBACKTN'
          ), 
        nullif(CAST(b.label_val as STRING),''),  
        NULL
        ) AS telephone
  from `cio-datahub-enterprise-pr-183a.ent_cust_intractn_ccai.bq_voice_call_insights` a
    CROSS JOIN
      UNNEST(convrstn_label) b
  WHERE
    call_convrstn_dt between min_date and max_date
), 
conv_agg_labels as (
  select call_convrstn_id, STRING_AGG(distinct virtualqueue, ',') as virtualqueues, cast(max(ban) as string) as bus_bacct_num, 
  STRING_AGG(distinct agnt_id, ',') as agnt_id, STRING_AGG(distinct telephone, ',') as telephone
  from conv_with_label
  group by call_convrstn_id
), 
iss_topic as (
  select call_convrstn_id, STRING_AGG(distinct iss_nm, ',') as topics from (
    select call_convrstn_id, b.iss_nm 
    from `cio-datahub-enterprise-pr-183a.ent_cust_intractn_ccai.bq_voice_call_insights` a
      cross join unnest(convrstn_iss) b
      where call_convrstn_dt between min_date and max_date
  )
  group by call_convrstn_id
)
select a.* , b.bus_bacct_num, b.virtualqueues, b.agnt_id, b.telephone, c.topics
from conv_direct_field a
inner join conv_agg_labels b
on a.call_convrstn_id = b.call_convrstn_id
inner join iss_topic c
on a.call_convrstn_id = c.call_convrstn_id
where b.bus_bacct_num is not null or b.telephone IS NOT NULL
