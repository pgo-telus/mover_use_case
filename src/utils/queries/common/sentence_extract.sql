declare min_date date;
declare max_date date;

-- Analysis period run for the month of January 2022
set max_date = DATE('{max_date}');
set min_date = DATE('{min_date}');

with
sntnces as
(
    select
    a.call_convrstn_id, -- Conversation unique identifier
    lower(b.sntnce) as sntnce, -- Sentence of the conversation
    b.sntnce_create_nnsec_qty as sntnce_ts, -- Timestamp of the sentence in nanosecond
    b.sntnce_lang_cd, -- Detected lang of sentence
    b.sntnce_partcpnt_role, -- tag of who is talking
    b.sntnce_sntmnt_scor_qty, -- sentiment score (-1 negative, 1 positive)
    b.sntnce_sntmnt_mgntd_qty -- sentiment magnitude (0 no sentiment +infty sentiment in sentence)
    from
    `cio-datahub-enterprise-pr-183a.ent_cust_intractn_ccai.bq_voice_call_insights` a
    left join
    unnest(convrstn_sntnce) b
    where
    call_convrstn_dt between min_date and max_date
    and
    b.sntnce != ''
    and
    b.sntnce != '\n'
    and
    b.sntnce is not null
    and 
    b.sntnce_partcpnt_role is not null
    and 
    b.sntnce_lang_cd is not null
), 
sntnces_elligible_len as (
    select a.call_convrstn_id, count(a.sntnce_partcpnt_role) as cnt_partcpnt
    from (
        select call_convrstn_id, sntnce_partcpnt_role, count(sntnce) as cnt_sntce
        from sntnces
        group by call_convrstn_id, sntnce_partcpnt_role
    ) as a
    where a.cnt_sntce >= {min_exchange}
    group by a.call_convrstn_id
    having count(a.sntnce_partcpnt_role) > 1
), 
sntnces_elligible_lang as (
    select b.call_convrstn_id, count(b.sntnce_lang_cd) as cnt_lang, 
    min(b.sntnce_lang_cd) as min_lang
    from (
        select call_convrstn_id, sntnce_lang_cd, count(sntnce) as cnt_sntce
        from sntnces
        group by call_convrstn_id, sntnce_lang_cd
    ) as b
    group by b.call_convrstn_id
    having count(b.sntnce_lang_cd) = 1 and lower(min(b.sntnce_lang_cd)) = '{allowed_lang}'
)

select
    a.call_convrstn_id, -- Conversation unique identifier
    lower(a.sntnce) as sntnce, -- Sentence of the conversation
    a.sntnce_ts as sntnce_ts, -- Timestamp of the sentence
    a.sntnce_partcpnt_role, -- tag of who is talking
    a.sntnce_sntmnt_scor_qty * a.sntnce_sntmnt_mgntd_qty as sntnce_sntmnt
from 
    sntnces a
inner join 
    sntnces_elligible_len b
on
    a.call_convrstn_id = b.call_convrstn_id
inner join 
    sntnces_elligible_lang c
on
    a.call_convrstn_id = c.call_convrstn_id
