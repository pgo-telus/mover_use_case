# Global import
from typing import Optional, Any, Callable
from pathlib import Path
import pandas as pd
import pickle

# Local import


def format_sql_example_values(row: pd.Series) -> str:
    return f"('{row['call_convrstn_id']}',{row['chunk_id']},'''{row['text']}''')"


def load_examples_to_datahub(
    bq_client, df_examples: pd.DataFrame, pth_query: Path, project_name: str, 
    dataset: str, table: str
) -> None:
    
    # Create table if not exists
    sql_create = (pth_query / 'label_create_table.sql').read_text()\
        .format(project_name=project_name, dataset=dataset, table=table)
    code = bq_client.query(sql_create)
    
    # Upsert examples if any
    if df_examples.empty:
        return 
    
    new_values = ',\n'.join(
        df_examples.apply(lambda row: format_sql_example_values(row), axis=1)
    )
    sql_upsert = (pth_query / 'label_upsert_preds.sql').read_text()\
        .format(project_name=project_name, dataset=dataset, table=table, new_values=new_values)

    code = bq_client.query(sql_upsert)
    

def load_model_to_gcp(bucket: Any, model: Any, pth_model: str):
    bucket.blob(pth_model).upload_from_string(
        data=pickle.dumps(model), content_type='application/octet-stream'
    )

    
def batch_bq_upsert(
    df:pd.DataFrame, bq_client: Any, sql_upsert: str, project_name: str, dataset: str, 
    row_formater: Callable, batch_size: Optional[int] = 1000
):
    import time
    n_pass = int(len(df) / batch_size) + 1
    for i in range(n_pass):
        df_sub = df.iloc[i * batch_size: (i+1) * batch_size]

        # Upsert detection       
        new_values = ',\n'.join(
            df_sub.apply(lambda row: row_formater(row), axis=1)
        )
        sql_sub_upsert = sql_upsert.format(
            project_name=project_name, dataset=dataset, new_values=new_values
        )

        code = bq_client.query(sql_sub_upsert)
        time.sleep(2)
