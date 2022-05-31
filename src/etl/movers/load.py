# Global import
from typing import Dict, List, Any
from pydoc import locate
from pathlib import Path
import pandas as pd

# Local import 
from core.etl.load import batch_bq_upsert


class BqFormater:
    def __init__(self, l_opt_id_cols: List[Dict[str, str]]):
        self.opt_id_cols = l_opt_id_cols
        
    def __call__(self, row):
        # Get dates
        curr_date = str(pd.Timestamp.utcnow().date())
        conv_date = str(row['call_convrstn_date']) if not pd.isnull(row['call_convrstn_date']) else 'NULL'

        # Create values
        values  = f"'{row['conv_id']}','{conv_date}','{curr_date}'"

        # Add nullabe identification  
        for d in self.opt_id_cols:
            if row[d['name']] and not pd.isnull(row[d['name']]):
                val = f",'{locate(d['type'])(row[d['name']])}'"
                if d['type'] != 'str':
                    val = val.replace("'", "")

                values += val
            else:
                values += ',NULL'

        # Add scores
        score_values = ','.join(["'moves'", str(round(row['detection_score'], 3))])
        values += f",{score_values}"
        return '(' + values + ')'


def load_detection_to_datahub(
    bq_client: Any, df_detected: pd.DataFrame, pth_query: Path, project_name: str, dataset: str, 
    l_opt_id_cols: List[Dict[str, str]]
) -> None:
    
    # Create table if not exists
    sql_create = (pth_query / 'create_table.sql').read_text()\
        .format(project_name=project_name, dataset=dataset)
    code = bq_client.query(sql_create)

    # Upsert detection       
    sql_upsert = (pth_query / 'upsert_preds.sql').read_text()
    batch_bq_upsert(
        df_detected, bq_client, sql_upsert, project_name, dataset, BqFormater(l_opt_id_cols)
    )        
