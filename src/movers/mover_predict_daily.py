# import global modules
import warnings
warnings.filterwarnings('ignore')
import os
import re
import sys
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from yaml import safe_load
import google.oauth2.credentials
from google.cloud import storage
from google.cloud import bigquery
from IPython.core.display import display
from IPython.core.display import HTML
from typing import List, Dict, Optional, Any, Tuple

# Add paths
pth_project = Path(__file__.split('core')[0])
sys.path.insert(0, str(pth_project))

# import local modules
from core.utils.gcp import connect_bq_services
from core.utils.gcp import connect_storage_services
from core.etl.extract import extract_all_conv_data
from core.etl.movers.extract import extract_storage_model
from core.etl.movers.load import load_detection_to_datahub
from core.etl.movers.transform.predict import batch_detect


class MoverPredictArgParser(argparse.ArgumentParser):
    def __init__(self,):
        super().__init__(description='Arg parser for mover preds.')
        
        self.add_argument(
            '--model_type', metavar='model_type', type=str, default='unsupervised', 
            help='Start date to extract conversations (included)'
        )
        self.add_argument(
            '--date_start', metavar='date_start', type=str, 
            help='Start date to extract conversations (included)'
        )
        self.add_argument(
            '--date_end', metavar='date_end', type=str, 
            help='End date to extract conversations (included)'
        )
        self.add_argument(
            '--batch_size', metavar='batch_size', type=int, 
            help='Nb of conversation chunk to predict all together in a batch.'
        )


def predict_mover(
    d_params: Dict[str, str], bucket: Any, bq_client: Any, date_start: str, date_end: str, 
    pth_util_data: Path, pth_model: Path, model_type: str, batch_size: Optional[int] = 5000
):       
    # Get conversation
    df_conversations, df_sentences = extract_all_conv_data(
        bq_client, d_params['data_extract'], pth_queries / 'common' / 'conv_extract.sql', 
        pth_queries / 'common' / 'sentence_extract.sql', pth_util_data / 'cnty_codes.csv', 
        {'min_date': date_start, 'max_date': date_end}
    )
    
    # Extract mover model locally
    mover_detect = extract_storage_model(bucket, pth_util_data, pth_model, **d_params['model'])
    
    # Batch predict
    pth_tr_model = pth_util_data / 'transformer_model'
    df_movers = batch_detect(
        df_sentences, mover_detect, batch_size, pth_tr_model
    )
    # Format final result.
    l_conv_cols = [d['name'] for d in d_params['load']['other_id_cols']] + \
        [d_params['load']['id_col']] + [d_params['load']['date_col']]
    
    df_to_load = df_movers.merge(
        df_conversations[l_conv_cols], left_on='conv_id', 
        right_on=d_params['load']['id_col'], how='left'
    )\
        .drop(columns=d_params['load']['id_col'])

    # Load to GCP BiqQuery
    bq_client = connect_bq_services(d_config['gcp-project-name'])
    load_detection_to_datahub(
        bq_client, df_to_load, pth_queries / 'movers', d_config['gcp-project-name'], 
        d_config['dataset'], d_params['load']['other_id_cols']
    )

    
if __name__ == '__main__':
    # Set Path
    pth_data = pth_project / 'data'
    pth_util_data = pth_project / 'core' / 'utils' / 'data'
    pth_queries = pth_project / 'core' / 'utils' / 'queries'
    pth_creds = pth_project / 'conf' / 'local' / 'project_config.yaml'
    
    # Load project configs & params
    d_config = safe_load(pth_creds.open())
    d_params = safe_load((pth_project / 'core' / 'parameters' / 'movers.yaml').open())
    d_params.update(safe_load((pth_project / 'core' / 'parameters' / 'common.yaml').open()))
    
    # Gte classifier path
    model_dir, model_name = d_params['model']['model_dir'], d_params['model']['model_name']
    pth_model = f"{d_config['pth_topic_detection_model']}/{model_dir}/{model_name}"

    # Init Big query client
    bq_client = connect_bq_services(d_config['gcp-project-name'])
    storage_client = connect_storage_services(d_config['gcp-project-name'])
    bucket = storage_client.bucket(d_config['bucket'])
    
    # Get args
    args = MoverPredictArgParser().parse_args()

    date_start, date_end = args.date_start, args.date_end
    if date_end is None:
        date_end = str((pd.Timestamp.utcnow() - pd.Timedelta(days=1)).date())
        
    if date_start is None:
        date_start = date_end
        
    predict_mover(
        d_params, bucket, bq_client, date_start, date_end, pth_util_data, 
        pth_model, args.batch_size or 5000
    )

