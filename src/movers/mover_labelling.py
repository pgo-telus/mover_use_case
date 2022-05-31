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
from random import shuffle
from yaml import safe_load
import google.oauth2.credentials
from google.cloud import bigquery
from IPython.core.display import HTML
from IPython.core.display import display
from IPython.display import clear_output
from typing import Dict, List, Tuple, Any

# Set global vars
pth_project = Path(__file__.split('core')[0])
sys.path.insert(0, str(pth_project))

# import local modules
from core.utils.gcp import connect_bq_services
from core.etl.movers.extract import extract_all_model_data
from core.etl.load import load_examples_to_datahub
from core.etl.extract import extract_bq_data
from core.etl.text.transform import sub_tokens, get_match_regex, is_match_regex
from core.etl.movers.transform.features import intent_encoding, process_conv
from core.models.generic.intent import IntentDetector


class MoverLabellingArgParser(argparse.ArgumentParser):
    def __init__(self,):
        super().__init__(description='Arg parser for competitor preds.')
        
        self.add_argument(
            '--nb_select', metavar='nb_select', type=int, default=1500,
            help='Number of example to select for labelling'
        )


def build_examples_dataset(
    pth_data: Path, pth_util_data: Path, d_params: Dict[str, Any], 
    nb_select: int = 2000
):
    # Extract conversation and sentences 
    df_sentences = pd.read_csv(pth_data / 'extract' / 'sentences.csv', index_col=None)

    # Add a unique chunk_id to all sentences 
    df_sentences = df_sentences.sort_values(by=['call_convrstn_id', 'sntnce_ts'])\
        .assign(chunk_id=np.arange(len(df_sentences)))

    # Extract model data
    l_regex, l_tags, d_stopwords, intent_detector = extract_all_model_data(
        pth_util_data, **d_params['model']
    )
    
    # pre filtering of conversation's sentences
    df_sentences = df_sentences.loc[
        df_sentences['sntnce'].apply(lambda x: is_match_regex(x, l_regex))
    ]

    # Encode intents
    d_encodings = intent_encoding(df_sentences, intent_detector, False, 500, d_stopwords, l_tags)
    
    # Shuffle convs
    groups = [(conv_id, df) for conv_id, df in df_sentences.groupby('call_convrstn_id')]
    shuffle(groups)

    n, log_rate, l_examples  = 0, 500, []
    for conv_id, df_sub in groups:
        n += 1

        if n % log_rate == 0:
            n_examples = len(set([d['call_convrstn_id'] for d in l_examples]))
            print(f'{n} conv processed, {n_examples} example found')
            
        if len(set([d['call_convrstn_id'] for d in l_examples])) >= nb_select:
            break
        
        # process sentences 
        l_sntces = process_conv(df_sub, d_stopwords, l_tags)

        # Match competitor keywords & detect intent 
        ax_intent_scores = d_encodings[conv_id]
        for i, (cid, l) in enumerate(l_sntces):
            if not l:
                continue
            
            l_match = get_match_regex(
                l, l_regex, return_positions=False, return_regex_ind=False
            )

            if not l_match:
                continue

            l_examples.append({'call_convrstn_id': conv_id, 'chunk_id': cid, 'text': l})
            
    return pd.DataFrame(l_examples)


def batch_load_examples(df_examples: pd.DataFrame, pth_queries):
    bq_client = connect_bq_services(d_config['gcp-project-name'])
    batch_rate = 1000
    n_pass = int(len(df_examples) / batch_rate) + 1
    for i in range(n_pass):
        df_sub = df_examples.iloc[i * batch_rate: (i+1) * batch_rate]

        load_examples_to_datahub(
            bq_client, df_sub, pth_queries, d_config['gcp-project-name'], 
            d_config['dataset'], d_params['labelling']['table_name']
        )
        time.sleep(1)
        count = extract_bq_data(
            bq_client, 
            '''SELECT count(*) 
            from `divg-pgspeech-pr-b8a291.divg_pgspeech_pr_dataset.examples_mover`'''
        )
        print(f'Count of rows is {count.iloc[0, 0]}')
        print(f'{i}-{i * batch_rate}-{(i+1) * batch_rate}')


if __name__ == '__main__':
    # Set Path
    pth_data = pth_project / 'data'
    pth_util_data = pth_project / 'core' / 'utils' / 'data'
    pth_queries = pth_project / 'core' / 'utils' / 'queries' / 'common'
    pth_creds = pth_project / 'conf' / 'local' / 'project_config.yaml'
    
    # Load project configs & params
    d_config = safe_load(pth_creds.open())
    d_params = safe_load((pth_project / 'core' / 'parameters' / 'movers.yaml').open())
    
    # Init Big query client
    bq_client = connect_bq_services(d_config['gcp-project-name'])
    
    # Get args
    args = MoverLabellingArgParser().parse_args()
    
    # Build example df & load to datahub
    df_examples = build_examples_dataset(
        pth_data, pth_util_data, d_params, args.nb_select
    )    
    batch_load_examples(df_examples, pth_queries)

    
