# Global import
from typing import Dict, List, Tuple, Optional, Any
from google.cloud.exceptions import NotFound
from pathlib import Path
import pandas as pd
import pickle
import json
import os

# Local import 


def extract_stopwords(
    pth_stopwords: Optional[Path] = None, l_others: Optional[List[str]] = None
):
    # Build dict
    l_stopwords = []
    if pth_stopwords is not None:
        l_stopwords = pth_stopwords.read_text().split('\n')
    d_stopwords = {k: True for k in l_stopwords}    
    
    # Add other words if any
    if l_others:
        d_stopwords.update({k: True for k in l_others})
    
    return d_stopwords


def regex_from_text(l_texts: List[str]) -> str:
    return list(set([rf"(?:(?:\s|^|\.|\,)({t})(?:\s|$|\.|\,|\?|\!|\;))" for t in l_texts if t]))


def extract_geo_regex(pth_cntr: Path, pth_rgn: Path) -> Dict[str, List[str]]:
    
    # Read geo files
    df_countries = pd.read_csv(pth_cntr, sep=';').fillna("")
    df_regions = pd.read_csv(pth_rgn, sep=';').fillna("")
    
    # Init geo regex dict
    d_geo_regex = {
        'country': [], 'nationality': [], 'language': [], 'region': [], 
        'region_people': []
    }

    # Get country info
    for _, row in df_countries.iterrows():
        # Country name & alias
        l_sub_cntrs = [c.lower().strip() for c in row['country'].split('/')]
        d_geo_regex['country'].extend([
            r for r in regex_from_text(l_sub_cntrs) if r not in d_geo_regex['country']
        ])
        
        # Country nationality & alias
        l_sub_ntnlts = [c.lower().strip() for c in row['nationality'].split('/')]
        d_geo_regex['nationality'].extend([
            r for r in regex_from_text(l_sub_ntnlts) if r not in d_geo_regex['nationality']
        ])
        
        # Country languages
        l_sub_lugs = [c.lower().strip() for c in row['language'].split('/')]
        d_geo_regex['language'].extend([
            r for r in regex_from_text(l_sub_lugs) if r not in d_geo_regex['language']
        ])
        
    # Get regional info
    for _, row in df_regions.iterrows():
        # Country name & alias
        d_geo_regex['region'].append(regex_from_text([row['name'].lower().strip()])[0])
        d_geo_regex['region_people'].append(regex_from_text([row['population'].lower().strip()])[0])
        
    return d_geo_regex


def extract_from_bucket(
    bucket: Any, path: str, data_ext: str, err_raise: bool = False
) -> Any:
    
    # Download file locally
    local_path = f'/tmp/bucket_content.{data_ext}'

    try:
        bucket.blob(path).download_to_filename(local_path)
    except (NotFound, FileNotFoundError) as e:
        if err_raise:
            raise FileNotFoundError()
        return None
    
    # Read downloaded file
    if data_ext == 'json':
        with open(local_path, 'r') as handle:
            data = json.load(handle)
            
    elif data_ext in ('pickle', 'pkl', 'pckl'):
        with open(local_path, 'rb') as handle:
            data = pickle.load(handle)
    else:
        raise ValueError(f'Not recognize ext {data_ext}')
    
    return data


def extract_dir_from_bucket(bucket: Any, local_path: Path, prefix: str):
    for blob in bucket.list_blobs(prefix=prefix):
        path = local_path / os.path.relpath(Path(blob.name), prefix)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        blob.download_to_filename(path.as_posix()) 


def extract_bq_data(bq_client, sql=None, pth_query=None):
    if sql is not None:
        df = bq_client.query(sql).to_dataframe()
    elif pth_query is not None:
        sql = pth_query.read_text()
        df = bq_client.query(sql).to_dataframe()
    else:
        raise ValueError('`sql` or `pth_query` should be set')  
    return df


def extract_pr_codes(pth_codes: Path):
    d_pr_codes = pd.read_csv(pth_codes)\
        .astype({"code": str})\
        .set_index('code')\
        .loc[:, 'prvnce']\
        .to_dict()
    
    return d_pr_codes


def format_conv_df(
    df_conv: pd.DataFrame, d_pr_codes: Dict[str, str], num_col: List[str]
) -> pd.DataFrame:
    
    df_conv = df_conv.assign(
        pr_name=lambda df: df[num_col].apply(lambda x: d_pr_codes.get(str(x)[:3], None))
    )         
    return df_conv


def filter_convs(
    df_convs: pd.DataFrame, df_sntces: pd.DataFrame, conv_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out non elligible conv
    
    The criterion for elligibility of conv are:
        - The conversation should have be matched with conv id of sentences dataframe 
    """
    # Build mask dictionary
    d_conv_cid = {x: True for x in df_convs[conv_id_col].unique()}
    d_sntc_cid = {x: True for x in df_sntces[conv_id_col].unique()}
    
    # Filter convs and sentences
    df_filtered_convs = df_convs.loc[
        df_convs[conv_id_col].apply(lambda x: d_sntc_cid.get(x, False))
    ]
    df_filtered_sntnces = df_sntces.loc[
        df_sntces[conv_id_col].apply(lambda x: d_conv_cid.get(x, False))
    ]
    
    return df_filtered_convs, df_filtered_sntnces


def extract_all_conv_data(
    bq_client, d_params, pth_conv_query: Path, pth_sntc_query: Path, pth_cnty_codes: Path, 
    d_query_params: Dict[str, str]
):
    # Query conversations
    sql = pth_conv_query.read_text().format(**d_query_params)
    df_conv = extract_bq_data(bq_client, sql)

    # load province code and format conversation data
    d_pr_codes = extract_pr_codes(pth_cnty_codes)
    df_conv_formated = format_conv_df(df_conv, d_pr_codes, d_params['num_col'])[d_params['conv_col_formatted']]

    # Get sentences
    sql = pth_sntc_query.read_text()\
        .format(
            allowed_lang=d_params['allowed_lang'], min_exchange=d_params['min_exchange'], 
            **d_query_params
        )
    df_sntces = extract_bq_data(bq_client, sql)

    # Filter eligible conversation
    df_conversations, df_sentences = filter_convs(
        df_conv_formated, df_sntces, d_params['conv_id_col']
    )
    
    return df_conversations, df_sentences
