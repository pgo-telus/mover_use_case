# Global import
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np

# Local import
from core.models.movers import MoverDetector, NaiveMoverDetector


def batch_detect(
    df_sentences: pd.DataFrame, mover_detect: MoverDetector, batch_size: int,
    pth_tr_model: Path, pth_tmp_backup: Optional[Path] = None
):
    # Build random partition
    n_partitions = 1
    if len(df_sentences['call_convrstn_id'].unique()) > batch_size:
        n_partitions = int(len(df_sentences['call_convrstn_id'].unique()) / batch_size)
    d_partitions = {
        k: np.random.randint(0, n_partitions) for k in df_sentences['call_convrstn_id'].unique()
    }

    # Assign partition to conv
    df_sentences = df_sentences.assign(
        partition_ind=lambda df: df['call_convrstn_id'].apply(lambda x: d_partitions[x])
    )
    
    l_df_res, n_total, n_detected = [], 0, 0
    for pid, df_sub in df_sentences.groupby('partition_ind'):

        l_df_res.append(mover_detect.detect(df_sub, pth_tr_model, 'call_convrstn_id'))

        # Display stats and backup
        n_total += len(df_sub['call_convrstn_id'].unique())
        n_detected += len(l_df_res[-1])
        print(f"""
            There is {n_detected} (out of {n_total}) mover detected 
            ({round(n_detected / n_total, 2) * 100}%)
        """)
        
        if pth_tmp_backup is not None:
            df_detected = pd.concat(l_df_res, ignore_index=True)
            pd.concat(l_df_res, ignore_index=True).to_csv(pth_tmp_backup, index=False)
    
    return pd.concat(l_df_res, ignore_index=True)


def batch_naive_detect(
    df_sentences: pd.DataFrame, mover_detect: NaiveMoverDetector, batch_size: int
) -> pd.DataFrame:
    # Build random partition
    n_partitions = 1
    if len(df_sentences['call_convrstn_id'].unique()) > batch_size:
        n_partitions = int(len(df_sentences['call_convrstn_id'].unique()) / batch_size)
    d_partitions = {
        k: np.random.randint(0, n_partitions) for k in df_sentences['call_convrstn_id'].unique()
    }

    # Assign partition to conv
    df_sentences = df_sentences.assign(
        partition_ind=lambda df: df['call_convrstn_id'].apply(lambda x: d_partitions[x])
    )

    l_df_res, n_total, n_detected = [], 0, 0
    for pid, df_sub in df_sentences.groupby('partition_ind'):

        l_df_res.append(mover_detect.detect(df_sub, 'call_convrstn_id'))

        # Display stats and backup
        n_total += len(df_sub['call_convrstn_id'].unique())
        n_detected += len(l_df_res[-1])
        print(f"""
            There is {n_detected} (out of {n_total}) mover detected 
            ({round(n_detected / n_total, 2) * 100}%)
        """)
        
    return pd.concat(l_df_res, ignore_index=True)
    
