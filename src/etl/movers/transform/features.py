# Global import
from typing import Dict, Optional, List, Any, Generator
import pandas as pd
import numpy as np

# Local import
from core.data_models.movers import MoverDataset, MoverMetaDataset
from core.etl.text.transform import sub_tokens, get_match_regex, is_match_regex, intent_encoding


def apply_preselection_rules(
    df_convs: pd.DataFrame, d_regex: Dict[str, List[str]], offset: Optional[int] = 3
):
    l_selected, l_removed_cid = [], []
    for i, row in df_convs.iterrows(): 
        l_move_match = get_match_regex(row['sntnce'], d_regex['movers'])

        if not l_move_match or row['sntnce_partcpnt_role'] != 'END_USER':
            continue
        
        for k in [-offset, offset]:
            l_cancel_match = get_match_regex(df_convs.loc[i + k, 'sntnce'], d_regex['cancel'])
        
        l_holiday_match = get_match_regex(row['sntnce'], d_regex['holiday'])
        l_expression_match = get_match_regex(row['sntnce'], d_regex['expression'])
        l_things_match = get_match_regex(row['sntnce'], d_regex['things'])
        l_past_match = get_match_regex(row['sntnce'], d_regex['past'])
        l_negation_match = get_match_regex(row['sntnce'], d_regex['negation'])
        
        if l_cancel_match:
            l_removed_cid.append(row['call_convrstn_id'])
            continue
            
        if l_expression_match or l_holiday_match or l_things_match or l_past_match or l_negation_match:
            continue

        l_selected.append(row.to_dict())
    
    return pd.DataFrame(l_selected), l_removed_cid


def extract_convs_features(
    df_convs: pd.DataFrame, d_regex: Dict[str, List[str]],  
    intent_detector: Any, d_labels: Optional[Dict[str, Any]] = None,
    d_stopwords: Dict[str, bool] = None, l_tags: Optional[List[str]] = None
) -> MoverDataset:
    
    # Get meta dataset & encod convs
    meta_dataset = MoverMetaDataset().set_optional_names(d_labels is not None)
    processor = ConvProcessor(d_stopwords, l_tags)

    # pre filtering of conversation's sentences
    df_convs, l_removed_cid = apply_preselection_rules(df_convs, d_regex)
    
    # Encodings of sentences
    d_encoding = intent_encoding(
        df_convs, 'call_convrstn_id', processor, intent_detector, True, 3000
    )
    
    l_features = []
    for conv_id, df_sub in df_convs.groupby('call_convrstn_id'):
        # If d_labels is passed, skip when conv id is not in d_labels
        if d_labels is not None:
            if d_labels.get(conv_id, None) is None:
                print('here')
                continue
        
        if conv_id in l_removed_cid:
            continue
            
        # Get labels if any
        d_sub_labels = d_labels[conv_id] if d_labels is not None else None

        # Format & filter dataframe's sentences 
        l_sntces = processor(df_sub)

        # Compute chunk features
        l_chunk_features = extract_chunks_features(
            conv_id, meta_dataset, l_sntces, d_encoding[conv_id],
            intent_detector.detection_threshold,
            d_regex['movers'], d_sub_labels,  d_stopwords, l_tags
        )
        l_features.extend(l_chunk_features)
        
    # Gather and format features
    mover_dataset = format_dataset(
        l_features, meta_dataset, len(intent_detector),
        target_name='label' if d_labels is not None else None
    )

    return mover_dataset


class ConvProcessor:
    
    def __init__(self, d_stopwords: Dict[str, bool] = None, l_tags: Optional[List[str]] = None):
        self.stopwords = d_stopwords or {}
        self.tags = l_tags or []
    
    def __call__(self, df_conv: pd.DataFrame):
        
        df_conv = df_conv.sort_values(by='sntnce_ts').reset_index(drop=True)
        l_ind_user = [i for i, r in df_conv.iterrows() if r['sntnce_partcpnt_role'] == 'END_USER']
        l_sntces = [(r['chunk_id'], r['sntnce']) for i, r in df_conv.iterrows() if i in l_ind_user]

        l_token_to_remove = self.tags + [k for k in self.stopwords.keys()]
        l_sntces = [(cid, sub_tokens(l, l_token_to_remove)) for cid, l in l_sntces]

        return l_sntces


def extract_chunks_features(
    conv_id: str, meta_dataset: MoverMetaDataset, l_sntces: List[str], 
    ax_intent_scores: np.ndarray, intent_threshold: float, l_regex: List[str], 
    d_labels: Optional[Dict[str, int]] = None, d_stopwords: Dict[str, bool] = None, 
    l_tags: Optional[List[str]] = None, 
) -> Generator[Dict[str, Any], None, None]:
    
    for i, (cid, l) in enumerate(l_sntces):
        if not l:
            continue
            
        # Skip if labels not None and chunk unlabelled
        d_features = {}
        if d_labels is not None:
            if d_labels.get(str(cid), None) is None:
                print('there')
                continue
                
            d_features = {'label': d_labels[str(cid)]}
        
        # Get tabular information about detection
        d_features.update({
            "conv_id": conv_id, "chunk_id": cid, 'sntce_len': len(l),
            "rel_pos_sntce": 0
        })
         
        # Get match regex in the sentence
        l_regex_ind, l_pos, _ = get_match_regex(
            l, l_regex, return_positions=True, return_regex_ind=True
        )
        
        # Feature match
        rel_pos_mtch = -1 if not l_pos else sum(l_pos) / (len(l) * len(l_pos))
        d_features["rel_pos_sntce"] = rel_pos_mtch
        
        # Add transformer features.
        d_features['intent'] = ax_intent_scores[i, :]
                    
        # Make sure the list of keys computed match with the data_model
        meta_dataset.validate_meta(list(d_features.keys()))

        yield d_features


def format_dataset(
    l_features: List[Dict[str, Any]], meta_dataset: MoverMetaDataset,
    n_intent: int, target_name: Optional[str] = None, 
) -> MoverDataset:
    """
    
    """
    # Start formating features
    l_X, l_y, d_ind2ids = [], [], {}
    for i, d_feature in enumerate(l_features):

        # Update mapping        
        d_ind2ids[i] = {k: d_feature[k] for k in meta_dataset.id_names}

        # Get tabular features
        ax_tabular_features = np.array(
            [d_feature[k] for k in meta_dataset.tabular_feature_names], dtype=float
        )

        # Format context embeddings features
        ax_context_features = d_feature[meta_dataset.context_feature_names[0]]

        # Gather all features
        l_X.append(np.hstack([ax_tabular_features, ax_context_features]))
        if target_name is not None:
            l_y.append(d_feature[target_name])

    # Set mover dataset
    mv_dataset = MoverDataset(
        np.vstack(l_X), d_ind2ids, y=np.array(l_y)[:, np.newaxis] if l_y else None
    ).set_feature_names(meta_dataset, n_intent)

    return mv_dataset

        
