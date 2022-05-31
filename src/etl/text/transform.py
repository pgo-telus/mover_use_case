# Global import
from typing import Dict, Optional, List, Tuple, Union, Callable
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import copy
import re

# Local import
from core.models.generic.intent import IntentDetector, MultiIntentDetector, MultiIntentComp, IntentComp


def build_bigrams(corpus: List[str]) -> List[List[str]]:
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)
        
    return lst_corpus


def sub_tokens(text: str, l_tokens: List[str]):
    text = re.sub(rf"({'|'.join(l_tokens)})", '', text.lower()).strip()
    return text


def clean_text(
    text: str, d_stopwords: Optional[Dict[str, bool]] = None, l_tags: Optional[List[str]] = None
) -> List[str]:
    """
    Remove stop words at extrimities of the text, remove nonascii character

    :param text: str
        text to clean.
    :return: list[str]
        Cleaned tokens.
    """
    if d_stopwords is None:
        d_stopwords = {}
    
    if not text:
        return []

    # Remove tags if any
    if l_tags is not None:
        text = re.sub(rf"({'|'.join(l_tags)})", '', text).strip().lower()

    # Clean text from special sep char
    text = re.sub(r'(\-|\_|\.|\/|\\|\:|\;|\,|\n|\[|\])', ' ', text).strip()

    # Remove stopwords and length 1 tokens
    l_tokens = [
        t for t in re.findall(r'(?i)((?:[a-z]|\')+)', text) 
        if not d_stopwords.get(t, False) and len(t) > 1
    ]

    return l_tokens


def get_first_clean_text(
    l_lines: List[str], d_stopwords: Optional[Dict[str, bool]] = None, 
    l_tags: Optional[List[str]] = None, reverse: Optional[bool] = False
):
    if not l_lines:
        return []

    if reverse:
        l_lines = copy.deepcopy(l_lines[::-1])

    for l in l_lines:
        l_tokens = clean_text(l, d_stopwords, l_tags)
        if l_tokens:
            return l_tokens

    return []


def get_match_keywords(
    text: str, l_targets: List[str], return_positions: Optional[bool] = False
) -> List[str]:
    """
    
    """
    if not return_positions:
        l_match_targets = sum(
            [re.findall(fr"((?:\s|^){tar}(?:\s|\.|\,|\!|\?|$|(?:\'s)))", text) for tar in l_targets], []
        )
        return [m.strip() for m in set(l_match_targets)] 

    else:
        l_match_targets = []
        for tar in l_targets: 
            p = re.compile(fr"((?:\s|^){tar}(?:\s|\.|\,|\!|\?|$|(?:\'s)))")
            l_match_targets.extend([(m.start(), m.group()) for m in p.finditer(text)])
        
        return l_match_targets
    

def is_match_regex(text: str, l_regex: List[str]) -> bool:
    for i, p in enumerate([re.compile(r) for r in l_regex]): 
        if len(re.compile(p).findall(text)) > 0:
            return True
    
    return False

    
def get_match_regex(
    text: str, l_regex: List[str], return_positions: Optional[bool] = False, 
    return_regex_ind: Optional[bool] = False
) -> Union[List[str], Tuple[List[int], List[str]], Tuple[List[int], List[int], List[str]]]:
    """
    
    """
    l_match_targets, l_regex_ind, l_positions = [], [], []
    for i, p in enumerate([re.compile(r) for r in l_regex]): 
        if not return_positions:
            l_sub_match = p.findall(text)
            
        else:
            l_tmp = [(m.start(), m.group()) for m in p.finditer(text)]
            l_sub_pos, l_sub_match = [t[0] for t in l_tmp], [t[1] for t in l_tmp]
            l_positions.extend(l_sub_pos)

        if l_sub_match:
            l_regex_ind.append(i)
            l_match_targets.extend(l_sub_match)
            
    if not return_positions:
        if return_regex_ind:
            return l_regex_ind, l_match_targets
        else:
            return l_match_targets
    
    else:
        if return_regex_ind: 
            return l_regex_ind, l_positions, l_match_targets
        else:
            return l_positions, l_match_targets

        
def intent_encoding(
    df_convs: pd.DataFrame, conv_id_col: str, processor: Callable, 
    intent_model: Union[IntentDetector, IntentComp], split_intent: bool, 
    batch_size: int = 500, sep_sntce: str = r'(?:\.|\;|\,)'
):
    # Create group
    l_groups, n_part = [(conv_id, df) for conv_id, df in df_convs.groupby(conv_id_col)], 1
    if len(l_groups) > batch_size:
        n_part = int(len(l_groups) / batch_size) + 1

    d_encodings = {}
    for i in range(n_part):
        d_docs = {}
        for conv_id, df_sub in l_groups[i * batch_size:(i + 1) * batch_size]:
            # Format & filter dataframe's sentences 
            d_docs[conv_id] = [t[1] for t in processor(df_sub)]
        
        # Update encoding dict
        if not split_intent:
            d_encodings.update(intent_model.min_dist_multi_doc(d_docs, split_sntce_by=sep_sntce))
        else:
            d_encodings.update(intent_model.all_dist_multi_doc(d_docs, split_sntce_by=sep_sntce))

    return d_encodings


def multi_intent_encoding(
    df_convs: pd.DataFrame, conv_id_col: str, processor: Callable, 
    intent_model: Union[MultiIntentDetector, MultiIntentComp], 
    split_intent: bool, batch_size: int = 500, sep_sntce: str = r'(?:\.|\;|\,)'
):
    # Create group
    l_groups, n_part = [(conv_id, df) for conv_id, df in df_convs.groupby(conv_id_col)], 1
    if len(l_groups) > batch_size:
        n_part = int(len(l_groups) / batch_size) + 1

    d_encodings = {}
    for i in range(n_part):
        d_docs = {}
        for conv_id, df_sub in l_groups[i * batch_size:(i + 1) * batch_size]:
            # Format & filter dataframe's sentences 
            d_docs[conv_id] = [t[1] for t in processor(df_sub)]

        # Update encoding dict
        if not split_intent:
            for key, d in intent_model.min_dist_multi_doc(d_docs, split_sntce_by=sep_sntce).items():
                d_encodings[key] = {**d_encodings.get(key, {}), **d}
        else:
            for key, d in intent_model.all_dist_multi_doc(d_docs, split_sntce_by=sep_sntce).items():
                d_encodings[key] = {**d_encodings.get(key, {}), **d}
                
    return d_encodings


def sentence_embedings(sentence: str, model: SentenceTransformer) -> np.ndarray:
    """
    
    """
    embeddings = model.encode(sentence)
    return embeddings
