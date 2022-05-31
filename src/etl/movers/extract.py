# Global import 
from typing import Tuple, List, Optional, Any, Dict
from yaml import safe_load
from pathlib import Path
import pandas as pd
import pickle

# Local import 
from core.models.generic.intent import IntentDetector
from core.models.movers import MoverDetector, NaiveMoverDetector
from core.etl.extract import extract_stopwords, extract_from_bucket



def extract_training_data(
    pth_data='', csv_file='convo.csv', pth_stopwords: Optional[Path] = None, 
    pth_other_kws: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    '''
        Load dataset from SQL or Flatfile
        :parameter
            :param mode: string - format of the data source (sql, csv)
            :param csv_file: string - CSV source file
            :param sql_file: string - SQL source file
        :return
            dataset
    '''
    df = pd.read_csv(pth_data)
    l_stopwords = None
    if pth_stopwords is not None:
        l_stopwords = list(extract_stopwords(pth_stopwords).keys())
    
    l_tags = None
    if pth_other_kws is not None:
        d_other_kw = safe_load(pth_other_kws.open())
        l_stopwords.extend(d_other_kw['to_remove'])
        l_tags = d_other_kw['tags']
        
    
    return df, l_stopwords, l_tags


def extract_all_model_data(
    pth_model_data: Path, detection_threshold: float, use_stopwords: bool, 
    split_regex: Optional[bool] = True, **kwargs
):
    
    # Extract target competitor's keywords
    d_keywords = safe_load((pth_model_data / 'movers' / 'keywords.yaml').open())
    regex = {
        'movers': d_keywords['regex_movers'], 
        'cancel': d_keywords['regex_cancel'], 
        'holiday': d_keywords['regex_holiday'], 
        'expression': d_keywords['regex_expression'],  
        'things': d_keywords['regex_things'],
        'negation': d_keywords['regex_negation'],
        'past': d_keywords['regex_past']
    }
    if not split_regex:
        regex = [reg for l_regs in regex.values() for reg in l_regs]
    
    # Load utils datas
    l_word_to_remove, l_tags = d_keywords['to_remove'], d_keywords['tags']
    pth_stopwords = None
    if use_stopwords:
        pth_stopwords = pth_model_data / 'stopwords' / 'english'
    d_stopwords = extract_stopwords(pth_stopwords, l_word_to_remove)

    # Load intent detection model
    l_sntnces = safe_load(
        (pth_model_data / 'movers' / 'sentences.yaml').open()
    )['intent_move']
    intent_detector = IntentDetector(
        l_sntnces, pth_model_data / 'transformer_model', 
        detection_threshold=detection_threshold
    )
    
    return regex, l_tags, d_stopwords, intent_detector


def extract_naive_model(
    pth_model_data: Path, use_stopwords: Optional[bool] = False, **kwargs
) -> NaiveMoverDetector:
    
    # Get all model data
    d_regex, l_tags, d_stopwords, _ = extract_all_model_data(
        pth_model_data, 0, use_stopwords, split_regex=True
    )

    mover_detect = NaiveMoverDetector(d_regex, d_stopwords, l_tags) 

    return mover_detect


def extract_local_model(
    pth_model_data: Path, pth_clf: Path, detection_threshold: float,
    use_stopwords: Optional[bool] = False, **kwargs
) -> MoverDetector:
    
    # Get all model data
    d_regex, l_tags, d_stopwords, intent_detector = extract_all_model_data(
        pth_model_data, detection_threshold, use_stopwords
    )
    
    # Load classifier
    with pth_clf.open(mode='rb') as m:
        mov_clf = pickle.load(m)
        
    mover_detect = MoverDetector(d_regex, mov_clf, intent_detector, d_stopwords, l_tags) 

    return mover_detect


def extract_storage_model(
    bucket: Any, pth_model_data:Path, pth_clf: Path, detection_threshold: float, 
    use_stopwords: bool, **kwargs: Dict[str, Any] 
) -> MoverDetector:
    d_regex, l_tags, d_stopwords, intent_detector = extract_all_model_data(
        pth_model_data, detection_threshold, use_stopwords
    )
    
    # Extract classifier from bucket
    mov_clf = extract_from_bucket(bucket, pth_clf, data_ext='pickle', err_raise=True)

    # Instantiate competitor model
    mover_detect = MoverDetector(d_regex, mov_clf, intent_detector, d_stopwords, l_tags) 

    return mover_detect
