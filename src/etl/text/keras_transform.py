# Global import
from typing import List, Tuple, Dict, Any, Optional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.strings import lower, regex_replace
import numpy as np
import string
import re


def tokenize_text(
    corpus: List[List[str]], tokenizer: Optional[Tokenizer] = None
) -> Tuple[Tokenizer, Dict[str, int], Any]:

    if tokenizer is None:
        tokenizer = Tokenizer(
            lower=True, oov_token='[UNK]', split=' ',filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        tokenizer.fit_on_texts(corpus)

    dic_vocabulary = tokenizer.word_index

    ## padding sequence
    X = pad_sequences(
        tokenizer.texts_to_sequences(corpus), maxlen=15, padding="post", 
        truncating="post"
    )

    return tokenizer, dic_vocabulary, X


def build_embedding_matrix(dic_vocabulary: Dict[str, int], nlp: Any) -> np.ndarray:
    embeddings = np.zeros((len(dic_vocabulary) + 1 , 200))
    for word, idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] = nlp.wv[word]
        ## if word not in model then skip and the row stays all 0s
        except:
            pass

    return embeddings


def normalize_text(text: str) -> str:
    remove_regex = f'[{re.escape(string.punctuation)}]'
    result = lower(text)
    result = regex_replace(result, remove_regex, '')
    
    return result