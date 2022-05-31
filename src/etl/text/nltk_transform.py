# General module
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# text processing modules
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


def transform_text(
    text: str, pth_nltk: Path, flg_stemm: bool=False, flg_lemm: bool=True, 
    lst_stopwords: Optional[List[str]]=None, lst_tags: Optional[List[str]] = None
):
    '''
    Preprocess a string.
    :parameter
        :param text: string - name of column containing text
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    '''
    # Set the searching path of nltk
    nltk.data.path.append(pth_nltk.as_posix())
    
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    if lst_tags is not None:
        text = re.sub(rf"({'|'.join(lst_tags)})", '', text).strip().lower()
            
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in  lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word.lower()) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)

    return text