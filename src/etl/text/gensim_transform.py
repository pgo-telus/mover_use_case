# General module
from typing import List

# module word embedding
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser, Phrases


def fit_word2vec(lst_corpus: List[List[str]]) -> Word2Vec:
    ## fit word2vec embedding to corpus
    nlp = Word2Vec(
        lst_corpus, window=8, min_count=1, sg=1, size=200, iter=30
    )

    return nlp


def fit_grams_detector(lst_corpus: List[List[str]]):
    ## Detect bigrams and trigrams
    bigrams_detector = Phrases(
        lst_corpus, delimiter=" ".encode(), min_count=5, threshold=10
    )
    bigrams_detector = Phraser(bigrams_detector)

    trigrams_detector = Phrases(
        bigrams_detector[lst_corpus], delimiter=" ".encode(), min_count=5, threshold=10
    )
    trigrams_detector = Phraser(trigrams_detector)

    return bigrams_detector, trigrams_detector
