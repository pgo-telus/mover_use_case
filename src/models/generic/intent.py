# Global import
import transformers
from transformers.utils import logging
logging.set_verbosity(transformers.logging.CRITICAL)

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import re

# local import / global var


class IntentDetector:
    """Detect specific intent from text.

    To fit the model, a set of reference sentences are encoded using a transformer. Then a nearest neighbour 
    model is fitted on the transformed sentences with cosine metric. To score the intent of a new sentence, 
    we start by transforming the new sentence and output the minimum cosine distance using previously fitted
    nearest neighbour.  
    """
    def __init__(
        self, l_sentences: List[str], pth_model: str, detection_threshold: Optional[float] = 0.5
    ):
        """
        Args:
            l_sentences (List[str]): List of sentences that illustrate the intent we need to capture.
            pth_model (str): Path of the transformer model.
            detection_threshold (float): Optional threshold to use for hard detetction.
        """
        # Encode Intent sentences using transformer
        self.model = SentenceTransformer(pth_model)
        self.encoded_sntnces = self.model.encode(l_sentences, show_progress_bar=False)

        # Set up KNN
        self.knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
        self.knn.fit(self.encoded_sntnces, np.ones(self.encoded_sntnces.shape[0]))
        self.detection_threshold = detection_threshold
        
    def __len__(self):
        return self.encoded_sntnces.shape[0]
    
    @staticmethod
    def gather_sentences(
        d_docs: Dict[str, List[str]], split_sntce_by: str
        ) -> Tuple[Dict[str, Dict[int, List[int]]], List[str]]:
        """Gather all sentences of all docs.

        A doc isa list of chunkz, a chunk is a list of sentences. The method compute the total list 
        of sentences and a dictionnary that enable to find back sentences linked to a particular doc and 
        a particular chunk.

        Args:
            d_docs (Dict[str, List[str]]): Dict of docs, key is the id of doc. value is a list of str (chunks).
            split_sntce_by (str): The regex to use to split chunk string to get the sentences.

        Returns:
            Tuple[Dict[str, Dict[int, List[int]]], List[str]]: (doc / chunk mapping to sentence ind, 
                list of sentences).
        """
        # Split text by sentence
        d_id2ind, l_all_sentences, n_tot = {}, [], 0
        for key, l_chunks in d_docs.items():
            d_id2ind[key] = {}
            for i, chunk in enumerate(l_chunks):
                # Split chunk to get sentences
                l_sntces = [sntc for sntc in re.split(split_sntce_by, chunk) if len(sntc) > 5]
                
                # Update mapping
                d_id2ind[key][i] = [n_tot + n for n in range(len(l_sntces))]
                
                # Add to all sentences
                l_all_sentences.extend(l_sntces)
                n_tot = len(l_all_sentences)
        
        return d_id2ind, l_all_sentences
    
    @staticmethod
    def min_dist(
        knn: KNeighborsClassifier, ax_sntnces: np.ndarray, d_id2ind: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Compute the mininum cosine distance with reference intent transformed sentences.

        Args:
            knn (KNeighborsClassifier): KNN Fitted with reference intent transformed sentences.
            ax_sntnces (np.ndarray): An array of transform sentences of shape (n_sentences, n_embeddings)
            d_id2ind (Dict[str, Any]): Dict that map doc ids & chunk ids to ind of sentences.

        Returns:
            Dict[str, np.ndarray]: Dict where the key is the id of the document and the array is the minimum
                cosine distance with reference intent transformed sentences whith shape (n_chunk,)  
        """
        ax_sntce_dist, _ = knn.kneighbors(ax_sntnces, n_neighbors=1, return_distance=True)
        
        d_encodings = {}
        for key, d_map in d_id2ind.items(): 
            ax_dist = np.ones(len(d_map))
            for i, l_inds in d_map.items():
                if l_inds:  
                    ax_dist[i] = ax_sntce_dist[l_inds, 0].min()
                    
            # Udpate dict of encodings
            d_encodings[key] = ax_dist
            
        return d_encodings
    
    @staticmethod
    def all_dist(
        knn: KNeighborsClassifier, ax_sntnces: np.ndarray, d_id2ind: Dict[str, Any], 
        n_intent: int
    ) -> Dict[str, np.ndarray]:
        """Compute all cosine distance with reference intent transformed sentences.

        Args:
            knn (KNeighborsClassifier): KNN Fitted with reference intent transformed sentences.
            ax_sntnces (np.ndarray): An array of transform sentences of shape (n_sentences, n_embeddings)
            d_id2ind (Dict[str, Any]): Dict that map doc ids & chunk ids to ind of sentences.
            n_intent (int): number of reference intent sentences

        Returns:
            Dict[str, np.ndarray]: Dict where the key is the id of the document and the array host all the
                cosine distance with reference intent transformed sentences whith shape (n_chunk, n_intent)  
        """
        ax_sntce_dist, ax_inds = knn.kneighbors(
            ax_sntnces, n_neighbors=n_intent, return_distance=True
        )
        
        # For each doc create dist matrix for each chunk
        d_encodings = {}
        for key, d_map in d_id2ind.items(): 
            ax_dist = np.ones((len(d_map),n_intent))
            for i, l_inds in d_map.items():
                if not l_inds:
                    continue
                    
                for j in range(n_intent):
                    ax_dist[i, j] = ax_sntce_dist[l_inds, :][ax_inds[l_inds, :] == j].min()
                    
            # Udpate dict of encodings
            d_encodings[key] = ax_dist
            
        return d_encodings    
    
    def min_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.'
    ) -> Dict[str, np.ndarray] :
        """Compute the mininum cosine distance with reference intent transformed sentences for multiple docs.

        Documents are passed as dict with an identification key. The list of string linked to a key correspond 
        to different chunks of a document. First we split every chunk into sentences keep a trace of a mapping 
        between (document key, chunk ind) and sentences. Then we transformed all sentences. Finally we compute 
        the minimum cosine distance of chunks with the reference intent sentences.

        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. Defaults to '\.'.

        Returns:
            Dict[str, np.ndarray]: Dict where the key is the id of the document and the array is the minimum
                cosine distance with reference intent transformed sentences whith shape (n_chunk,)  
        """
        # Gather sentences
        d_id2ind, l_all_sentences = self.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces = self.model.encode(l_all_sentences, show_progress_bar=False)
        d_encodings = self.min_dist(self.knn, ax_sntnces, d_id2ind)

        return d_encodings
        
    def all_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
    ) -> bool:
        """Compute all cosine distance with reference intent transformed sentences for multiple docs.

        Documents are passed as dict with an identification key. The list of string linked to a key correspond 
        to different chunks of a document. First we split every chunk into sentences keep a trace of a mapping 
        between (document key, chunk ind) and sentences. Then we transformed all sentences. Finally we compute 
        all cosine distance of chunks with the reference intent sentences.

        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. 
                Defaults to '\.'.

        Returns:
            Dict[str, np.ndarray]: Dict where the key is the id of the document and the array is all
                cosine distance with reference intent transformed sentences whith shape (n_chunk, n_intent)  
        """
        n_intent = self.encoded_sntnces.shape[0]
        
        # Gather sentences
        d_id2ind, l_all_sentences = self.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces = self.model.encode(l_all_sentences, show_progress_bar=False)        
        d_encodings = self.all_dist(self.knn, ax_sntnces, d_id2ind, n_intent)
        
        return d_encodings
    
    
class MultiIntentDetector:
    """Detect multiple intent from text. 

    The way it works is similar to IntentDetector, but it allowed to encode more than 1 intent.
    """
    def __init__(
        self, d_sentences: Dict[str, str], pth_model: str, detection_threshold: Optional[float] = 0.5
    ):
        """_summary_

        Args:
            d_sentences (Dict[str, str]): Dict where the key identify the intent and the values is a list of
                sentences that illustrate the intent.
            pth_model (str): _description_
            detection_threshold (Optional[float], optional): _description_. Defaults to 0.5.
        """
        # Encode Intent sentences using transformer
        self.model = SentenceTransformer(pth_model)
        self.encoded_sntnces = {
            k: self.model.encode(l_sentences, show_progress_bar=False)
            for k, l_sentences in d_sentences.items()
        }

        # Set up KNN
        self.knn = {}
        for k, l_sentences in d_sentences.items():
            knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
            knn.fit(self.encoded_sntnces[k], np.ones(self.encoded_sntnces[k].shape[0]))
            self.knn[k] = knn

        self.detection_threshold = detection_threshold
        
    def shape(self):
        return {k: v.shape[0] for k, v in self.encoded_sntnces.items()}
    
    def min_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
        debug=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute the mininum cosine distance with reference intent transformed sentences for multiple docs.

        The method relies heavily on IntentDetector.min_dist. 

        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. Defaults to '\.'.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Each key of the dict refer to an intent. For a particular intent the values is
                a dict where the key is the id of the document and the array is the minimum
                cosine distance with reference intent transformed sentences whith shape (n_chunk,)  
        """
        # Gather sentences
        d_id2ind, l_all_sentences = IntentDetector.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces, d_encodings = self.model.encode(l_all_sentences, show_progress_bar=False), {}
        for k, knn in self.knn.items():
            d_encodings[k] = IntentDetector.min_dist(knn, ax_sntnces, d_id2ind)
            
        return d_encodings
        
    def all_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
        debug=False
    ) -> Dict[str, Dict[str, np.ndarray]]:    
        """Compute all cosine distance with reference intent transformed sentences for multiple docs.

        The method relies heavily on IntentDetector.min_dist.
        
        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. 
                Defaults to '\.'.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Each key of the dict refer to an intent. For a particular intent the values is
                a Dict where the key is the id of the document and the array is all cosine distance with 
                reference intent transformed sentences whith shape (n_chunk, n_intent)  
        """    
        # Gather sentences
        d_id2ind, l_all_sentences = IntentDetector.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces, d_encodings = self.model.encode(l_all_sentences, show_progress_bar=False), {}    
        for k, knn in self.knn.items():
            n_intent = self.encoded_sntnces[k].shape[0]
            d_encodings[k] = IntentDetector.all_dist(knn, ax_sntnces, d_id2ind, n_intent)
        
        return d_encodings
    

class IntentComp:
    """Compare specific intent from text.

    To fit the model, a set of reference sentences encoding 2 different intents are encoded using a transformer. 
    Each intent are usually representing 2 opposite intent and are separate with a '/' char 
    (i.e 'I am not happy / I am happy'). The sentences lying at the left side is refer as the positive intent, 
    the one on the right is the negative. Two nearest neighbour models are fitted on positive and negative 
    transformed sentences using cosine metric. 
    To score the relative intent of a new sentence,  we start by transforming the new sentence, then output the 
    minimum cosine distance using previously fitted nearest neighbours. The final score is obtained by the 
    formula (1 + dist_positive - dist_negative) / 2 so that the result lies between 0 and 1. A score close 
    to 0 means that dist_positive is lower than dist_negative, a score close to 1 means the opposite.

    i.e: ref sentence:
        "I am not happy / I am happy"
    
    the sentence of interest are splitted by the '/' char.
    new sentence:
        "I am disappointed"

    the cosine dist with "I am not happy" is 0.1, the cosine dist with "I am happy" is 0.6.
    The final intent dist is 
        (1 + 0.1 - 0.6) / 2 = 0.25
    """
    def __init__(
        self, l_sentences: List[str], pth_model: str, detection_threshold: Optional[float] = 0
    ):
        """_summary_

        Args:
            l_sentences (List[str]): List of sentence. Intents are sep by a '/' char.
            pth_model (str): Path of the transformer model.
            detection_threshold (float): Optional threshold to use for hard detetction.
        """
        # Encode Intent sentences using transformer
        self.model = SentenceTransformer(pth_model)
        self.encoded_pos_sntnces = self.model.encode(
            [sntnce.split('/')[0] for sntnce in l_sentences], show_progress_bar=False
        )
        self.encoded_neg_sntnces = self.model.encode(
            [sntnce.split('/')[1] for sntnce in l_sentences], show_progress_bar=False
        )

        # Set up KNN
        self.pos_knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
        self.neg_knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
        
        self.pos_knn.fit(self.encoded_pos_sntnces, np.ones(self.encoded_pos_sntnces.shape[0]))
        self.neg_knn.fit(self.encoded_neg_sntnces, np.ones(self.encoded_neg_sntnces.shape[0]))
        self.detection_threshold = detection_threshold
        
    def __len__(self):
        return self.encoded_pos_sntnces.shape[0]
    
    def min_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
        debug=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute the mininum relative cosine distance with reference intent transformed sentences for multiple docs.

        The method relies heavily on IntentDetector.min_dist. here the minimum distance is the cosine distance
        of 'positive' intent relatively to the negative intent.  

        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. Defaults to '\.'.

        Returns:
            Dict[str, np.ndarray]: Dict where the key is the id of the document and the array is the minimum
                relative cosine distance with reference intent transformed sentences whith shape (n_chunk,)  
        """
        # Gather sentences
        d_id2ind, l_all_sentences = IntentDetector.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces, d_encodings = self.model.encode(l_all_sentences, show_progress_bar=False), {}
        pos_encodings = IntentDetector.min_dist(self.pos_knn, ax_sntnces, d_id2ind)
        neg_encodings = IntentDetector.min_dist(self.neg_knn, ax_sntnces, d_id2ind)
        for k, ax_pos in pos_encodings.items():
            d_encodings[k] = (1 + ax_pos - neg_encodings[k]) / 2
            
        return d_encodings
        
    def all_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
        debug=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute all relative cosine distance with reference intent transformed sentences for multiple docs.

        The method relies heavily on IntentDetector.all_dist. here the minimum distance is the cosine distance
        of 'positive' intent relatively to the negative intent. 

        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. 
                Defaults to '\.'.

        Returns:
            Dict[str, np.ndarray]: Dict where the key is the id of the document and the array is all
                relative cosine distance with reference intent transformed sentences whith shape (n_chunk, n_intent)  
        """
        n_intent = self.encoded_pos_sntnces.shape[0]
        
        # Gather sentences
        d_id2ind, l_all_sentences = IntentDetector.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces, d_encodings = self.model.encode(l_all_sentences, show_progress_bar=False), {}
        pos_encodings = IntentDetector.all_dist(self.pos_knn, ax_sntnces, d_id2ind, n_intent)
        neg_encodings = IntentDetector.all_dist(self.neg_knn, ax_sntnces, d_id2ind, n_intent)
        for k, ax_pos in pos_encodings.items():
            d_encodings[k] = (1 + ax_pos - neg_encodings[k]) / 2


class MultiIntentComp:
    """Compare specific intent from text. 

    The way it works is similar to IntentComp, but it allowed to make more than 1 intent comparison.
    """
    def __init__(
        self, d_sentences: Dict[str, List[str]], pth_model: str, detection_threshold: Optional[float] = 0
    ):
        """_summary_

        Args:
            d_sentences (Dict[str, str]): Dict where the key identify the intent and the values is a list of
                sentences that illustrate the intent.
            pth_model (str): _description_
            detection_threshold (Optional[float], optional): _description_. Defaults to 0.5.
        """
        # Encode Intent sentences using transformer
        self.model = SentenceTransformer(pth_model)
        
        self.encoded_pos_sntnces = {
            k: self.model.encode(
                [sntnce.split('/')[0] for sntnce in l_sentences], show_progress_bar=False
            ) for k, l_sentences in d_sentences.items()
        }
        self.encoded_neg_sntnces = {
            k: self.model.encode(
                [sntnce.split('/')[1] for sntnce in l_sentences], show_progress_bar=False
            ) for k, l_sentences in d_sentences.items()
        }
        
        # Set up KNN
        self.pos_knn = {}
        for k, ax_sntnces in self.encoded_pos_sntnces.items():
            knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
            knn.fit(ax_sntnces, np.ones(self.encoded_pos_sntnces[k].shape[0]))
            self.pos_knn[k] = knn
            
        self.neg_knn = {}
        for k, ax_sntnces in self.encoded_neg_sntnces.items():
            knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
            knn.fit(ax_sntnces, np.ones(self.encoded_neg_sntnces[k].shape[0]))
            self.neg_knn[k] = knn
            
        self.detection_threshold = detection_threshold
        
    def shape(self):
        return {k: v.shape[0] for k, v in self.encoded_pos_sntnces.items()}
    
    def min_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
        debug=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute the mininum cosine distance with reference intent transformed sentences for multiple docs.

        The method relies heavily on IntentDetector.min_dist. 

        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. Defaults to '\.'.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Each key of the dict refer to an intent. For a particular 
                intent the values is a dict where the key is the id of the document and the array is 
                the minimum relative cosine distance with reference intent transformed sentences 
                whith shape (n_chunk,)  
        """
        # Gather sentences
        d_id2ind, l_all_sentences = IntentDetector.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces, d_encodings = self.model.encode(l_all_sentences, show_progress_bar=False), {}
        
        for k, pos_knn in self.pos_knn.items():
            pos_encodings = IntentDetector.min_dist(pos_knn, ax_sntnces, d_id2ind)
            neg_encodings = IntentDetector.min_dist(self.neg_knn[k], ax_sntnces, d_id2ind)
            d_sub_encodings = {}
            for did, ax_pos in pos_encodings.items():
                d_sub_encodings[did] = (1 + ax_pos - neg_encodings[did]) / 2
                
            d_encodings[k] = d_sub_encodings
            
        return d_encodings
        
    def all_dist_multi_doc(
        self, d_docs: Dict[str, List[str]], split_sntce_by: Optional[str] = '\.', 
        debug=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute all cosine distance with reference intent transformed sentences for multiple docs.

        The method relies on IntentDetector.all_dist.
        
        Args:
            d_docs (Dict[str, List[str]]): Dict of documents
            split_sntce_by (Optional[str], optional): regex that determine how to split chunks into sentences. 
                Defaults to '\.'.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Each key of the dict refer to an intent. For a particular intent 
                the values is a Dict where the key is the id of the document and the array is all relative cosine 
                distance with reference intent transformed sentences whith shape (n_chunk, n_intent)  
        """    
        # Gather sentences
        d_id2ind, l_all_sentences = IntentDetector.gather_sentences(d_docs, split_sntce_by)
        
        # encode sentences
        ax_sntnces, d_encodings = self.model.encode(l_all_sentences, show_progress_bar=False), {}        
        for k, pos_knn in self.pos_knn.items():
            n_intent = self.encoded_pos_sntnces[k].shape[0]
            pos_encodings = IntentDetector.all_dist(pos_knn, ax_sntnces, d_id2ind, n_intent)
            neg_encodings = IntentDetector.all_dist(self.neg_knn[k], ax_sntnces, d_id2ind, n_intent)
            d_sub_encodings = {}
            for did, ax_pos in pos_encodings.items():
                d_sub_encodings[did] = (1 + ax_pos - neg_encodings[did]) / 2
                
            d_encodings[k] = d_sub_encodings
            
        return d_encodings
