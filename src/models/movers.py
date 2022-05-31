# Global import
from typing import List, Dict, Any, Optional
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

# Local import
from core.etl.extract import extract_dir_from_bucket
from core.etl.movers.transform.features import extract_convs_features
from core.etl.text.transform import get_match_regex, is_match_regex
from core.models.generic.clf import LRClassifier
from core.models.generic.intent import IntentDetector


class MoverDetector:
    """Classifier whether or not a text refers to a moving intent (Logistic Regression)."""
    intent_offset = 2
    def __init__(
        self, d_regex: Dict[str, List[str]], 
        classifier: LRClassifier, 
        intent_model: IntentDetector, 
        d_stopwords: Optional[Dict[str, bool]] = None, 
        l_tags: Optional[List[str]] = None
    ):
        self.regex = d_regex
        self.clf = classifier
        self.intent_model = intent_model
        self.stopwords = d_stopwords
        self.tags = l_tags
        
    def detect(self, df_convs: pd.DataFrame, pth_tr_model: str, conv_id_col: str):

         # Add a unique chunk_id to all sentences 
        df_convs = df_convs.sort_values(by=['call_convrstn_id', 'sntnce_ts'])\
            .assign(chunk_id=np.arange(len(df_convs)))\
            .reset_index(drop=True)

        # Transform conv wher match occurs
        dataset = extract_convs_features(
            df_convs, self.regex, self.intent_model, d_stopwords=self.stopwords, 
            l_tags=self.tags
        )
        
        # Get proba from classifier
        ax_move_score = self.clf.predict_proba(dataset.X)[:, 1]
        
        # Gather detected
        l_detected = []
        for ind, d_ids in dataset.ind2ids.items():
            if ax_move_score[ind] <= 0.5:
                continue
            
            # Is there a cancellation metnioned early in the conversation
            df_sub_conv = df_convs.loc[df_convs['call_convrstn_id'] == d_ids['conv_id']] 
            l_cancellation = [
                is_match_regex(s, self.regex['cancel']) for s in df_sub_conv.sntnce.iloc[:10]
            ]
            if any(l_cancellation):
                continue         
                
            # Save result
            l_detected.append({'detection_score': ax_move_score[ind], **d_ids})
        
        # Return conv where
        df_detected = pd.DataFrame(l_detected).groupby('conv_id')\
            .agg({'detection_score': 'max'})\
            .reset_index()
        
        return df_detected
    
    
class NaiveMoverDetector:
    """Naive detector of move intent (based on keywords)."""
    intent_offset = 2
    def __init__(
        self, d_regex: Dict[str, List[str]], 
        d_stopwords: Optional[Dict[str, bool]] = None, 
        l_tags: Optional[List[str]] = None
    ):
        self.regex = d_regex
        self.stopwords = d_stopwords
        self.tags = l_tags
        
    def detect(self, df_convs: pd.DataFrame, conv_id_col: str):
        l_detected = []
        for _, row in df_convs.iterrows(): 
            l_move_match = get_match_regex(row['sntnce'], self.regex['movers'])

            if not l_move_match or row['sntnce_partcpnt_role'] != 'END_USER':
                continue
            
            l_cancel_match = get_match_regex(row['sntnce'], self.regex['cancel'])
            l_holiday_match = get_match_regex(row['sntnce'], self.regex['holiday'])
            l_expression_match = get_match_regex(row['sntnce'], self.regex['expression'])
            l_things_match = get_match_regex(row['sntnce'], self.regex['things'])
    
            if l_cancel_match or l_expression_match or l_holiday_match or l_things_match:
                continue
                
            l_detected.append({
                'conv_id': row['call_convrstn_id'], 'detection_score': 1.
            })
        
        # Return conv where
        df_detected = pd.DataFrame(l_detected).groupby('conv_id')\
            .agg({'detection_score': 'max'})\
            .reset_index()
        
        return df_detected


class MoveDeepClf():

    def __init__(
        self, max_features: int, sequence_length: int, name_input: str = 'text', 
        embeddings: Optional[np.ndarray] = None, vocab: Optional[np.ndarray] = None, 
        pth_weights: Optional[Path] = None
    ):
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.name_input = name_input
        self.model = None
        self.embeddings = embeddings
        self.vocab = vocab
        
        if pth_weights is not None:
            self.build_keras_model(pth_weights)
        
    @staticmethod
    def extract_from_bucket(
        bucket: Any, path: Path, model_name: str, max_features: int, 
        sequence_length: int, name_input: str = 'text'
    ):
        local_path = Path(f'/tmp/bucket_content') / model_name
        extract_dir_from_bucket(bucket, local_path, (path / model_name).as_posix())
        
        with open(local_path / 'embeddings.pickle', 'rb') as f:
            embeddings = pickle.load(f)
        
        with open(local_path / 'vocab.pickle', 'rb') as f:
            vocab = pickle.load(f)
            
        return MoveDeepClf(
            max_features, sequence_length, name_input=name_input, embeddings=embeddings, 
            vocab=vocab, pth_weights=local_path / 'weights' / 'data'
        )

    @staticmethod
    def attention_layer(inputs, neurons):
        x = layers.Permute((2,1))(inputs)
        x = layers.Dense(neurons, activation="softmax")(x)
        x = layers.Permute((2,1), name="attention")(x)
        x = layers.multiply([inputs, x])
        return x
        
    def build_attention_network(self, embeddings: np.ndarray):
        # Input
        x_in = layers.Input(shape=(self.sequence_length,), dtype=tf.int32, ragged=False)

        # Embedding layer
        x = layers.Embedding(
            input_dim=embeddings.shape[0],
            output_dim=embeddings.shape[1], 
            weights=[embeddings],
            input_length=50, trainable=False
        )(x_in)

        ## apply attention
        x = self.attention_layer(x, neurons=50)

        ## 2 layers of bidirectional lstm
        x = layers.Bidirectional(layers.LSTM(units=50, dropout=0.2, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(units=50, dropout=0.2))(x)

        ## final dense layers
        x = layers.Dense(256, activation='relu')(x)
        y_out = layers.Dense(1, activation='sigmoid')(x)

        ## compile
        model = models.Model(x_in, y_out, name='prediction')
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        return model
    
    def build_keras_model(self, pth_weights: Optional[Path] = None):
      
        vectorize_layer = layers.TextVectorization(
            vocabulary=self.vocab,  
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length, 
            ragged=False
        )

        self.model = tf.keras.Sequential([
                tf.keras.Input(shape=(1,), dtype=tf.string, name='text_input', ragged=False),
                vectorize_layer, self.build_attention_network(self.embeddings)
            ], name="end_model"
        )
        # set in/out layer names
        self.model.layers[0]._name = 'text'
        
        # compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Load weight if path passed
        if pth_weights is not None:
            self.model.load_weights(pth_weights.as_posix())
            
        print(self.model.summary())

    def fit_model(self, train_text: np.ndarray, y_train: np.ndarray, epochs: int): 
        self.model.fit(
            x=train_text, y=y_train,
            batch_size=64, shuffle=True, verbose=1, 
            epochs=epochs
        )

    def predict(self, x: List[str]):
        return self.model.predict(x)
    
    def save_to_bucket(self, bucket: Any, model_name: str, remote_path: Path):
        
        # Save model locally (embeddings & weights
        local_path = Path(f'/tmp/bucket_content')
        with (local_path / model_name / 'embeddings.pickle').open(mode='wb') as f:
            pickle.dump(self.embeddings, f)
        with (local_path / model_name / 'vocab.pickle').open(mode='wb') as f:
            pickle.dump(self.vocab, f)
            
        self.model.save_weights((local_path / model_name / 'weights'/ 'data').as_posix())
        
        # Walk directory tree and save files to bucket
        l_file_path = []
        for r, d, f in os.walk(local_path / model_name):
            for file in f:
                remote_path_sub = remote_path / os.path.relpath(os.path.join(r, file), local_path)
                bucket.blob(remote_path_sub.as_posix()).upload_from_filename(os.path.join(r, file))    
    