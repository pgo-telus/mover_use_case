# Global import
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List

# Local import 
from core.etl.text.nltk_transform import transform_text


def build_training_data(df: pd.DataFrame, pth_nltk_data: Path, l_stopwords: List[str] = None, l_tags: List[str] = None):
    # clean text data
    df["text_clean"] = df['sntnce'].apply(
            lambda x: transform_text(x, pth_nltk_data, flg_lemm=True, lst_stopwords=l_stopwords, lst_tags=l_tags)
    )

    # check on class imbalance and get minimum
    min_class = df['movehit'].value_counts().min()
    df = df.groupby('movehit').sample(n=min_class, random_state=1)

    # select relevant fields
    df = df[['text_clean', 'movehit']]
    df.columns = list('xy')

    # split dataset to train and test
    df_train, df_test = train_test_split(df, test_size=0.3, shuffle=True)

    # get target
    y_train = df_train["y"].values
    y_test = df_test["y"].values

    return df, df_train, df_test, y_train, y_test