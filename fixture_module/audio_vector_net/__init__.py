"""
Example importable module holding customized ingredients of a workflow with hover.
Specifically for audio data in URLs.
"""

import os
import re
import numpy as np
import wrappy
import requests
import librosa
from io import BytesIO


DIR_PATH = os.path.dirname(__file__)
RAW_CACHE_PATH = os.path.join(DIR_PATH, "raws.pkl")
AUD_CACHE_PATH = os.path.join(DIR_PATH, "auds.pkl")
VEC_CACHE_PATH = os.path.join(DIR_PATH, "vecs.pkl")


@wrappy.memoize(
    cache_limit=50000,
    return_copy=False,
    persist_path=RAW_CACHE_PATH,
    persist_batch_size=100,
)
def url_to_content(url):
    """
    Turn a URL to response content.
    """
    response = requests.get(url)
    return response.content


@wrappy.memoize(
    cache_limit=50000,
    return_copy=False,
    persist_path=AUD_CACHE_PATH,
    persist_batch_size=100,
)
def url_to_audio(url):
    """
    Turn a URL to audio data.
    """
    data, sampling_rate = librosa.load(BytesIO(url_to_content(url)))
    return data, sampling_rate


def get_vectorizer():
    @wrappy.memoize(
        cache_limit=50000,
        return_copy=False,
        persist_path=VEC_CACHE_PATH,
        persist_batch_size=100,
    )
    def vectorizer(url):
        """
        Averaged MFCC over time.
        Resembles word-embedding-average-as-doc-embedding for texts.
        """
        y, sr = url_to_audio(url)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
        return mfcc.mean(axis=1)

    return vectorizer


def get_architecture():
    from hover.utils.common_nn import LogisticRegression

    return LogisticRegression


def get_state_dict_path():
    return os.path.join(DIR_PATH, "model.pt")
