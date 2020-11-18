import os
import re
import numpy as np
from hover.utils.common_nn import MLP
import wrappy

CACHE_PATH = os.path.join(os.path.dirname(__file__), "vecs.pkl")


def dummy_vectorizer(text):
    """
    A 'minimal' text vectorizer.
    """
    return np.random.rand(128)


@wrappy.guard(fallback_retval=dummy_vectorizer)
def get_vectorizer():
    """
    :returns: a text vectorizer.
    """
    import spacy

    # SpaCy 'vector' models are perfect for this
    # nlp = spacy.load('en_vectors_web_lg')

    # 'core' models are slower due to linguistic features
    nlp = spacy.load("en_core_web_md")

    # could use a transformer if speed is ok
    # nlp = spacy.load('en_trf_bertbaseuncased_lg')

    # memoization can be useful if the function takes a while to run, e.g. transformer models
    @wrappy.memoize(
        cache_limit=50000,
        return_copy=False,
        persist_path=CACHE_PATH,
        persist_batch_size=1000,
    )
    def vectorizer(text):
        """
        A more serious example of a text vectorizer.
        """
        clean_text = re.sub(r"[\t\n]", r" ", text)
        return nlp(clean_text, disable=nlp.pipe_names).vector

    return vectorizer


def get_architecture():
    return MLP


def get_state_dict_path():
    dir_path = os.path.dirname(__file__)
    return os.path.join(dir_path, "model.pt")
