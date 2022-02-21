import random

PSEUDO_LABELS = ["A", "B"]

VECTORIZER_BREAKER = "VECTORIZER_FALLS_SPART"


def RANDOM_LABEL(row):
    return random.choice(PSEUDO_LABELS)


def RANDOM_SCORE(row):
    return random.uniform(0.2, 1.0)
