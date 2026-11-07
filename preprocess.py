"""Text cleaning shared by the notebook (training) and predict.py (inference)."""
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Negations carry sentiment ("not good"), so they are kept even though NLTK lists them as stop words.
NEGATIONS = {
    "no", "not", "nor", "never", "don", "don't", "didn", "didn't", "doesn", "doesn't",
    "isn", "isn't", "wasn", "wasn't", "weren", "weren't", "won", "won't",
    "wouldn", "wouldn't", "couldn", "couldn't", "shouldn", "shouldn't",
    "aren", "aren't", "haven", "haven't", "hasn", "hasn't",
}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS

_stem = lru_cache(maxsize=None)(PorterStemmer().stem)


def clean_text(text) -> str:
    """Lower-case, expand n't, strip non-letters, drop stop words (keeping negations), Porter-stem."""
    s = str(text).lower()
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(_stem(w) for w in s.split() if w not in STOP_WORDS and len(w) > 1)
