"""
detector.py  -  Paraphrase detection logic
Method : WordNet synonym normalisation + TF-IDF + Cosine Similarity
"""
import math
import os
import re
from collections import Counter

import nltk

# --------------------------------------------------------------------------
# NLTK data setup
# Vercel's file system is read-only except /tmp, so on Vercel the NLTK data
# is downloaded to /tmp/nltk_data the first time the function starts.
# On your own PC it is downloaded to the normal NLTK folder.
# --------------------------------------------------------------------------
if os.environ.get("VERCEL"):
    NLTK_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DIR, exist_ok=True)
    nltk.data.path.insert(0, NLTK_DIR)
else:
    NLTK_DIR = None  # default location


def _ensure(resource_path, package):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, download_dir=NLTK_DIR, quiet=True)


_ensure("corpora/wordnet", "wordnet")
_ensure("corpora/omw-1.4", "omw-1.4")
_ensure("corpora/stopwords", "stopwords")

from nltk.corpus import stopwords, wordnet  # noqa: E402

stop_words = set(stopwords.words("english"))

# Same tokenisation rule that scikit-learn's TfidfVectorizer uses by default
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
def get_synonym(word):
    """Return the first WordNet lemma of the word (or the word itself)."""
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    if synonyms:
        return synonyms[0].replace("_", " ")
    return word


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    processed_words = []
    for word in words:
        if word not in stop_words:
            processed_words.append(get_synonym(word))
    return " ".join(processed_words)


# --------------------------------------------------------------------------
# TF-IDF + Cosine similarity (same maths as sklearn's TfidfVectorizer,
# written in plain Python so the app stays small enough for Vercel)
# --------------------------------------------------------------------------
def tfidf_vectors(documents):
    """Return (vocabulary, list of L2-normalised TF-IDF vectors)."""
    tokenised = [TOKEN_PATTERN.findall(d.lower()) for d in documents]
    vocabulary = sorted(set(t for doc in tokenised for t in doc))
    n_docs = len(documents)

    df = {t: sum(1 for doc in tokenised if t in doc) for t in vocabulary}
    # smooth idf:  ln((1 + n) / (1 + df)) + 1
    idf = {t: math.log((1 + n_docs) / (1 + df[t])) + 1 for t in vocabulary}

    vectors = []
    for doc in tokenised:
        counts = Counter(doc)
        vec = [counts.get(t, 0) * idf[t] for t in vocabulary]
        norm = math.sqrt(sum(v * v for v in vec))
        vectors.append([v / norm for v in vec] if norm else vec)
    return vocabulary, vectors


def calculate_similarity(sentence1, sentence2):
    text1 = preprocess_text(sentence1)
    text2 = preprocess_text(sentence2)
    if not text1 or not text2:
        return 0.0, text1, text2
    _, (v1, v2) = tfidf_vectors([text1, text2])
    similarity = sum(a * b for a, b in zip(v1, v2))
    return min(max(similarity, 0.0), 1.0), text1, text2


def detect_paraphrase(sentence1, sentence2, threshold=0.30):
    similarity, text1, text2 = calculate_similarity(sentence1, sentence2)
    result = "Paraphrase" if similarity >= threshold else "Not a Paraphrase"
    return result, similarity, text1, text2
