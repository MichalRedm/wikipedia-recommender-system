from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from typing import List

def stemmer(string: str) -> List[str]:
    """
    Stem the words, remove stopwords, and retain only valid English words.

    Args:
        string (str): The input string to process.

    Returns:
        List[str]: A list of stemmed words that are English, non-stopwords, and alphabetic.
    """
    porter = PorterStemmer()
    tokens = word_tokenize(string.replace('\n', ' '))
    english_stopwords = set(stopwords.words('english'))
    english_words = set(words.words())  # Load a set of English words for quick lookup.
    allowed_words = english_words - english_stopwords

    stemmed_words = (porter.stem(x) for x in tokens)
    return [
        x for x in stemmed_words
        if x.isalpha() and x in allowed_words
    ]
