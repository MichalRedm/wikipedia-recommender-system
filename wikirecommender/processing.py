from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List

def stemmer(string: str) -> List[str]:
    """
    Stem the words and remove stopwords.

    Args:
        string (str): The input string to process.

    Returns:
        List[str]: A list of stemmed words with stopwords removed.
    """
    porter = PorterStemmer()
    tokens = word_tokenize(string.replace('\n', ' '))
    
    stemmed_words = (porter.stem(x) for x in tokens)
    return [
        x for x in stemmed_words
        if x not in stopwords.words('english') and x.isalpha()
    ]
