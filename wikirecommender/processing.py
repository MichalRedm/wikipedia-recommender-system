from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List

def stemmer(string: str) -> List[str]:
    """Stem the words and remove stopwords."""
    porter = PorterStemmer()
    stemmed_words = (
        porter.stem(x)
        for x in word_tokenize(string.replace('\n',' '))
    )
    return [
        x for x in stemmed_words
        if x not in stopwords.words('english') and x.isalpha()
    ]
