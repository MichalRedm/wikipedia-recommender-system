import nltk

def download_nltk_resources():
    """Download necessary NLTK resources."""
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

# Call this when initializing the package
download_nltk_resources()
