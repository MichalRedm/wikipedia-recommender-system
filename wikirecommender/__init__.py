"""
Wikipedia Recommender System
"""

from .recommender import WikipediaRecommender
from .utils import download_nltk_resources

# Automatically download NLTK resources when the package is loaded
download_nltk_resources()
