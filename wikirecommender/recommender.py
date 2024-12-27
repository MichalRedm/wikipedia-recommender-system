import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from typing import List, Union

from wikirecommender.scraping import wikipedia_scrapper, wikipedia_scrapper_single_page
from wikirecommender.processing import stemmer

class WikipediaRecommender:
    dataset: pd.DataFrame
    vectorizer: TfidfVectorizer

    def __init__(self):
        self.dataset = None
        self.vectorizer = None

    def load_articles(self, page_count: int = 20, start_link="https://en.wikipedia.org/wiki/Wikipedia:Popular_pages", verbose: bool = True) -> None:
        """Fetch and process articles to create a dataset with TF-IDF representation."""
        df = wikipedia_scrapper(start_link, page_count, verbose=verbose)
        if verbose:
            tqdm.pandas(desc="Processing articles")
            df['stemmed_words'] = df['text'].progress_apply(stemmer)
        else:
            df['stemmed_words'] = df['text'].apply(stemmer)
        df['processed_text'] = df['stemmed_words'].apply(lambda words: " ".join(words))
        
        if verbose:
            print("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(df['processed_text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        self.dataset = pd.concat([df[['wikipedia_url']], tfidf_df], axis=1)
    
    def _get_similarities(self, url: str) -> np.ndarray:
        """Compare a new article to the dataset using cosine similarity."""
        if self.dataset is None or self.vectorizer is None:
            raise Exception("The dataset is not loaded. Please call load_articles() first.")

        # Scrape and process the given URL
        new_text = wikipedia_scrapper_single_page(url)
        
        # Stem and process the new article text
        stemmed_words = stemmer(new_text)
        new_article_text = " ".join(stemmed_words)
        
        # Transform the new article into TF-IDF using the existing vectorizer
        new_article_tfidf = self.vectorizer.transform([new_article_text]).toarray()
        
        # Ensure TF-IDF matrix includes only numerical columns
        tfidf_columns = self.vectorizer.get_feature_names_out()  # Get the feature names from the vectorizer
        existing_tfidf_matrix = self.dataset[tfidf_columns].values  # Select only TF-IDF feature columns

        # Compute cosine similarity between the new article and the dataset
        similarities = cosine_similarity(new_article_tfidf, existing_tfidf_matrix).flatten()
        
        return similarities

    def recommend(self, url: Union[str, List[str]]) -> pd.DataFrame:
        """Compare a new article to the dataset using cosine similarity."""
        similarities = None

        if isinstance(url, str):
            similarities = self._get_similarities(url)
        elif isinstance(url, list):
            if len(url) == 0:
                raise ValueError("URL list cannot be empty.")
            similarities_list = np.array([self._get_similarities(u) for u in url])
            similarities = similarities_list.mean(axis=0)
        else:
            raise ValueError("URL must be a string or a list of strings.")
        
        # Build the result DataFrame
        result_df = pd.DataFrame({
            'URL': self.dataset['wikipedia_url'].values,
            'Similarity': similarities
        }).sort_values(by='Similarity', ascending=False)

        result_df.index = pd.RangeIndex(start=1, stop=len(result_df) + 1, step=1)
        
        return result_df
    
    def save_to_file(self, filename: str) -> None:
        """Save the current instance to a CSV file."""
        # Ensure the filename has the correct extension
        if not filename.lower().endswith('.csv'):
            filename += '.csv'

        vocabulary = self.vectorizer.vocabulary_
        idf_values = self.vectorizer.idf_

        # Convert to DataFrame
        df = pd.DataFrame(
            [(term, index, idf_values[index]) for term, index in vocabulary.items()],
            columns=['term', 'index', 'idf']
        )
        df.set_index('term', inplace=True)
        df = df.transpose()
        df = pd.concat([df, self.dataset], axis=0)

        df.set_index('wikipedia_url', inplace=True)
            
        df.to_csv(filename)

    @staticmethod
    def load_from_file(filename: str) -> 'WikipediaRecommender':
        """Load an instance from a CSV file."""
        # Ensure the filename has the correct extension
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
    
        df = pd.read_csv(filename, index_col='wikipedia_url')

        vocabulary = dict(zip(df.columns, df.iloc[0].astype("int")))
        metadata = df.iloc[:2]
        metadata.index = pd.Index(['index', 'idf'])
        idf_values = metadata.transpose().sort_values(by='index')['idf'].values
        dataset = df.iloc[2:]

        recommender = WikipediaRecommender()
        recommender.vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        recommender.vectorizer.idf_ = idf_values
        recommender.dataset = dataset.reset_index()

        return recommender

