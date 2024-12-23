import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wikirecommender.scraping import wikipedia_scrapper, wikipedia_scrapper_single_page
from wikirecommender.processing import stemmer

class WikipediaRecommender:
    dataset: pd.DataFrame
    vectorizer: TfidfVectorizer

    def __init__(self):
        self.dataset = None
        self.vectorizer = None

    def load_articles(self, page_count: int = 20, start_link="https://en.wikipedia.org/wiki/Wikipedia:Popular_pages"):
        """Fetch and process articles to create a dataset with TF-IDF representation."""
        df = wikipedia_scrapper(start_link, page_count)
        df['stemmed_words'] = df['text'].apply(stemmer)
        df['processed_text'] = df['stemmed_words'].apply(lambda words: " ".join(words))
        
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(df['processed_text'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        self.dataset = pd.concat([df[['wikipedia_url']], tfidf_df], axis=1)

    def recommend(self, url: str) -> pd.DataFrame:
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
        
        # Build the result DataFrame
        result_df = pd.DataFrame({
            'URL': self.dataset['wikipedia_url'].values,
            'Similarity': similarities
        }).sort_values(by='Similarity', ascending=False).reset_index()
        
        return result_df
    
    def save_to_file(self, filename: str) -> None:
        """Save the current instance to a file using pickle."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename: str) -> 'WikipediaRecommender':
        """Load an instance from a file using pickle."""
        with open(filename, 'rb') as file:
            return pickle.load(file)
