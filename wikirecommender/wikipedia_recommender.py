import re
import bs4
import tqdm
import random
import pickle
import requests
import pandas as pd
from time import sleep
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class WikipediaRecommender:
    dataset: Optional[pd.DataFrame]
    vectorizer: Optional[TfidfVectorizer]

    def __init__(self):
        self.dataset = None
        self.vectorizer = None

    def custom_stemmer(self, string: str) -> List[str]:
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

    def wikipedia_scrapper(self, link: str, page_count: int = 20, *, verbose: bool = True) -> pd.DataFrame:
        """Scrape Wikipedia articles and return a DataFrame."""
        assert link.startswith("https://en.wikipedia.org/wiki/"), "Link must start with https://en.wikipedia.org/wiki/"

        result = []
        queue = [link]
        visited = set()

        with tqdm.tqdm(total=page_count, desc="Scrapping progress", disable=(not verbose)) as pbar:
            while len(result) < page_count:
                current_link = queue.pop()
                if current_link in visited:
                    continue
                visited.add(current_link)

                # Fetch the webpage content
                response = requests.get(current_link)
                if response.status_code != 200:
                    raise Exception(f"Website returned status code {response.status_code}.")
                
                # Parse the page
                parsed = bs4.BeautifulSoup(response.text, 'html.parser')
                body_content = parsed.find('div', id='bodyContent')  # Get content from <div id="bodyContent">

                # Skip if <div id="bodyContent"> is not found
                if body_content is None:
                    continue

                # Extract links from within <div id="bodyContent">
                links = body_content.find_all('a', attrs={'href': re.compile(r'^/wiki')})
                queue[:0] = ["https://en.wikipedia.org" + link['href'] for link in links if ":" not in link['href']]

                # Add the page content to results
                if "List_of" not in current_link:
                    result.append({"wikipedia_url": current_link, "text": body_content.get_text(strip=True)})
                    pbar.update(1)

                # Pause to avoid overwhelming Wikipedia servers
                sleep(random.random() * 3)

        return pd.DataFrame(result)

    def load_articles(self, page_count: int = 20) -> pd.DataFrame:
        """Fetch and process articles to create a dataset with TF-IDF representation."""
        # Fetch Wikipedia articles
        df = self.wikipedia_scrapper("https://en.wikipedia.org/wiki/Wikipedia:Popular_pages", page_count)
        
        # Apply custom_stemmer to the text column
        df['stemmed_words'] = df['text'].apply(self.custom_stemmer)
        
        # Combine stemmed words into a single string per article for TF-IDF
        df['processed_text'] = df['stemmed_words'].apply(lambda words: " ".join(words))
        
        # Apply TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(df['processed_text'])
        
        # Convert TF-IDF matrix to a DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())
        
        # Combine the original URL with the TF-IDF DataFrame
        df = pd.concat([df[['wikipedia_url']], tfidf_df], axis=1)
        
        self.dataset = df

    def recommend(self, url: str) -> pd.DataFrame:
        """Compare a new article to the dataset using cosine similarity."""
        if self.dataset is None or self.vectorizer is None:
            raise Exception("The dataset is not loaded. Please call load_articles() first.")

        # Scrape and process the given URL
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the URL: {url} with status code {response.status_code}")
        
        # Extract the text from the <div id="bodyContent">
        parsed = bs4.BeautifulSoup(response.text, 'html.parser')
        body_content = parsed.find('div', id='bodyContent')
        if body_content is None:
            raise Exception(f"Could not find <div id='bodyContent'> in the page: {url}")
        new_text = body_content.get_text(strip=True)
        
        # Stem and process the new article text
        stemmed_words = self.custom_stemmer(new_text)
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
        }).sort_values(by='Similarity', ascending=False)
        
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
