# Wikipedia Recommender System

![Python package](https://github.com/MichalRedm/wikipedia-recommender-system/actions/workflows/python-package.yml/badge.svg)

## Usage

```python
from wikirecommender import WikipediaRecommender

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

# Instantiate the recommender
recommender = WikipediaRecommender()

# Load articles into the dataset
dataset = recommender.load_articles()

# Compare a new article to the dataset
recommendations = recommender.recommend(url)

# Show top 5 recommendations
print(recommendations.head())
```

## Scripts

There are a few available scripts intended for more general use that make use of `WikipediaRecommender`. They can be found in `./scripts` directory.

### `create_recommender.py`
Creates an instance of `WikipediaRecommender` and saves it into a `.pickle` file. The user should provide the name of the output file. Usage:
```
python -m scripts.create_recommender <output_file_name>
```

### `recommend.py`
Loads an instance of `WikipediaRecommender` from a `.pickle` file (name provided by the user) and uses it to find most similar documents to the one for which a URL is provided. Usage:
```
python -m scripts.recommend <recommender_file_name> <wikipedia_url>
```
