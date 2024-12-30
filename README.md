# Wikipedia Recommender System

![Python package](https://github.com/MichalRedm/wikipedia-recommender-system/actions/workflows/python-package.yml/badge.svg)

## Usage

```python
from wikirecommender import WikipediaRecommender

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

# Instantiate the recommender
recommender = WikipediaRecommender()

# Load articles into the recommender
recommender.load_articles()

# Compare a new article to the dataset
recommendations = recommender.recommend(url)

# Show top 5 recommendations
print(recommendations.head())
```

## Scripts

There are a few available scripts intended for more general use that make use of `WikipediaRecommender`. They can be found in the `./scripts` directory.

### `create_recommender.py`
Creates an instance of `WikipediaRecommender` and saves it into a `.csv` file. The user must provide the name of the output file. By default, it loads articles from 1000 pages starting from the Wikipedia Popular Pages. The starting URL and the number of pages to load can be customized using optional arguments. Usage:
```
python -m scripts.create_recommender <output_file_name> [--page-count <number_of_pages>] [--start-link <starting_url>]
```
- `<output_file_name>`: Required. The name of the output file to save the recommender.
- `--page-count`: Optional. The number of pages to load for the recommender (default: 1000).
- `--start-link`: Optional. The starting Wikipedia article URL (default: `https://en.wikipedia.org/wiki/Wikipedia:Popular_pages`).

Example:
```
python -m scripts.create_recommender recommender.csv --page-count 500 --start-link https://en.wikipedia.org/wiki/Main_Page
```

### `recommend.py`
Loads an instance of `WikipediaRecommender` from a `.csv` file (name provided by the user) and uses it to find the most similar documents to the one(s) for which a URL is provided. By default, it returns the top 5 recommendations but allows customization. Users can optionally include the provided URLs in the recommendations. Usage:
```
python -m scripts.recommend <recommender_file_name> <wikipedia_url> [<wikipedia_url> ...] [--top <number_of_recommendations>] [--include-provided]
```
- `<recommender_file_name>`: Required. The name of the file containing the recommender system.
- `<wikipedia_url>`: Required. One or more Wikipedia article URLs to generate recommendations for.
- `--top`: Optional. The number of recommendations to display (default: 5).
- `--include-provided`: Optional. Include the provided URLs in the recommendations.

Example:
```
python -m scripts.recommend recommender.csv https://en.wikipedia.org/wiki/Python_(programming_language) --top 10 --include-provided
```
