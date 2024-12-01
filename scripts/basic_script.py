from wikirecommender import WikipediaRecommender

# Instantiate the recommender
recommender = WikipediaRecommender()

# Load articles into the dataset
dataset = recommender.load_articles()

# Compare a new article to the dataset
result = recommender.compare_article_to_dataset('https://en.wikipedia.org/wiki/Epidemiology')
print(result.head())
