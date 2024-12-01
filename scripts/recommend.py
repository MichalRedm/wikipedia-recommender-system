import sys
from typing import List
from wikirecommender import WikipediaRecommender

def main(args: List[str]) -> None:
    if len(args) != 3:
        print("Usage: python recommend.py <recommender_file_name> <wikipedia_url>")
    
    filename = args[1]
    wikipedia_url = args[2]

    recommender = WikipediaRecommender.load_from_file(filename)
    result = recommender.compare_article_to_dataset(wikipedia_url)

    print(result.head())

if __name__ == "__main__":
    main(sys.argv)
