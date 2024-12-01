import sys
from typing import List
from wikirecommender import WikipediaRecommender

def main(args: List[str]) -> None:
    if len(args) != 3:
        print("Usage: python -m scipts.recommend <recommender_file_name> <wikipedia_url>")
    
    filename = args[1]
    wikipedia_url = args[2]

    recommender = WikipediaRecommender.load_from_file(filename)
    recommendations = recommender.recommend(wikipedia_url)

    print(recommendations.head())

if __name__ == "__main__":
    main(sys.argv)
