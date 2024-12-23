import sys
from typing import List
from wikirecommender import WikipediaRecommender

def main(args: List[str]) -> None:
    if len(args) < 3 or len(args) > 4:
        print("Usage: python -m scripts.recommend <recommender_file_name> <wikipedia_url> [top_n]")
        exit(1)

    # Required arguments
    filename = args[1]
    wikipedia_url = args[2]
    
    # Optional argument with default value
    top_n = int(args[3]) if len(args) == 4 else 5

    # Load recommender and generate recommendations
    recommender = WikipediaRecommender.load_from_file(filename)
    recommendations = recommender.recommend(wikipedia_url)

    # Display top N recommendations
    top_recommendations = recommendations.head(top_n)
    print(top_recommendations)

if __name__ == "__main__":
    main(sys.argv)
