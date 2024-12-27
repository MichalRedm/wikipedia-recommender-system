import argparse
import pandas as pd
from wikirecommender import WikipediaRecommender

def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Wikipedia article recommendations.")
    parser.add_argument(
        "recommender_file_name",
        type=str,
        help="The name of a CSV file from which the recommender data should be loaded."
    )
    parser.add_argument(
        "wikipedia_urls",
        nargs="+",  # Allows one or more URLs to be passed
        help="One or more Wikipedia article URLs for which recommendations should be provided."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top recommended articles to display (default: 5)."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    filename = args.recommender_file_name
    wikipedia_urls = args.wikipedia_urls
    top_n = args.top_n

    # Load recommender and generate recommendations
    recommender = WikipediaRecommender.load_from_file(filename)
    recommendations = recommender.recommend(wikipedia_urls)

    # Display top N recommendations
    top_recommendations = recommendations.head(top_n)
    pd.set_option('display.max_colwidth', None)  # Ensure full URLs are displayed
    print(top_recommendations)

if __name__ == "__main__":
    main()
