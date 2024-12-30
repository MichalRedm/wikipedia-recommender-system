import argparse
from wikirecommender import WikipediaRecommender

def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a Wikipedia recommender system and save it to a file."
    )
    parser.add_argument(
        "output_file_name",
        type=str,
        help="The name of the output file to save the recommender system."
    )
    parser.add_argument(
        "--page-count",
        type=int,
        default=1000,
        help="Number of pages to load for the recommender system (default: 1000)."
    )
    parser.add_argument(
        "--start-link",
        type=str,
        default="https://en.wikipedia.org/wiki/Wikipedia:Popular_pages",
        help="The starting Wikipedia article URL (default: Wikipedia Popular Pages)."
    )

    # Parse the arguments
    args = parser.parse_args()
    output_file_name = args.output_file_name
    page_count = args.page_count
    start_link = args.start_link

    # Create and load the recommender system
    recommender = WikipediaRecommender()
    recommender.load_articles(page_count=page_count, start_link=start_link)
    recommender.save_to_file(output_file_name)
    
    print(f"Successfully saved recommender system to {output_file_name}")

if __name__ == "__main__":
    main()
