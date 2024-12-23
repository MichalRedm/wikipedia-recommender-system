import sys
from typing import List
from wikirecommender import WikipediaRecommender

def main(args: List[str]) -> None:
    if len(args) < 2:
        print("Usage: python -m scripts.create_recommender <output_file_name> [page_count] [start_link]")
        exit(1)
    
    # Required argument
    output = args[1]
    
    # Optional arguments with default values
    page_count = int(args[2]) if len(args) > 2 else 20
    start_link = args[3] if len(args) > 3 else "https://en.wikipedia.org/wiki/Wikipedia:Popular_pages"
    
    # Create and load the recommender system
    recommender = WikipediaRecommender()
    recommender.load_articles(page_count=page_count, start_link=start_link)
    recommender.save_to_file(output)
    
    print(f"Successfully saved recommender system to {output}")

if __name__ == "__main__":
    main(sys.argv)
