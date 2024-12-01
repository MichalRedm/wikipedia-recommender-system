import sys
from typing import List
from wikirecommender import WikipediaRecommender

def main(args: List[str]) -> None:
    if len(args) != 2:
        print("Usage: python -m scripts.create_recommender <output_file_name>")
        exit(1)
    
    output = args[1]
    recommender = WikipediaRecommender()
    recommender.load_articles()
    recommender.save_to_file(output)
    print(f"Successfuly saved recommender system to {output}")

if __name__ == "__main__":
    main(sys.argv)
