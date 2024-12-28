import re
import random
import requests
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

def wikipedia_scrapper_single_page(url: str) -> str:
    # Scrape and process the given URL
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {url} with status code {response.status_code}")
    
    # Extract the text from the <div id="bodyContent">
    parsed = BeautifulSoup(response.text, 'html.parser')
    body_content = parsed.find('div', id='bodyContent')
    if body_content is None:
        raise Exception(f"Could not find <div id='bodyContent'> in the page: {url}")
    new_text = body_content.get_text(strip=True)

    return new_text

def wikipedia_scrapper(link: str, page_count: int = 20, *, verbose: bool = True, min_sleep_time: float = 0.25, max_sleep_time: float = 0.5) -> pd.DataFrame:
    """Scrape Wikipedia articles and return a DataFrame."""
    assert link.startswith("https://en.wikipedia.org/wiki/"), "Link must start with https://en.wikipedia.org/wiki/"

    result = []
    queue = [link]
    visited = set()

    with tqdm(total=page_count, desc="Scraping articles", disable=(not verbose)) as pbar:
        while len(result) < page_count:
            current_link = queue.pop()
            if current_link in visited:
                continue
            visited.add(current_link)

            response = requests.get(current_link)
            if response.status_code != 200:
                raise Exception(f"Website returned status code {response.status_code}.")
            
            parsed = BeautifulSoup(response.text, 'html.parser')
            body_content = parsed.find('div', id='bodyContent')

            if body_content is None:
                continue

            links = body_content.find_all('a', attrs={'href': re.compile(r'^/wiki')})
            queue[:0] = ["https://en.wikipedia.org" + link['href'] for link in links if ":" not in link['href']]

            if "List_of" not in current_link:
                result.append({"wikipedia_url": current_link, "text": body_content.get_text(strip=True)})
                pbar.update(1)

            sleep((max_sleep_time - min_sleep_time) * random.random() + min_sleep_time)

    return pd.DataFrame(result)
