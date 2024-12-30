import re
import random
import requests
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from typing import Tuple

def wikipedia_scrapper_single_page(url: str) -> Tuple[str, str]:
    # Scrape and process the given URL
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {url} with status code {response.status_code}")
    
    # Extract the text from the <div id="mw-content-text">
    parsed = BeautifulSoup(response.text, 'html.parser')
    resolved_link = parsed.find('link', rel='canonical')["href"]
    body_content = parsed.find('div', id='mw-content-text')
    if body_content is None:
        raise Exception(f"Could not find <div id='mw-content-text'> in the page: {url}")
    new_text = body_content.get_text(strip=True)

    return new_text, resolved_link

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

            response = requests.get(current_link)
            if response.status_code != 200:
                raise Exception(f"Website returned status code {response.status_code}.")
            
            parsed = BeautifulSoup(response.text, 'html.parser')

            # Handle redirects by resolving the final URL
            resolved_link = parsed.find('link', rel='canonical')["href"]
            if resolved_link in visited:
                continue

            visited.add(resolved_link)

            body_content = parsed.find('div', id='mw-content-text')

            if body_content is None:
                continue

            links = body_content.find_all('a', attrs={'href': re.compile(r'^/wiki')})
            queue[:0] = ["https://en.wikipedia.org" + link['href'] for link in links if ":" not in link['href']]

            if "List_of" not in resolved_link:
                result.append({"wikipedia_url": resolved_link, "text": body_content.get_text(strip=True)})
                pbar.update(1)

            sleep((max_sleep_time - min_sleep_time) * random.random() + min_sleep_time)

    return pd.DataFrame(result)
