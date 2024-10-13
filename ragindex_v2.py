import base64
import requests
import logging
import urllib3
import os
import re  # To handle alphanumeric filtering
from pathlib import Path

# Suppress the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Freshservice API Details
api_key = "dfsfdsdfsfsdfs"  # Replace with your API key
encoded_api_key = base64.b64encode(f"{api_key}:".encode()).decode()

headers = {
    'Authorization': f'Basic {encoded_api_key}',
    'Content-Type': 'application/json'
}

# Freshservice domain and URL with folder ID
FRESHSERVICE_DOMAIN = 'domain.freshservice.com'
SOLUTIONS_API_URL = f'https://{FRESHSERVICE_DOMAIN}/api/v2/solutions/articles'
FOLDER_ID = 32000005167  # Replace with your folder ID

# Directory for saving articles
ARTICLE_SAVE_PATH = Path('/home/brandon/graphrag/input')

# Ensure the directory exists
ARTICLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Fetch articles from the specified folder
def fetch_articles():
    try:
        params = {'folder_id': FOLDER_ID}  # Specify folder ID to fetch articles from
        response = requests.get(SOLUTIONS_API_URL, headers=headers, params=params, verify=False)
        response.raise_for_status()  # Check for HTTP errors
        articles = response.json().get('articles', [])
        if articles:
            logger.info(f"Fetched {len(articles)} articles.")
        else:
            logger.info("No articles found.")
        return articles
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        logger.error(f"Error occurred: {err}")
    return []

# Save article content to a text file with an alphanumeric name
def save_article_to_file(article):
    article_id = article['id']
    title = re.sub(r'[^a-zA-Z0-9]', '', article['title'])  # Remove non-alphanumeric characters
    file_name = f"article_{article_id}_{title}.txt"  # Unique file name
    file_path = ARTICLE_SAVE_PATH / file_name
    
    try:
        with open(file_path, 'w') as file:
            file.write(article['description'])  # Assuming 'description' contains the article content
        logger.info(f"Article {article_id} saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving article {article_id}: {e}")
        return None

# Process and save articles
def process_articles():
    articles = fetch_articles()

    if not articles:
        logger.info("No articles to process.")
        return

    # Save each article to a text file
    for article in articles:
        save_article_to_file(article)

if __name__ == '__main__':
    process_articles()
