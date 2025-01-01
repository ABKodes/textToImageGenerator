import requests
from bs4 import BeautifulSoup

# Set the base URL
base_url = "http://www.addisadmassnews.com/"
response = requests.get(base_url)

# Parse the homepage
soup = BeautifulSoup(response.text, 'html.parser')

# Find all article links (modify this based on inspection)
links = soup.find_all('a', href=True)

# File to save text
output_file = 'addis_admass_news.txt'

# Open the file for writing
with open(output_file, 'w', encoding='utf-8') as file:
    for link in links:
        # Get the href attribute
        page_url = link['href']

        # Handle relative URLs
        if not page_url.startswith("http"):
            page_url = base_url + page_url

        try:
            # Request each page
            page_response = requests.get(page_url)
            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            # Extract all text from the article
            page_text = page_soup.get_text()
            cleaned_text = ' '.join(page_text.split())

            # Write the cleaned text to the file
            file.write(f"URL: {page_url}\n")
            file.write(cleaned_text + "\n\n")
            print(f"Scraped: {page_url}")

        except Exception as e:
            print(f"Failed to scrape {page_url}: {e}")

print(f"All text data saved in {output_file}!")
