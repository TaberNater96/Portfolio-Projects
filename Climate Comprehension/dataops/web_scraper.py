##############################################################################################
#                                         Web Scraper                                        #
#                                                                                            #
#                                       By: Elijah Taber                                     #
##############################################################################################

import requests
from bs4 import BeautifulSoup
from typing import List
import io
from PyPDF2 import PdfReader

class WebScraper:
    """
    This class provides functionality for web scraping text content from specified URLs. 
    It can handle both single and multiple URLs, extracting the text content from each web 
    page and returning it as a string or a list of strings respectively.
    """
    def scrape_website_text(
        self, 
        url: str
    ) -> str:
        """
        Scrapes the text content from the specified URL.

        Args:
            url (str): The URL of the web page to scrape.

        Returns:
            str: The extracted text content from the web page.
        """
        try:
            # Fetch the HTML content of the web page
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find and extract article text
                article_text = ""
                for paragraph in soup.find_all('p'):
                    article_text += paragraph.get_text() + " "

                return article_text.strip()  # strip leading/trailing whitespace
            else:
                print(f"Failed to fetch URL: {url}. Status code: {response.status_code}")
                return ""
        except requests.exceptions.RequestException as e:
            print("Error occurred while scraping:", e)
            return ""

    def scrape_multiple_website_texts(
        self, 
        urls: List[str]
    ) -> List[str]:
        """
        Scrapes text content from multiple URLs by utilizing the scrape_website_text
        method, but iterating through 1 URL at a time automatically.

        Args:
            urls (List[str]): A list of URLs of web pages to scrape.

        Returns:
            List[str]: A list of extracted text content from the web pages.
        """
        article_texts = []
        for url in urls:
            article_text = self.scrape_website_text(url)
            article_texts.append(article_text)
        return article_texts
    
    def scrape_pdf_text(
        self, 
        url: str
    ) -> str:
        """
        Scrapes the text content from a PDF file at the specified URL. 
        Websites such as ArXiv have all of their articles published as a pdf.

        Args:
            url (str): The URL of the PDF file to scrape.

        Returns:
            str: The extracted text content from the PDF file.
        """
        try:
            # Fetch the PDF content
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                pdf_file = io.BytesIO(response.content) # create a file-like object from the content
                pdf_reader = PdfReader(pdf_file)

                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + " "

                return text.strip()  # strip leading/trailing whitespace
            else:
                print(f"Failed to fetch PDF: {url}. Status code: {response.status_code}")
                return ""
        except Exception as e:
            print("Error occurred while scraping PDF:", e)
            return ""
        
if __name__ == "__main__":
    # Example usage
    web_scraper = WebScraper()

    # Scrape text from a single URL
    url = "https://www.example.com"
    text = web_scraper.scrape_website_text(url)
    print(text)

    # Scrape text from multiple URLs
    urls = ["https://www.example.com", "https://www.another-example.com"]
    texts = web_scraper.scrape_multiple_website_texts(urls)
    for text in texts:
        print(text)

    # Scrape text from a PDF URL
    pdf_url = "https://arxiv.org/pdf/2303.06731.pdf"
    pdf_text = web_scraper.scrape_pdf_text(pdf_url)
    print(pdf_text)