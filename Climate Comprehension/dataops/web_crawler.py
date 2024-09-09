#####################################################################################################################
#                                           Async Climate Change Web Crawler                                        #
#                                                                                                                   #
#                                                  By: Elijah Taber                                                 #
#####################################################################################################################

import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional

class ArXivClimateCrawler:
    """
    A web crawler designed to search arXiv for climate-related articles.
    It uses asynchronous requests to crawl through arXiv's research articles, while
    categorizing articles as climate-related or not based on their titles and abstracts.
    The crawler implements rate limiting and error handling to ensure responsible and robust crawling.
    The async protocol I have implemted puts each topic in a que based on which task is ready first. This
    allows for 1 request every 5 seconds (adhering to arXiv's terms of service). While each task waits in
    the request que, it will parse over the crawled urls to check what has and has not been crawled. This 
    allows the the crawler to perform 1 request every 5 seconds with no down time as each task is ready to go
    from the que.
    """
    
    def __init__(self):
        self.base_url = "https://arxiv.org/abs/"
        self.rate_limit = 5  # seconds
        self.climate_urls = set()
        self.non_climate_urls = set()
        self.df = pd.DataFrame(columns=['Climate Change URLs', 'Non-Climate Change URLs'])
        self.topics = {
            'climate feedback': ('2001', '2006'),
            'geoengineering': ('2007', '2012'),
            'climate model': ('2101', '2106'),
            'atmospheric chemistry': ('2107', '2112'),
            'glacier melting': ('2201', '2206'),
            'CO2 emissions': ('2207', '2212'),
            'greenhouse gas emissions': ('2301', '2306'),
            'ocean acidification': ('2307', '2312'),
            'heatwaves': ('1401', '1406'),
            'nuclear energy': ('1407', '1412'),
            'climate change': ('1501', '1506'),
            'renewable energy': ('1507', '1512'),
            'paleoclimatology': ('1601', '1606'),
            'global warming': ('1607', '1612'),
            'sea level rise': ('1701', '1706')
        }
        self.request_semaphore = asyncio.Semaphore(1)
        self.last_request_time = 0

    async def fetch(
        self, 
        session: aiohttp.ClientSession, 
        url: str
    ) -> Optional[str]:
        """
        Fetches the HTML content of a URL while implementing rate limiting and error handling.
        This method uses a semaphore to ensure only one request is made at a time and implements
        exponential backoff for failed requests.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for making requests.
            url (str): The URL to fetch.

        Returns:
            Optional[str]: The HTML content of the page if successful, None otherwise.
        """
         # Semaphore limits concurrent requests ensures only one request at a time
        async with self.request_semaphore:
            # Rate limiting: checks if enough time has passed since last request
            current_time = asyncio.get_running_loop.time()
            if current_time - self.last_request_time < self.rate_limit:
                await asyncio.sleep(self.rate_limit - (current_time - self.last_request_time))
                
            # Fetches URL: if successful, then return response text 
            try:
                async with session.get(url) as response:
                    self.last_request_time = asyncio.get_event_loop().time() # update for proper rate limiting
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"Error fetching {url}: Status {response.status}")
                        return None
                    
            # Catches timeout errors, if one occurs, wait 10 seconds before retry
            except asyncio.TimeoutError:
                print(f"Timeout error for {url}. Waiting before retrying...")
                await asyncio.sleep(10)
                return await self.fetch(session, url)
            
            # Catch-all handler
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                return None
            
    def is_climate_related(
        self, 
        title: str, 
        abstract: str, 
        topic: str
    ) -> bool:
        """
        Determines if an article is climate-related based on its title, abstract, and the current topic.
        This method checks for the presence of climate-related keywords in the combined text of the title and abstract.

        Args:
            title (str): The title of the article.
            abstract (str): The abstract of the article.
            topic (str): The current topic being searched.

        Returns:
            bool: True if the article is climate-related, False otherwise.
        """
        # Keywords to check in the abstract
        keywords = topic.split() + ['climate', 'global', 'warming', 'emission', 'greenhouse']
        text = (title + ' ' + abstract).lower()
        
        # Generator expression to check if any keywords are in the abstract
        return any(keyword in text for keyword in keywords)
    
    async def crawl_topic(
        self, 
        session: aiohttp.ClientSession, 
        topic: str, 
        start_date: str, 
        end_date: str
    ) -> None:
        """
        Crawls arXiv for a specific topic within a given date range. This range is based on arXiv's URL structure.
        It stops crawling once 5,000 climate-related articles have been found or the end date is reached.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for making requests.
            topic (str): The climate-related topic to search for.
            start_date (str): The start date for the search in YYMM format.
            end_date (str): The end date for the search in YYMM format.
        """
        # Parse the start and end dates, this format is specific to arXiv
        start = datetime.strptime(start_date, "%y%m")
        end = datetime.strptime(end_date, "%y%m")
        current = start
        
        # End the algorithm once 5,000 climate change articles have been found
        while current <= end and len(self.climate_urls) < 5_000:
            year_month = current.strftime("%y%m") # arXiv URL format: 2408 = August of 2024
            
            for i in range(1, 15_000): # around 15,000 articles are submitted every month
                article_id = f"{year_month}.{i:05d}" # specific article numbers, padded to 5 numbers
                url = self.base_url + article_id # build the full URL to crawl
                
                # Checks if the URL has been crawled to skip to next iteration
                if url in self.climate_urls or url in self.non_climate_urls:
                    continue
                
                html = await self.fetch(session, url)
                
                # Title and abstract are extracted using html elements
                if html:
                    
                    soup = BeautifulSoup(html, 'html.parser')
                    title = soup.find('h1', class_='title mathjax').text.strip() if soup.find('h1', class_='title mathjax') else ''
                    abstract = soup.find('blockquote', class_='abstract mathjax').text.strip() if soup.find('blockquote', class_='abstract mathjax') else ''
                    
                    if self.is_climate_related(title, abstract, topic):
                        self.climate_urls.add(url)
                    else:
                        self.non_climate_urls.add(url)
                        
                    self.update_dataframe()
                    self.print_progress()
                
                # Failsafe to ensure loop breaks after 5000 articles
                if len(self.climate_urls) >= 5000:
                    break
                
            # Move to the next month
            current += timedelta(days=32)
            current = current.replace(day=1)
                
    def update_dataframe(self) -> None:
        """
        Updates the DataFrame with the latest crawled URLs.
        This method appends the most recently found climate-related and non-climate-related URLs to the DataFrame.
        """
        new_row = {'Climate Change URLs': list(self.climate_urls)[-1] if self.climate_urls else None,
                   'Non-Climate Change URLs': list(self.non_climate_urls)[-1] if self.non_climate_urls else None}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
    def print_progress(self) -> None:
        """
        Prints the current progress of the crawler.
        This method displays the number of climate-related and non-climate-related articles found so far.
        """
        print(f"\rClimate Change Articles Found: {len(self.climate_urls)} | "
              f"Non-Climate Change Articles Found: {len(self.non_climate_urls)}", end='', flush=True)
        
    async def run(self) -> None:
        """
        Runs the crawler for all defined topics.
        This method creates and manages asynchronous tasks for crawling each topic concurrently.
        """
        async with aiohttp.ClientSession() as session: # open the client session
            tasks = [] # crawling tasks
            # Within each topic's assigned dates, crawl for the specific topic
            for topic, (start, end) in self.topics.items():
                task = asyncio.create_task(self.crawl_topic(session, topic, start, end))
                tasks.append(task)
            await asyncio.gather(*tasks) # run all tasks concurrently and wait for all to complete
            
    def save_results(
        self, 
        filename: str = 'arxiv_climate_urls.csv'
    ) -> None:
        """
        Saves the crawling results to a CSV file.
        This method exports the DataFrame containing the categorized URLs to a specified file. Here 
        the recommended file type is CSV. 

        Args:
            filename (str): The name of the file to save the results to. Defaults to 'arxiv_climate_urls.csv'.
        """
        self.df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
if __name__ == "__main__":
    crawler = ArXivClimateCrawler()
    asyncio.run(crawler.run())
    crawler.save_results()