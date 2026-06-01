"""
Web Scraper - Extract content from Ed Donner's website
"""
import time
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup


class WebContentExtractor:
    """Extract and structure content from Ed Donner's website"""

    def __init__(self, base_url: str = "https://edwarddonner.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Bot for Personal Study Notes)'
        })

    def fetch_page(self, url: str, delay: float = 1.0) -> Optional[BeautifulSoup]:
        """Fetch a page with rate limiting"""
        try:
            time.sleep(delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_course_resources_page(self) -> Dict:
        """Extract the main course resources page"""
        url = "https://edwarddonner.com/2024/11/13/llm-engineering-resources/"
        soup = self.fetch_page(url)

        if not soup:
            return {}

        # Find the main content area
        content = soup.find('article') or soup.find('main') or soup.find('div', class_='entry-content')

        if not content:
            return {'error': 'Could not find main content'}

        # Extract structured data
        data = {
            'url': url,
            'title': soup.find('h1').get_text(strip=True) if soup.find('h1') else 'LLM Engineering Resources',
            'sections': [],
            'links': []
        }

        # Extract sections with headings
        current_section = None
        for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'pre', 'code']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # New section
                if current_section:
                    data['sections'].append(current_section)
                current_section = {
                    'heading': element.get_text(strip=True),
                    'level': element.name,
                    'content': []
                }
            elif current_section:
                # Add content to current section
                if element.name == 'p':
                    text = element.get_text(strip=True)
                    if text:
                        current_section['content'].append({'type': 'paragraph', 'text': text})
                elif element.name in ['ul', 'ol']:
                    items = [li.get_text(strip=True) for li in element.find_all('li')]
                    current_section['content'].append({'type': 'list', 'items': items})
                elif element.name in ['pre', 'code']:
                    current_section['content'].append({'type': 'code', 'text': element.get_text()})

        # Add last section
        if current_section:
            data['sections'].append(current_section)

        # Extract all external links
        for link in content.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            if href and not href.startswith('#'):
                full_url = urljoin(url, href)
                data['links'].append({'text': text, 'url': full_url})

        return data

    def extract_faq_page(self) -> Dict:
        """Extract FAQ content"""
        url = "https://edwarddonner.com/faq/"
        soup = self.fetch_page(url)

        if not soup:
            return {}

        content = soup.find('article') or soup.find('main')
        if not content:
            return {}

        faqs = []
        current_q = None

        for element in content.find_all(['h2', 'h3', 'p']):
            if element.name in ['h2', 'h3']:
                # Question
                if current_q:
                    faqs.append(current_q)
                current_q = {
                    'question': element.get_text(strip=True),
                    'answer': []
                }
            elif current_q and element.name == 'p':
                # Answer
                text = element.get_text(strip=True)
                if text:
                    current_q['answer'].append(text)

        if current_q:
            faqs.append(current_q)

        return {
            'url': url,
            'faqs': faqs
        }

    def extract_key_blog_posts(self) -> List[Dict]:
        """Extract relevant blog posts"""
        # Start with the main page
        soup = self.fetch_page(self.base_url)
        if not soup:
            return []

        posts = []
        # Look for recent posts related to LLM/AI
        for article in soup.find_all('article', limit=10):
            title_elem = article.find(['h1', 'h2', 'h3'])
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            # Filter for relevant posts
            if any(keyword in title.lower() for keyword in ['llm', 'ai', 'gpt', 'claude', 'agent', 'rag', 'course']):
                link_elem = article.find('a')
                if link_elem and link_elem.get('href'):
                    url = urljoin(self.base_url, link_elem['href'])
                    posts.append({
                        'title': title,
                        'url': url
                    })

        return posts

    def extract_all(self) -> Dict:
        """Extract all website content"""
        print("Scraping Ed Donner's website...")

        data = {
            'course_resources': self.extract_course_resources_page(),
            'faq': self.extract_faq_page(),
            'blog_posts': self.extract_key_blog_posts()
        }

        print("✓ Website scraping complete")
        return data

    def format_for_markdown(self, data: Dict) -> str:
        """Convert extracted data to markdown format"""
        md = []

        # Course resources
        if 'course_resources' in data:
            resources = data['course_resources']
            md.append(f"# {resources.get('title', 'Course Resources')}\n")
            md.append(f"Source: {resources.get('url')}\n")

            for section in resources.get('sections', []):
                # Add heading
                level = section['level'].replace('h', '')
                md.append(f"{'#' * int(level)} {section['heading']}\n")

                # Add content
                for item in section.get('content', []):
                    if item['type'] == 'paragraph':
                        md.append(f"{item['text']}\n")
                    elif item['type'] == 'list':
                        for list_item in item['items']:
                            md.append(f"- {list_item}")
                        md.append("")
                    elif item['type'] == 'code':
                        md.append(f"```\n{item['text']}\n```\n")

        return '\n'.join(md)


def scrape_website() -> Dict:
    """Main function to scrape Ed Donner's website"""
    scraper = WebContentExtractor()
    return scraper.extract_all()


if __name__ == '__main__':
    print("Testing web scraper...")
    data = scrape_website()

    if 'course_resources' in data:
        print(f"\n✓ Extracted course resources page")
        print(f"  Found {len(data['course_resources'].get('sections', []))} sections")

    if 'faq' in data:
        print(f"✓ Extracted FAQ page")
        print(f"  Found {len(data['faq'].get('faqs', []))} FAQ entries")

    if 'blog_posts' in data:
        print(f"✓ Found {len(data['blog_posts'])} relevant blog posts")
