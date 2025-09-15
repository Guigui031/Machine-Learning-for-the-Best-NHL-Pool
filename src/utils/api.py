import requests
import time
import logging
from typing import Optional, Dict, Any
from src.config.settings import API_CONFIG

logger = logging.getLogger(__name__)

class APIError(Exception):
    pass

class RateLimitError(APIError):
    pass

class APIClient:
    def __init__(self):
        self.base_url = API_CONFIG.base_url
        self.headers = API_CONFIG.headers
        self.rate_limit_delay = API_CONFIG.rate_limit_delay
        self.max_retries = API_CONFIG.max_retries
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request with retry logic and rate limiting"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error for {url}: {e}")
                    raise APIError(f"HTTP {response.status_code}: {e}")
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise APIError(f"Request failed: {e}")
                
                wait_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        raise APIError(f"Failed to get data from {url} after {self.max_retries} attempts")
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()