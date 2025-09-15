import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.utils.api import APIClient
from src.config.settings import DATA_CONFIG

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.data_config = DATA_CONFIG
    
    def ensure_directory(self, path: str) -> None:
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)
    
    def cache_exists(self, filepath: str) -> bool:
        """Check if cached data exists"""
        return os.path.exists(filepath)
    
    def load_cached_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if exists"""
        if not self.cache_exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cached data from {filepath}: {e}")
            return None
    
    def save_to_cache(self, data: Dict[str, Any], filepath: str) -> None:
        """Save data to cache"""
        self.ensure_directory(os.path.dirname(filepath))
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved data to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
    
    def get_or_fetch(self, endpoint: str, filepath: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get data from cache or fetch from API"""
        if not force_refresh:
            cached_data = self.load_cached_data(filepath)
            if cached_data is not None:
                logger.info(f"Using cached data from {filepath}")
                return cached_data
        
        logger.info(f"Fetching data from API: {endpoint}")
        data = self.api_client.get(endpoint)
        self.save_to_cache(data, filepath)
        return data
    
    @abstractmethod
    def collect(self, *args, **kwargs) -> Dict[str, Any]:
        """Collect data - to be implemented by subclasses"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api_client.close()