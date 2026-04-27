# data/cache_manager.py
import json
from datetime import datetime, timedelta
import os
from pathlib import Path

class CacheManager:
    def __init__(self, cache_file="data/cache/delays_cache.json", ttl_minutes=5):
        self.cache_file = cache_file
        self.ttl = timedelta(minutes=ttl_minutes)
        # Create directory if it doesn't exist
        cache_dir = Path(cache_file).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = self._load()
    
    def get(self, key):
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - datetime.fromisoformat(entry['timestamp']) < self.ttl:
                return entry['value']
        return None
    
    def set(self, key, value):
        self._cache[key] = {'value': value, 'timestamp': datetime.now().isoformat()}
        self._save()
    
    def _load(self):
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _save(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f)