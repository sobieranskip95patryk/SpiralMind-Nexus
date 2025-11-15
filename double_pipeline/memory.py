from collections import deque
from typing import Any

class ShortTermMemory:
    def __init__(self, maxlen: int = 128):
        self.data = deque(maxlen=maxlen)
        self.maxlen = maxlen
    
    def put(self, key: str, value: Any) -> None:
        self.data.append({"key": key, "value": value})
    
    def get_last(self, key: str, default: Any = None) -> Any:
        for item in reversed(self.data):
            if item["key"] == key:
                return item["value"]
        return default

class LongTermMemory:
    def __init__(self):
        self.storage = {}
    
    def upsert(self, key: str, value: Any) -> None:
        self.storage[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.storage.get(key, default)

class EpisodicMemory:
    def __init__(self):
        self.episodes = []
    
    def add(self, **kwargs) -> None:
        self.episodes.append(kwargs)
    
    def get_recent(self, n: int = 10) -> list:
        return self.episodes[-n:]
