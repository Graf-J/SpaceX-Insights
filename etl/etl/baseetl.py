from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pathlib import Path
import httpx

T = TypeVar("T")


class BaseETL(ABC, Generic[T]):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def extract(self, url: str) -> list[dict]:
        # Fetch Data from API
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        # Ensure the return is always a List of Dicts
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise TypeError(f"Unexpected response type: {type(data)}")

    @abstractmethod
    def transform(self, raw_data: list[dict]) -> list[T]:
        """Clean and transform the raw data"""
        pass

    @abstractmethod
    async def load(self, transformed_data: list[T]) -> None:
        """Load data into storage"""
        pass
