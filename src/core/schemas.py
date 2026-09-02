from pydantic import BaseModel, Field
from typing import Dict, List

class SearchResult(BaseModel):
    memory: str
    rationale: str