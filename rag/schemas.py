from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Chunk:
    text: str
    source: str          
    chunk_id: int
    page: Optional[int] = None