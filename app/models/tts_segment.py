import asyncio
from dataclasses import dataclass

@dataclass
class TtsSegment:
    segment_id: int
    generation_id: int
    text: str
    queue: asyncio.Queue
    done: asyncio.Event