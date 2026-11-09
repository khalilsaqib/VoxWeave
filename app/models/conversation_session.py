import asyncio
import time
from dataclasses import dataclass, field

@dataclass
class ConversationSession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)