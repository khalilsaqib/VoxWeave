from dataclasses import dataclass
from typing import Literal
import numpy as np
from app.services.asr.service import AsrService

AsrRequestType = Literal["partial", "final"]

@dataclass
class AsrRequest:
    session_id: str
    request_type: AsrRequestType
    audio: np.ndarray
    sequence: int
    asr_service: AsrService

