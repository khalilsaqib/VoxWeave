import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad
from app.core.logger import logger

class VadService:
    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
    ):
        logger.info("[VAD] | Loading Silero VAD")

        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        logger.info("[VAD] | Silero VAD loaded")

    def create_iterator(self) -> VADIterator:
        return VADIterator(
            self.model,
            sampling_rate=16000,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def process(
        self,
        vad_iterator: VADIterator,
        audio_chunk: np.ndarray,
    ):
        audio = torch.from_numpy(audio_chunk)
        return vad_iterator(audio)
