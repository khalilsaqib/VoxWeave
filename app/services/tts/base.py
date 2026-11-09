import numpy as np
from abc import ABC, abstractmethod
from typing import AsyncIterator
from enum import Enum
from app.models.tts_segment import TtsSegment

class TtsMode(str, Enum):
    STREAM = "stream"
    SYNTHESIZE = "synthesize"

class TtsServiceBase(ABC):

    def __init__(self, mode: TtsMode = TtsMode.STREAM):
        self.mode = mode

    async def start(self):
        """
        Start resources required by this TTS service.

        Default implementation does nothing.
        Engines such as VoxCPM2 can override this to
        establish a WebSocket connection.
        """
        pass

    async def close(self):
        """
        Release resources owned by this TTS service.

        Default implementation does nothing.
        Engines such as VoxCPM2 can override this to
        close a WebSocket connection.
        """
        pass

    @abstractmethod
    async def cancel_generation(self, generation_id: int):
        """
        Cancel Current Generation
        """
        raise NotImplementedError

    @abstractmethod
    async def submit(self, text: str, generation_id: int) -> TtsSegment:
        """
        Submit one sentence for TTS generation.

        IMPORTANT:

        This method should normally return quickly.

        It returns a TtsSegment handle which can later be
        consumed through stream_segment().
        """
        raise NotImplementedError

    @abstractmethod
    async def _stream_segment(
        self,
        segment: TtsSegment,
    ) -> AsyncIterator[np.ndarray]:
        """
        Backend-specific raw streaming implementation.

        This method must yield PCM16 audio chunks:

            np.ndarray
            dtype=np.int16
            mono
            configured sample rate

        The base class handles TtsMode.STREAM vs
        TtsMode.SYNTHESIZE.
        """
        raise NotImplementedError

    async def stream_segment(
        self,
        segment: TtsSegment,
    ) -> AsyncIterator[np.ndarray]:
        """
        Public interface used by TtsWorker.

        STREAM:
            Pass every backend chunk directly to the worker.

        SYNTHESIZE:
            Collect all chunks for the sentence and expose
            one complete PCM16 array.
        """

        if self.mode == TtsMode.STREAM:
            async for audio in self._stream_segment(segment):
                if audio is None:
                    continue

                audio = np.asarray(audio, dtype=np.int16)
                if audio.size == 0:
                    continue
                yield audio
            return

        # -------------------------------------------------
        # SYNTHESIZE MODE
        # -------------------------------------------------

        chunks: list[np.ndarray] = []
        async for audio in self._stream_segment(segment):
            if audio is None:
                continue

            audio = np.asarray(audio, dtype=np.int16)
            if audio.size == 0:
                continue

            chunks.append(audio)

        if not chunks:
            return

        yield np.concatenate(chunks)

    async def stream(
        self,
        text: str,
    ) -> AsyncIterator[np.ndarray]:
        """
        Convenience API.
        Submit text and stream the resulting segment.
        The behavior follows self.mode.
        """
        segment = await self.submit(text)
        async for audio in self.stream_segment(segment):
            yield audio

    async def synthesize(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Convenience API that always returns the complete
        audio for one sentence.

        Regardless of the configured mode, the returned
        value is a complete PCM16 numpy array.
        """

        chunks: list[np.ndarray] = []

        segment = await self.submit(text)

        async for audio in self.stream_segment(segment):
            if audio is None:
                continue

            audio = np.asarray(audio, dtype=np.int16)
            if audio.size == 0:
                continue
            chunks.append(audio)
        if not chunks:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(chunks)