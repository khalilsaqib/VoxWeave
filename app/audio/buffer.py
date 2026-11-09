from collections import deque
import numpy as np

class AudioBuffer:
    """
    Efficient rolling audio buffer.
    Stores mono float32 audio at a fixed sample rate.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration_seconds: float = 6.0,
    ):
        self.sample_rate = sample_rate

        self.max_samples = int(
            sample_rate * max_duration_seconds
        )

        self._chunks: deque[np.ndarray] = deque()
        self._sample_count = 0

    def append(
        self,
        audio: np.ndarray,
    ):
        if audio.size == 0:
            return

        if audio.dtype != np.float32:
            audio = audio.astype(
                np.float32,
                copy=False,
            )

        self._chunks.append(audio)
        self._sample_count += len(audio)
        self._trim()

    def _trim(self):
        while (
            self._sample_count > self.max_samples
            and self._chunks
        ):
            excess = (
                self._sample_count
                - self.max_samples
            )

            first = self._chunks[0]

            if len(first) <= excess:
                self._chunks.popleft()
                self._sample_count -= len(first)

            else:
                self._chunks[0] = first[excess:]
                self._sample_count -= excess

    def get_audio(
        self,
        max_duration_seconds: float | None = None,
    ) -> np.ndarray:

        if not self._chunks:
            return np.empty(
                0,
                dtype=np.float32,
            )

        audio = np.concatenate(
            tuple(self._chunks)
        )

        if max_duration_seconds is not None:
            max_samples = int(
                self.sample_rate
                * max_duration_seconds
            )

            if len(audio) > max_samples:
                audio = audio[-max_samples:]

        return audio

    def duration_seconds(self) -> float:
        return (
            self._sample_count
            / self.sample_rate
        )

    def sample_count(self) -> int:
        return self._sample_count

    def clear(self):
        self._chunks.clear()
        self._sample_count = 0

    
