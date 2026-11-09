from abc import ABC, abstractmethod

import numpy as np

class SttServiceBase(ABC):

    async def start(self):
        """
        Start resources required by this STT service.

        Default implementation does nothing.
        Engines such as Kaggle Whisper can override this to
        establish a WebSocket connection.
        """
        pass

    async def close(self):
        """
        Release resources owned by this STT service.

        Default implementation does nothing.
        Engines such as Kaggle Whisper can override this to
        close a WebSocket connection.
        """
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_16k: np.ndarray
    ) -> str:
        """
        Generate STT
        """
        raise NotImplementedError

    @abstractmethod
    async def transcribe_partial(
        self,
        audio_16k: np.ndarray
    ) -> str:
        """
        Generate STT
        """
        raise NotImplementedError