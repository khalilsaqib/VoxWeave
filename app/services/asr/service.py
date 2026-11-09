import json
import asyncio
from typing import Literal
import numpy as np
import websockets

from app.core.logger import logger
from app.services.asr.base import SttServiceBase

AsrRequestType = Literal["partial", "final"]

class AsrService(SttServiceBase):
    """
    ASR service for ONE VoiceSession.
    Each instance owns exactly one WebSocket connection
    to the remote Kaggle Faster-Whisper server.

    Lifecycle:
        create()
            ↓
        start()
            ↓
        WebSocket connected
            ↓
        transcribe(...)
            ↓
        close()
    """
    def __init__(
        self,
        websocket_url: str,
        language: str | None = None,
    ):
        self.websocket_url = websocket_url
        self.language = language

        self.websocket = None
        self._connect_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()

        logger.info("[ASR] | Service created | url=%s", self.websocket_url)

    async def start(self):
        """
        Establish the WebSocket connection before
        the user starts speaking.
        """
        await self._connect()

    async def _connect(self):
        """
        Establish the WebSocket connection and wait
        for the Kaggle server's ready message.
        """
        if self.websocket is not None:
            return       

        async with self._connect_lock:
            if self.websocket is not None:
                return

            logger.info("[ASR] | Opening new WebSocket connection | url=%s", self.websocket_url)

            websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            )
            
            try:                
                await websocket.send(
                    json.dumps({
                            "type": "start",
                            "language": self.language,
                    })
                )

                response = await websocket.recv()

                if not isinstance(response, str):
                    raise RuntimeError(
                        "Kaggle ASR server returned "
                        "non-text response during startup"
                    )

                message = json.loads(response)
                if message.get("type") != "ready":
                    raise RuntimeError(
                        "Kaggle STT server did not return "
                        f"ready message: {message}"
                    )

                self.websocket = websocket
                logger.info("[ASR] | WebSocket connected | url=%s", self.websocket_url)


            except Exception:
                try:
                    await websocket.close()
                except Exception:
                    pass
                raise        

    async def _get_websocket(self):
        """
        Return the active WebSocket.

        Normally start() already connected it, but this
        also makes the service resilient if the connection
        was lost.
        """
        await self._connect()
        websocket = self.websocket

        if websocket is None:
            raise RuntimeError("Kaggle ASR WebSocket is not connected")

        return websocket

    async def _transcribe(
        self,
        audio_16k: np.ndarray,
        request_type: AsrRequestType
    ) -> str:
        """
        Transcribe a complete 16 kHz mono float32
        audio snapshot.
        The snapshot replaces the previous remote
        WebSocket audio buffer.
        """

        if audio_16k.size == 0:
            return ""

        async with self._transcribe_lock:
            websocket = await self._get_websocket()

            if audio_16k.dtype != np.float32:
                audio_16k = audio_16k.astype(np.float32, copy=False)

            audio_pcm16 = np.clip(audio_16k,-1.0,1.0)
            audio_pcm16 = (audio_pcm16 * 32767.0).astype(np.int16)
            audio_bytes = audio_pcm16.tobytes()

            await websocket.send(
                json.dumps({
                    "type": "replace",
                })
            )

            await websocket.send(audio_bytes)

            await websocket.send(
                json.dumps({
                    "type": request_type,
                })
            )

            response = await websocket.recv()
            if not isinstance(response, str):
                raise RuntimeError(
                    "Unexpected binary response from "
                    "Kaggle ASR server"
                )

            print(f"[ASR] response | {response}")

            message = json.loads(response)

            if message.get("type") != "transcript":
                raise RuntimeError(
                    "Unexpected Kaggle STT response: "
                    f"{message}"
                )

            text = message.get("text", "")

            logger.info(
                "[ASR] | Transcribe | Transcript | Status=%s | Text=%s | Language=%s | LanguageProbability=%s",
                message.get('status'), text, message.get('language'), message.get('language_probability')
            )

            return text.strip()

    async def transcribe_partial(
        self,
        audio_16k: np.ndarray
    ) -> str:

        return await self._transcribe(
            audio_16k=audio_16k,
            request_type="partial",
        )

    async def transcribe(
        self,
        audio_16k: np.ndarray
    ) -> str:
        return await self._transcribe(
            audio_16k=audio_16k,
            request_type="final",
        )

    async def close(self):

        websocket = self.websocket
        self.websocket = None

        if websocket is None:
            return
        
        logger.info("[ASR] | Closing WebSocket connection")

        try:            
            await websocket.close()
        except Exception:
            logger.exception("[ASR] | Error occurred while closing WebSocket connection")

        logger.info("[ASR] | WebSocket connection closed")