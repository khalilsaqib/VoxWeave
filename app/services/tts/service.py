import asyncio
import json
from typing import AsyncIterator

import numpy as np
import websockets

from app.services.tts.base import (
    TtsMode,
    TtsSegment,
    TtsServiceBase)
from app.core.logger import logger


class TtsService(TtsServiceBase):
    def __init__(
        self,
        websocket_url: str,
        sample_rate: int = 24000,
        mode: TtsMode = TtsMode.STREAM,
    ):
        super().__init__(mode=mode)

        self.websocket_url = websocket_url
        self.sample_rate = sample_rate

        self.websocket = None

        # ---------------------------------------------------------
        # Connection lifecycle
        # ---------------------------------------------------------
        self._connect_lock = asyncio.Lock()

        # ---------------------------------------------------------
        # Text requests waiting to be sent to VoxCPM
        # ---------------------------------------------------------
        self._send_queue: asyncio.Queue = asyncio.Queue()

        # ---------------------------------------------------------
        # Segments submitted by the application.
        #
        # The server processes them sequentially, so this queue
        # tells the receiver which segment an audio_start belongs to.
        # ---------------------------------------------------------
        self._pending_segments: asyncio.Queue = asyncio.Queue()

        # ---------------------------------------------------------
        # Background tasks
        # ---------------------------------------------------------
        self._sender_task: asyncio.Task | None = None
        self._receiver_task: asyncio.Task | None = None

        # ---------------------------------------------------------
        # Segment IDs
        # ---------------------------------------------------------
        self._next_segment_id = 0
        self._cancelled_generations: set[int] = set()

        # ---------------------------------------------------------
        # Currently receiving segment
        # ---------------------------------------------------------
        self._current_segment: TtsSegment | None = None

        # ---------------------------------------------------------
        # Shutdown
        # ---------------------------------------------------------
        self._closing = False

        logger.info("[TTS] | Creating Service | URL=%s | SampleRate=%s",self.websocket_url, self.sample_rate)

    async def start(self):
        await self._connect()

        if self._sender_task is None:
            self._sender_task = asyncio.create_task(self._sender_loop())

        if self._receiver_task is None:
            self._receiver_task = asyncio.create_task(self._receiver_loop())

        self._closing = False
        logger.info("[TTS] | Start | Service started")

    async def _connect(self):

        if self.websocket is not None:
            return

        async with self._connect_lock:
            if self.websocket is not None:
                return
            logger.info("[TTS] | Connect | Connecting to webSocket | URL=%s",self.websocket_url)

            self.websocket = await websockets.connect(
                self.websocket_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=60,
            )
            logger.info("[TTS] | Connect | WebSocket connected")

    # =============================================================
    # SUBMIT TEXT
    # =============================================================

    async def submit(self, text: str, generation_id: int) -> TtsSegment:
        text = text.strip()

        if not text:
            raise ValueError("Text is empty")

        if self._closing:
            raise RuntimeError("TTS service is closing")

        await self._connect()

        if self._sender_task is None:
            self._sender_task = asyncio.create_task(self._sender_loop())

        if self._receiver_task is None:
            self._receiver_task = asyncio.create_task(self._receiver_loop())

        segment_id = self._next_segment_id
        self._next_segment_id += 1

        segment = TtsSegment(
            segment_id=segment_id,
            generation_id=generation_id,
            text=text,
            queue=asyncio.Queue(),
            done=asyncio.Event(),
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Put the segment into pending order BEFORE putting the
        # request into the send queue.
        #
        # This guarantees that when the server sends audio_start,
        # the receiver already knows which segment it belongs to.
        # ---------------------------------------------------------

        if generation_id in self._cancelled_generations:
            logger.info("[TTS] | Submit | Ignoring submission for cancelled generation | Generation=%s",generation_id)
            segment.done.set()
            await segment.queue.put(None)
            return segment

        await self._pending_segments.put(segment)
        await self._send_queue.put(segment)

        logger.info("[TTS] | Submit | Segment submitted | id=%s | Text=%s", segment_id, text)

        return segment

    # =============================================================
    # SENDER
    # =============================================================

    async def _sender_loop(self):
        logger.info("[TTS] | SenderLoop | Sender loop started")

        try:
            while not self._closing:
                segment: TtsSegment = await self._send_queue.get()
                try:
                    if segment.generation_id in self._cancelled_generations:
                        logger.info("[TTS] | SenderLoop | Skipping stale segment | Id=%s | Generation=%s", segment.segment_id, segment.generation_id)
                        segment.done.set()
                        await segment.queue.put(None)
                        continue

                    await self._connect()
                    websocket = self.websocket
                    if websocket is None:
                        raise RuntimeError("WebSocket is not connected")
                    
                    logger.info("[TTS] | SenderLoop | Sending segment | Id=%s | text=%s", segment.segment_id, segment.text)

                    await websocket.send(
                        json.dumps({
                            "type": "text",
                            "text": segment.text,
                            "generation_id": segment.generation_id,
                        })
                    )

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("[TTS] | SenderLoop | Error occurred | Id=%s", segment.segment_id)                    

                    await segment.queue.put(exc)
                    await segment.queue.put(None)
                    segment.done.set()
                finally:
                    self._send_queue.task_done()

        except asyncio.CancelledError:
            logger.error("[TTS] | SenderLoop | Sender loop cancelled")
            raise
        finally:
            logger.info("[TTS] | SenderLoop | Sender loop exited")

    # =============================================================
    # RECEIVER
    # =============================================================

    async def _receiver_loop(self):

        logger.info("[TTS] | ReceiverLoop | Receiver loop started")

        try:
            while not self._closing:
                await self._connect()
                websocket = self.websocket
                if websocket is None:
                    raise RuntimeError("VoxCPM2 WebSocket is not connected")

                message = await websocket.recv()

                # -------------------------------------------------
                # JSON CONTROL MESSAGE
                # -------------------------------------------------

                if isinstance(message, str):
                    response = json.loads(message)
                    message_type = response.get("type")

                    # ---------------------------------------------
                    # AUDIO START
                    # ---------------------------------------------

                    if message_type == "audio_start":

                        sample_rate = int(
                            response.get(
                                "sample_rate",
                                self.sample_rate,
                            )
                        )

                        self.sample_rate = sample_rate

                        try:
                            segment: TtsSegment = self._pending_segments.get_nowait()
                        except asyncio.QueueEmpty:
                            raise RuntimeError("Received audio_start but no pending TTS segment exists")


                        # If segment is stale, discard it right away
                        if segment.generation_id in self._cancelled_generations:
                            logger.info("[TTS] | ReceiverLoop | Discarding audio_start for cancelled generation | Generation=%s", segment.generation_id)
                            
                            segment.done.set()
                            await segment.queue.put(None)
                            self._current_segment = None
                            continue


                        self._current_segment = segment
                        logger.info("[TTS] | ReceiverLoop | Audio started | Id=%s | SampleRate=%s", segment.segment_id, sample_rate)
                        continue

                    # ---------------------------------------------
                    # AUDIO END
                    # ---------------------------------------------

                    if message_type == "audio_end":

                        segment = self._current_segment

                        if segment is None:
                            logger.info("[TTS] | ReceiverLoop | Warning: audio ended without current segment")
                            continue

                        await segment.queue.put(None)
                        segment.done.set()

                        logger.info("[TTS] | ReceiverLoop | Audio completed | Id=%s", segment.segment_id)
                        self._current_segment = None
                        continue

                    # ---------------------------------------------
                    # ERROR
                    # ---------------------------------------------

                    if message_type == "error":
                        error = RuntimeError(
                            response.get(
                                "message",
                                "Unknown VoxCPM2 TTS error",
                            )
                        )

                        segment = self._current_segment

                        if segment is not None:
                            await segment.queue.put(error)
                            await segment.queue.put(None)
                            segment.done.set()
                            self._current_segment = None

                        else:
                            logger.info("[TTS] | ReceiverLoop | Error without current segment | Error=%s", error)

                        continue
                    
                    logger.info("[TTS] | ReceiverLoop | Unknown message | Response=%s", response)
                    continue

                # -------------------------------------------------
                # BINARY AUDIO
                # -------------------------------------------------

                if not message:
                    continue

                segment = self._current_segment

                if segment is None or segment.generation_id in self._cancelled_generations:
                    logger.info("[TTS] | ReceiverLoop | Warning: audio received without current segment")
                    continue

                pcm16 = np.frombuffer(
                    message,
                    dtype=np.int16,
                )

                if pcm16.size == 0:
                    continue

                await segment.queue.put(pcm16)

                logger.info("[TTS] | ReceiverLoop | TTS chunk | Id=%s | Samples=%s | Duration=%.3fs", segment.segment_id, len(pcm16), len(pcm16) / self.sample_rate)

        except asyncio.CancelledError:
            logger.error("[TTS] | ReceiverLoop | Receiver loop cancelled")
            raise
        except Exception as exc:
            logger.exception("[TTS] | ReceiverLoop | Receiver loop error occurred")

            segment = self._current_segment

            if segment is not None:
                await segment.queue.put(exc)
                await segment.queue.put(None)
                segment.done.set()
                self._current_segment = None

        finally:
            logger.info("[TTS] | ReceiverLoop | Receiver loop stopped")

    # =============================================================
    # CONSUME ONE SEGMENT
    # =============================================================

    async def _stream_segment(
        self,
        segment: TtsSegment,
    ) -> AsyncIterator[np.ndarray]:
        
        logger.info("[TTS] | StreamSegment | Consuming segment | Id=%s", segment.segment_id)

        while True:
            item = await segment.queue.get()
            try:
                if item is None:
                    break

                if isinstance(item, Exception):
                    raise item

                if segment.generation_id in self._cancelled_generations:
                    break

                audio = np.asarray(
                    item,
                    dtype=np.int16,
                )

                if audio.size == 0:
                    continue

                yield audio
            
            finally:
                segment.queue.task_done()

        logger.info("[TTS] | StreamSegment | Segment consumed | Id=%s", segment.segment_id)

    # =============================================================
    # CANCEL GENERATION (BARGE-IN)
    # =============================================================

    async def cancel_generation(self, generation_id: int):
        logger.info("[TTS] | CancelGeneration | Cancelling generation | Generation=%s", generation_id)
        self._cancelled_generations.add(generation_id)

        # 1. Send interrupt/cancel signal to remote server if connected
        if self.websocket is not None:
            try:
                await self.websocket.send(
                    json.dumps({
                        "type": "cancel",
                        "generation_id": generation_id,
                    })
                )
            except Exception:
                logger.exception("[TTS] | CancelGeneration | Failed to send cancel signal to server")

        # 2. Mark current segment done if it matches stale generation
        if (
            self._current_segment is not None
            and self._current_segment.generation_id == generation_id
        ):
            await self._current_segment.queue.put(None)
            self._current_segment.done.set()
            self._current_segment = None

        # 3. Purge stale segments from _send_queue
        temp_send_list: list[TtsSegment] = []
        while not self._send_queue.empty():
            try:
                seg = self._send_queue.get_nowait()
                if seg.generation_id == generation_id:
                    seg.done.set()
                    await seg.queue.put(None)
                else:
                    temp_send_list.append(seg)
                self._send_queue.task_done()
            except asyncio.QueueEmpty:
                break

        for seg in temp_send_list:
            await self._send_queue.put(seg)

        # 4. Purge stale segments from _pending_segments
        temp_pending_list: list[TtsSegment] = []
        while not self._pending_segments.empty():
            try:
                seg = self._pending_segments.get_nowait()
                if seg.generation_id == generation_id:
                    seg.done.set()
                    await seg.queue.put(None)
                else:
                    temp_pending_list.append(seg)
                self._pending_segments.task_done()
            except asyncio.QueueEmpty:
                break

        for seg in temp_pending_list:
            await self._pending_segments.put(seg)

    # =============================================================
    # CLOSE
    # =============================================================    

    async def close(self):

        if self._closing:
            return

        self._closing = True
        logger.info("[TTS] | Close | Closing TTS service")

        # ---------------------------------------------------------
        # Cancel background tasks
        # ---------------------------------------------------------

        tasks = [
            self._sender_task,
            self._receiver_task,
        ]

        for task in tasks:

            if task is not None:
                task.cancel()

        valid_tasks = [
            task
            for task in tasks
            if task is not None
        ]

        if valid_tasks:
            await asyncio.gather(
                *valid_tasks,
                return_exceptions=True,
            )

        self._sender_task = None
        self._receiver_task = None

        # ---------------------------------------------------------
        # Close WebSocket
        # ---------------------------------------------------------

        websocket = self.websocket
        self.websocket = None

        if websocket is None:
            return
        
        logger.info("[TTS] | Close | Closing TTS WebSocket")
        try:                
            await websocket.close()

        except Exception:
            logger.exception("[TTS] | Close | Error closing WebSocket")

        finally:
            logger.info("[TTS] | Close | WebSocket closed")