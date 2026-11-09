import asyncio
import time
from typing import Callable
from app.services.tts.base import TtsServiceBase
from app.services.tts.publisher import TtsAudioPublisher
from app.core.logger import logger

class TtsWorker:
    def __init__(
        self,
        session_id: str,
        queue: asyncio.Queue[str],
        tts_service: TtsServiceBase,
        publisher: TtsAudioPublisher,
        generation_id_getter: Callable[[], int],
        max_pending_segments: int = 4
    ):
        self.session_id = session_id
        self.queue = queue
        self.tts_service = tts_service
        self.publisher = publisher
        self.generation_id_getter = generation_id_getter
        self.max_pending_segments = max_pending_segments
        self._playback_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.max_pending_segments
        )

        self._task: asyncio.Task | None = None
        self._producer_task: asyncio.Task | None = None
        self._playback_task: asyncio.Task | None = None

        self._stopping = False
        logger.info("[TTS Worker] | Service created | Session=%s | MaxPendingSegments=%s", self.session_id, self.max_pending_segments)        

    # =============================================================
    # START
    # =============================================================

    def start(self):
        if self._task is not None:
            return

        self._stopping = False
        self._task = asyncio.create_task(self._run())

        logger.info("[TTS Worker] | Started | Session=%s", self.session_id)

    def flush(self):
        """Instantly purge pending playback segments on barge-in."""
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # =============================================================
    # MAIN WORKER
    # =============================================================

    async def _run(self):
        try:

            self._producer_task = asyncio.create_task(
                self._produce_segments()
            )

            self._playback_task = asyncio.create_task(
                self._play_segments()
            )

            done, pending = await asyncio.wait(
                {
                    self._producer_task,
                    self._playback_task,
                },
                return_when=asyncio.FIRST_EXCEPTION,
            )

            # -----------------------------------------------------
            # If one task failed, propagate its exception.
            # -----------------------------------------------------

            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception

            if self._producer_task.done():
                await self._playback_task

        except asyncio.CancelledError:
            logger.error("[TTS Worker] | Run | Cancelled | Session=%s", self.session_id) 
            raise

        except Exception:
            logger.exception("[TTS Worker] | Run | Error | Session=%s", self.session_id)

        finally:
            for task in (
                self._producer_task,
                self._playback_task,
            ):
                if task is not None and not task.done():
                    task.cancel()

            await asyncio.gather(
                *[
                    task
                    for task in (
                        self._producer_task,
                        self._playback_task,
                    )
                    if task is not None
                ],
                return_exceptions=True,
            )

            self._producer_task = None
            self._playback_task = None

            logger.info("[TTS Worker] | Run | Stopped | Session=%s", self.session_id)


    # =============================================================
    # PRODUCER
    # =============================================================
    async def _produce_segments(self):

        """
        Reads text from the LLM queue and submits it to the
        persistent VoxCPM WebSocket.

        IMPORTANT:

        This method does NOT consume audio.

        That means it can submit sentence 2 while sentence 1
        is currently being played.
        """

        logger.info("[TTS Worker] | ProduceSegments | Started | Session=%s", self.session_id)

        try:

            while not self._stopping:

                queue_wait_start = time.perf_counter()
                text = await self.queue.get()
                queue_wait = time.perf_counter() - queue_wait_start

                try:
                    if not text or not text.strip():
                        continue

                    text = text.strip()
                    
                    logger.info("[TTS Worker] | ProduceSegments | Received text | Session=%s | Text=%s", self.session_id, text)                  

                    generation_id = self.generation_id_getter()
                    submit_start = time.perf_counter()
                    segment = await self.tts_service.submit(text, generation_id)

                    submit_time = time.perf_counter() - submit_start
                    
                    logger.info(
                        "[TTS Worker] | ProduceSegments | Submitted | Session=%s | Segment=%s | Generation=%s | SubmitTime=%.3fs | QueueWait=%.3fs",
                        self.session_id, segment.segment_id, generation_id, submit_time, queue_wait
                    ) 

                    if segment.generation_id != self.generation_id_getter():
                        logger.info("[TTS Worker] | ProduceSegments | Dropping stale segment post-submit | Generation=%s", segment.generation_id)
                        continue                    

                    await self._playback_queue.put(segment)

                    logger.info(
                        "[TTS Worker] | ProduceSegments | Queued for playback | Session=%s | Segment=%s | Generation=%s | Pending=%s",
                        self.session_id, segment.segment_id, generation_id, self._playback_queue.qsize()
                    )                    

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.exception("[TTS Worker] | ProduceSegments | Error | Session=%s", self.session_id)                    

                finally:
                    self.queue.task_done()

        except asyncio.CancelledError:
            logger.error("[TTS Worker] | ProduceSegments | Cancelled | Session=%s", self.session_id)
            raise

        finally:
            logger.info("[TTS Worker] | ProduceSegments | Stopped | Session=%s", self.session_id)

    # =============================================================
    # PLAYBACK
    # =============================================================

    async def _play_segments(self):
        """
        Consumes TTS segments in order and publishes their audio
        to LiveKit.

        While this method is publishing segment N, the producer
        can already submit segment N+1.
        """

        logger.info("[TTS Worker] | PlaySegments | Started | Session=%s", self.session_id)       

        try:
            while not self._stopping:

                segment = await self._playback_queue.get()

                try:

                    if (
                        segment.generation_id
                        != self.generation_id_getter()
                    ):
                        logger.info(
                            "[TTS Worker] | PlaySegments | Ignoring stale segment | Session=%s | Segment=%s | SegmentGeneration=%s | CurrentGeneration=%s",
                            self.session_id, segment.segment_id, segment.generation_id, self.generation_id_getter()
                        )                        
                        continue
                    
                    logger.info("[TTS Worker] | PlaySegments | Starting segment | Session=%s | Segment=%s", self.session_id, segment.segment_id) 

                    playback_start = time.perf_counter()
                    first_audio_time = None
                    total_audio_samples = 0
                    publish_time = 0.0 
                    published_chunks = 0              

                    async for audio_chunk in self.tts_service.stream_segment(segment):
                        if (
                            segment.generation_id
                            != self.generation_id_getter()
                        ):                            
                            logger.info("[TTS Worker] | PlaySegments | Interrupted stale segment | Session=%s | Segment=%s", self.session_id, segment.segment_id)
                            break
                        
                        if audio_chunk is None or audio_chunk.size == 0:
                            continue

                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()

                        total_audio_samples += len(audio_chunk)
                        publish_start = time.perf_counter()

                        await self.publisher.publish(
                            audio_chunk,
                            generation_id=segment.generation_id
                        )

                        publish_time += (time.perf_counter() - publish_start)
                        published_chunks += 1                   


                    playback_time = time.perf_counter() - playback_start                    

                    audio_duration = (
                        total_audio_samples
                        / self.publisher.sample_rate
                        if self.publisher.sample_rate > 0
                        else 0.0
                    )

                    rtf = (
                        playback_time / audio_duration
                        if audio_duration > 0
                        else 0.0
                    )

                    ttfa = (
                        first_audio_time
                        - playback_start
                        if first_audio_time is not None
                        else 0.0
                    )
                    logger.info(
                        "[TTS Worker] | PlaySegments | Completed | Session=%s | Segment=%s | Audio=%.3fs | PlaybackLoop=%.3fs | TTFA=%.3fs | Publish=%.3fs | Chunks=%s | RTF=%.3fs",
                        self.session_id, segment.segment_id, audio_duration, playback_time, ttfa, publish_time, published_chunks, rtf
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("[TTS Worker] | PlaySegments | Error | Session=%s | Segment=%s", self.session_id, segment.segment_id)
                finally:
                    self._playback_queue.task_done()
        except asyncio.CancelledError:
            logger.error("[TTS Worker] | PlaySegments | Cancelled | Session=%s", self.session_id)
            raise
        finally:
            logger.info("[TTS Worker] | PlaySegments | Stopped | Session=%s", self.session_id)

    # =============================================================
    # STOP
    # =============================================================

    async def stop(self):
        self._stopping = True        
        logger.info("[TTS Worker] | Stop | Stopping | Session=%s", self.session_id)

        # ---------------------------------------------------------
        # Cancel the main worker.
        # ---------------------------------------------------------
        if self._task is not None:
            self._task.cancel()

            await asyncio.gather(
                self._task,
                return_exceptions=True,
            )
            self._task = None

        # ---------------------------------------------------------
        # Make sure child tasks are gone.
        # ---------------------------------------------------------

        for task in (
            self._producer_task,
            self._playback_task,
        ):

            if task is not None and not task.done():
                task.cancel()

        await asyncio.gather(
            *[
                task
                for task in (
                    self._producer_task,
                    self._playback_task,
                )
                if task is not None
            ],
            return_exceptions=True,
        )

        self._producer_task = None
        self._playback_task = None

        # ---------------------------------------------------------
        # Drain incoming LLM queue.
        # ---------------------------------------------------------

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()

            except asyncio.QueueEmpty:
                break

        self.flush()
        logger.info("[TTS Worker] | Stop | Stopped cleanly | Session=%s", self.session_id)       