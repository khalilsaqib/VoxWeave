import asyncio
import itertools
import time
import numpy as np

from app.models.asr_request import AsrRequest
from app.core.logger import logger


class AsrScheduler:
    """
    Centralized ASR scheduler.
    Responsibilities:

    - Accept ASR requests from many sessions.
    - Prioritize FINAL requests over PARTIAL requests.
    - Allow useful partial results to reach the frontend.
    - Prevent partial results from appearing after FINAL.
    - Limit concurrent ASR inference.
    - Support synchronous and asynchronous ASR services.
    - Keep synchronous ASR work outside the asyncio event loop.
    """

    def __init__(
        self,
        max_workers: int = 1,
        max_queue_size: int = 100,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")

        self.max_workers = max_workers
        self.max_queue_size = max_queue_size

        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )

        self.workers: list[asyncio.Task] = []
        self._sequence_counter = itertools.count()        
        self._final_sequences: dict[str, int] = {}

        self._running = False
        self._active_jobs = 0

    async def start(self):
        if self._running:
            return

        self._running = True
        logger.info("[ASR Scheduler] | Starting schedular | Workers=%s | MaxQueueSize=%s", self.max_workers, self.max_queue_size)       
        
        for worker_id in range(self.max_workers):
            task = asyncio.create_task(self._worker(worker_id))
            self.workers.append(task)

        logger.info("[ASR Scheduler] | Schedular started")

    async def stop(self):
        if not self._running:
            return

        logger.info("[ASR Scheduler] | Stopping schedular")
        self._running = False

        for task in self.workers:
            if not task.done():
                task.cancel()

        await asyncio.gather(
            *self.workers,
            return_exceptions=True,
        )

        self.workers.clear()
        logger.info("[ASR Scheduler] | Schedular stopped")

    def begin_utterance(
        self,
        session_id: str,
    ):
        """
        Mark the beginning of a new speech utterance.

        Any previous FINAL state is removed so
        partial results from the new utterance
        are allowed through.
        """

        self._final_sequences.pop(session_id, None)

        logger.info("[ASR Scheduler] | New utterance created | Session=%s", session_id)

    async def submit_partial(
        self,
        session_id: str,
        audio: np.ndarray,
        asr_service
    ) -> str:
        sequence = next(self._sequence_counter) 

        request = AsrRequest(
            session_id=session_id,
            request_type="partial",
            audio=audio,
            sequence=sequence,
            asr_service=asr_service
        )

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self._put_request(
            priority=10,
            request=request,
            future=future,
        )
        return await future

    async def submit_final(
        self,
        session_id: str,
        audio: np.ndarray,
        asr_service
    ) -> str:
        sequence = next(self._sequence_counter)
        self._final_sequences[
            session_id
        ] = sequence

        request = AsrRequest(
            session_id=session_id,
            request_type="final",
            audio=audio,
            sequence=sequence,
            asr_service=asr_service
        )

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self._put_request(
            priority=0,
            request=request,
            future=future,
        )
        return await future

    async def _put_request(
        self,
        priority: int,
        request: AsrRequest,
        future: asyncio.Future,
    ):
        if not self._running:
            raise RuntimeError("ASR scheduler is not running")

        submitted_at = time.monotonic()
        
        item = (
            priority,
            request.sequence,
            submitted_at,
            request,
            future,
        )

        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            if request.request_type == "partial":
                if not future.done():
                    future.set_result("")

                logger.info("[ASR Scheduler] | Dropping partial asr request because queue is full | Session=%s", request.session_id)
                return

            logger.info("[ASR Scheduler] | Final request waiting for queue space | Session=%s", request.session_id)
            await self.queue.put(item)

    async def _worker(
        self,
        worker_id: int,
    ):
        logger.info("[ASR Scheduler] | ASR Worker (%s) | Started", worker_id)
        try:
            while True:
                (
                    priority,
                    queue_sequence,
                    submitted_at,
                    request,
                    future,
                ) = await self.queue.get()

                job_started = False

                try:
                    if future.cancelled():
                        continue

                    if (
                        request.request_type == "partial"
                        and self._is_partial_after_final(request)
                    ):
                        
                        logger.info(
                            "[ASR Scheduler] | ASR Worker (%s) | Dropping partial after the final | Session=%s | Sequence=%s", 
                            worker_id, request.session_id, request.sequence
                        )

                        if not future.done():
                            future.set_result("")

                        continue                    

                    job_started = True
                    self._active_jobs += 1

                    queue_wait = time.monotonic() - submitted_at
                    inference_started = time.monotonic()
                    
                    logger.info(
                        "[ASR Scheduler] | ASR Worker (%s) | Processing %s | Session=%s | Active=%s | QueueWait=%.2fs", 
                        worker_id, request.request_type, request.session_id, self._active_jobs, queue_wait
                    )
                   
                    text = await self._transcribe(request)                    

                    inference_time = time.monotonic() - inference_started                    
                    total_time = time.monotonic() - submitted_at

                    if (
                        request.request_type == "partial"
                        and self._is_partial_after_final(request)
                    ):
                        
                        logger.info(
                            "[ASR Scheduler] | ASR Worker (%s) | Discarding partial after the final | Session=%s | Sequence=%s", 
                            worker_id, request.session_id, request.sequence
                        )

                        if not future.done():
                            future.set_result("")

                        continue

                    if not future.done():
                        future.set_result(text)

                    logger.info(
                        "[ASR Scheduler] | ASR Worker (%s) | Completed %s | Session=%s | QueueWait=%.2fs | InferenceTime=%.2fs | TotalTime=%.2fs", 
                        worker_id, request.request_type, request.session_id, queue_wait, inference_time, total_time
                    )
                except asyncio.CancelledError:
                    if not future.done():
                        future.cancel()
                    raise
                except Exception as e:
                    logger.exception("[ASR Scheduler] | ASR Worker (%s) | Error occurred | Session=%s", worker_id, request.session_id)                    
                    if not future.done():
                        future.set_exception(e)
                finally:
                    if job_started:
                        self._active_jobs -= 1
                    self.queue.task_done()

        except asyncio.CancelledError:
            logger.exception("[ASR Scheduler] | ASR Worker (%s) Stopped", worker_id)
            raise

    async def _transcribe(
        self,
        request: AsrRequest,
    ) -> str:
        """
        Execute the configured ASR service.
        Supports both:
        1. Synchronous services
           Example: local Faster-Whisper

        2. Asynchronous services
           Example: Kaggle WebSocket ASR
        """

        if request.request_type == "partial":
            method = request.asr_service.transcribe_partial
        else:
            method = request.asr_service.transcribe

        return await method(request.audio)

    def _is_partial_after_final(
        self,
        request: AsrRequest,
    ) -> bool:
        final_sequence = (
            self._final_sequences.get(
                request.session_id
            )
        )

        if final_sequence is None:
            return False

        return (
            request.sequence
            < final_sequence
        )
        

    def cancel_session(
        self,
        session_id: str,
    ):
        """
        Invalidate pending PARTIAL requests belonging to a disconnected session.
        A Whisper inference already running in a background thread cannot be forcibly killed.
        """
        self._final_sequences[
            session_id
        ] = next(
            self._sequence_counter
        )

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def active_jobs(self) -> int:
        return self._active_jobs