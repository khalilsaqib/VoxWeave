import asyncio
from typing import Optional
from livekit import rtc
from app.services.tts.publisher import TtsAudioPublisher
from app.services.tts.worker import TtsWorker
from app.services.tts.base import TtsServiceBase
from app.services.asr.service import SttServiceBase
from app.core.logger import logger

class VoiceSession:
    def __init__(
        self,
        session_id: str,
        room_name: str
    ):
        self.session_id = session_id
        self.room_name = room_name
        
        # LiveKit resources
        self.room: Optional[rtc.Room] = None
        self.tts_publisher: Optional[TtsAudioPublisher] = None

        #STT
        self.asr_service: SttServiceBase | None = None

        # TTS
        self.tts_queue: asyncio.Queue[str] = asyncio.Queue()
        self.tts_worker: Optional[TtsWorker] = None
        self.tts_service: TtsServiceBase | None = None

        # LLM generation
        self.llm_task: asyncio.Task | None = None

        # Turn / generation state
        self.generation_id = 0
        self._pending_generation_id: int | None = None
        self._generation_lock = asyncio.Lock()
        self.is_interrupted = False

        # Session tasks
        self.session_task: Optional[asyncio.Task] = None
        self.audio_tasks: set[asyncio.Task] = set()

        # Lifecycle state
        self.cleanup_started = False

        self.ready_event = asyncio.Event()
        self.startup_error: Exception | None = None

    def is_generation_current(
        self,
        generation_id: int,
    ) -> bool:
        return generation_id == self.generation_id

    # =====================================================
    # Barge-in
    # =====================================================
    async def interrupt_response(self):

        async with self._generation_lock:
            generation_id = self.generation_id
            
            if generation_id <= 0 or self.is_interrupted:
                return False, generation_id
            

            self.is_interrupted = True
            self._pending_generation_id = generation_id + 1

            old_llm_task = self.llm_task
            self.llm_task = None
        logger.info("[Voice Session] | InterruptResponse | Start Interrupting | Session=%s | Generation=%s", self.session_id, generation_id)        

        # =========================================================
        # 1. Cancel LLM
        # =========================================================

        if (
            old_llm_task is not None 
            and not old_llm_task.done()
        ):
            logger.info("[Voice Session] | InterruptResponse | Cancelling LLM task | Session=%s | Generation=%s", self.session_id, generation_id)
            old_llm_task.cancel()    
                
            await asyncio.gather(
                old_llm_task,
                return_exceptions=True,
            )

        # =========================================================
        # 2. Cancel TTS generation
        # =========================================================

        if self.tts_service is not None:
            logger.info("[Voice Session] | InterruptResponse | Cancelling TTS generation | Session=%s | Generation=%s", self.session_id, generation_id)            

            await self.tts_service.cancel_generation(
                generation_id
            )

        # -------------------------------------------------
        # 3. Clear any remaining TTS queue items
        # -------------------------------------------------

        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

         # -------------------------------------------------
        # 4. Stop current TTS playback
        # -------------------------------------------------
        
        if self.tts_publisher is not None:
            logger.info("[Voice Session] | InterruptResponse | Stopping TTS publisher | Session=%s | Generation=%s", self.session_id, generation_id)
            # await self.tts_worker.stop()
            self.tts_publisher.flush(self._pending_generation_id)
                
        # -------------------------------------------------
        # 5. Stop current TTS playback
        # -------------------------------------------------
        
        if self.tts_worker is not None:
            logger.info("[Voice Session] | InterruptResponse | Stopping TTS worker | Session=%s | Generation=%s", self.session_id, generation_id)
            # await self.tts_worker.stop()
            self.tts_worker.flush()
        
        return True, generation_id
        
    async def barge_in(self):

        logger.info("[Voice Session] | BargeIn | Triggered | Session=%s | CurrentGeneration=%s", self.session_id, self.generation_id)

        async with self._generation_lock:
            old_generation_id = self.generation_id

        if old_generation_id == 0:
            self.generation_id = 1
            new_generation_id = 1
            self.is_interrupted = False
            logger.info("[Voice Session] | BargeIn | First chat | Session=%s | NewGeneration=%s", self.session_id, new_generation_id)            
            return old_generation_id, new_generation_id

        if not self.is_interrupted:
            await self.interrupt_response()

        async with self._generation_lock:
            if self.generation_id != old_generation_id:
                return old_generation_id, self.generation_id

            if self._pending_generation_id is not None:
                self.generation_id = self._pending_generation_id
                self._pending_generation_id = None
            else:
                self.generation_id += 1

            new_generation_id = self.generation_id
            self.is_interrupted = False

        logger.info("[Voice Session] | BargeIn | Completed | Session=%s | OldGeneration=%s | NewGeneration=%s", self.session_id, old_generation_id, new_generation_id)
        
        return old_generation_id, new_generation_id

    # =====================================================
    # Audio task management
    # =====================================================

    def add_audio_task(
        self,
        task: asyncio.Task,
    ):
        self.audio_tasks.add(task)
        task.add_done_callback(
            self._on_audio_task_done
        )

    def _on_audio_task_done(
        self,
        task: asyncio.Task,
    ):
        self.audio_tasks.discard(task)

        if task.cancelled():
            logger.info("[Voice Session] | OnAudioTaskDone | Task cancelled | Session=%s", self.session_id)
            return
        
        exception = task.exception()
        if exception is not None:
            logger.exception("[Voice Session] | OnAudioTaskDone | Task failed. Error occurred | Session=%s", self.session_id)            
        else:
            logger.info("[Voice Session] | OnAudioTaskDone | Task Finished | Session=%s", self.session_id)

    async def cancel_audio_tasks(self):
        if not self.audio_tasks:
            return

        logger.info("[Voice Session] | CancelAudioTasks | Cancelling %s audio tasks | Session=%s",len(self.audio_tasks), self.session_id)
        
        tasks = list(self.audio_tasks)

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        self.audio_tasks.clear()
        logger.info("[Voice Session] | CancelAudioTasks | Audio tasks cleaned up | Session=%s", self.session_id)