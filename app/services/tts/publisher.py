import asyncio
import numpy as np
from livekit import rtc
from app.core.logger import logger


class TtsAudioPublisher:
    def __init__(
        self,
        sample_rate: int = 24000,
        num_channels: int = 1,
        frame_duration_ms: int = 20,
        max_buffer_ms: int = 2000,
    ):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frame_duration_ms = frame_duration_ms

        self.samples_per_frame = int(self.sample_rate * self.frame_duration_ms / 1000)

        # Maximum amount of audio allowed to sit inside the
        # publisher before backpressure reaches the TTS worker.
        self.max_buffer_frames = max(
            1,
            int(max_buffer_ms / frame_duration_ms),
        )

        self.source = rtc.AudioSource(
            self.sample_rate,
            self.num_channels,
        )

        self.track = rtc.LocalAudioTrack.create_audio_track(
            "ai-tts",
            self.source,
        )

        self._frame_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(
            maxsize=self.max_buffer_frames
        )

        self._publisher_task: asyncio.Task | None = None
        self._closing = False
        self._current_generation: int | None = None
        logger.info(
            "[TTS Audio Publisher] | Service Created | SampleRate=%s | Channels=%s | SamplesPerFrame=%s | MaxBuffer=%sms",
            self.sample_rate,
            self.num_channels,
            self.samples_per_frame,
            max_buffer_ms,
        )

    def _ensure_started(self):
        if self._publisher_task is not None:
            return

        self._publisher_task = asyncio.create_task(self._publish_loop())
        logger.info("[TTS Audio Publisher] | EnsureStarted | Publish loop started")

    def flush(self, generation_id: int):
        """
        Invalidate all earlier generations and discard queued frames.

        Call this immediately during barge-in, after updating the
        session's current generation ID.
        """
        self._current_generation = generation_id

        dropped = 0

        while True:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
                dropped += 1
            except asyncio.QueueEmpty:
                break

        logger.info(
            "[TTS Audio Publisher] | Flush | Successfully flushed | NewGeneration=%s | DroppedFrames=%s",
            generation_id,
            dropped,
        )

    async def publish(
        self,
        audio: np.ndarray,
        generation_id: int,
    ):
        """
        Accept PCM16 audio and enqueue it as 20ms LiveKit frames.
        Input:
            np.ndarray(dtype=np.int16)
        """
        if self._closing:
            raise RuntimeError("TTS publisher is closed")

        if audio is None or audio.size == 0:
            return

        if self._current_generation is None:
            self._current_generation = generation_id

        if generation_id != self._current_generation:
            logger.info(
                "[TTS Audio Publisher] | Publish | Reject stale audio | AudioGeneration=%s | CurrentGeneration=%s",
                generation_id,
                self._current_generation,
            )
            return

        self._ensure_started()

        audio_pcm16 = np.asarray(audio, dtype=np.int16)
        total_samples = len(audio_pcm16)
        offset = 0

        while offset < total_samples:
            if self._closing or generation_id != self._current_generation:
                return

            end = min(
                offset + self.samples_per_frame,
                total_samples,
            )

            frame_data = audio_pcm16[offset:end].copy()
            if frame_data.size == 0:
                break

            await self._frame_queue.put((generation_id, frame_data))
            offset = end

    async def _publish_loop(self):
        """
        Continuously feed 20ms PCM16 frames into LiveKit.
        """

        try:
            while not self._closing:
                item = await self._frame_queue.get()

                try:
                    if item is None:
                        break

                    generation_id, frame_data = item

                    if generation_id != self._current_generation:
                        continue

                    frame = rtc.AudioFrame(
                        data=frame_data.tobytes(),
                        sample_rate=self.sample_rate,
                        num_channels=self.num_channels,
                        samples_per_channel=len(frame_data),
                    )

                    await self.source.capture_frame(frame)

                finally:
                    self._frame_queue.task_done()

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception("[TTS Audio Publisher] | PublishLoop | Publish loop error occurred")

        finally:
            logger.info("[TTS Audio Publisher] | PublishLoop | Publish loop stopped")

    async def publish_and_wait(
        self,
        audio: np.ndarray,
        generation_id: int,
    ):
        await self.publish(audio, generation_id)

    async def close(self):
        if self._closing:
            return

        self._closing = True

        logger.info("[TTS Audio Publisher] | Close | Closing audio publisher")

        if self._publisher_task is not None:
            self._publisher_task.cancel()

            await asyncio.gather(
                self._publisher_task,
                return_exceptions=True,
            )

            self._publisher_task = None

        # Clear anything that was waiting.
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
            except asyncio.QueueEmpty:
                break

        logger.info("[TTS Audio Publisher] | Close | Audio publisher closed")
