import asyncio
import json
import time
from collections import deque
import numpy as np
import torch
import torchaudio
from livekit import rtc
from livekit.plugins import dtln
from app.services.asr.scheduler import AsrScheduler
from app.services.vad.service import VadService
from app.services.llm.service import LlmService
from app.audio.buffer import AudioBuffer
from app.services.tts.text_segmenter import TextSegmenter
from app.services.session.voice_service import VoiceSession
from app.core.logger import logger
from app.services.session.conversation_service import ConversationManager

class AudioProcessor:
    def __init__(
        self,
        vad_service: VadService,
        asr_scheduler: AsrScheduler,
        llm_service: LlmService,
        conversation_manager: ConversationManager,
        sample_rate: int = 16000,
        partial_interval_seconds: float = 0.4,
        partial_window_seconds: float = 1.5,
        pre_roll_seconds: float = 0.25,
        min_rms_threshold: float = 0.007,
    ):
        self.vad_service = vad_service
        self.asr_scheduler = asr_scheduler
        self.llm_service = llm_service
        self.conversation_manager = conversation_manager

        self.sample_rate = sample_rate
        self.partial_interval_seconds = partial_interval_seconds
        self.partial_window_seconds = partial_window_seconds
        self.pre_roll_seconds = pre_roll_seconds
        self.min_rms_threshold = min_rms_threshold

        logger.info(
            "[Audio Processor] | Service created | SampleRate=%s | PartialInterval=%.2fs | PartialWindow=%.2fs | PreRoll=%.2fs",
            self.sample_rate,
            self.partial_interval_seconds,
            self.partial_window_seconds,
            self.pre_roll_seconds,
        )

    async def process(
        self,
        session: VoiceSession,
        participant_identity: str,
        track: rtc.Track,
        room: rtc.Room,
    ):

        session_id = session.session_id
        logger.info(
            "[Audio Processor] | Process | Audio track received from: %s | Session=%s | TrackSID=%s",
            participant_identity,
            session_id,
            track.sid,
        )

        audio_stream = rtc.AudioStream(
            track,
            sample_rate=48000,
            num_channels=1,
            noise_cancellation=dtln.noise_suppression(),
        )
        vad = self.vad_service.create_iterator()

        frame_count = 0
        speech_active = False
        vad_buffer = np.empty(0, dtype=np.float32)
        speech_buffer = AudioBuffer(
            sample_rate=self.sample_rate, max_duration_seconds=30.0
        )

        pre_roll_max_samples = int(self.sample_rate * self.pre_roll_seconds)

        pre_roll_chunks: deque[np.ndarray] = deque()
        pre_roll_sample_count = 0

        final_tasks: set[asyncio.Task] = set()

        current_utterance_id = 0
        active_utterance_id: int | None = None

        partial_lock = asyncio.Lock()
        partial_event = asyncio.Event()

        pending_partial: np.ndarray | None = None
        partial_worker_task: asyncio.Task | None = None
        last_partial_time = 0.0

        async def partial_worker():
            """
            Process the newest pending partial snapshot.
            There is never more than one of these workers running.
            """
            nonlocal pending_partial

            logger.info(
                "[Audio Processor] | Process | ASR partial worker started | Session=%s",
                session_id,
            )

            try:
                while True:
                    await partial_event.wait()
                    while True:
                        async with partial_lock:
                            if pending_partial is None:
                                partial_event.clear()
                                break

                            audio_snapshot = pending_partial
                            pending_partial = None

                        if isinstance(
                            audio_snapshot,
                            tuple,
                        ):
                            (
                                audio_snapshot,
                                snapshot_utterance_id,
                            ) = audio_snapshot
                        else:
                            snapshot_utterance_id = current_utterance_id

                        await self._submit_partial_asr(
                            session=session,
                            participant_identity=participant_identity,
                            room=room,
                            audio=audio_snapshot,
                            utterance_id=snapshot_utterance_id,
                            active_utterance_id_getter=(lambda: active_utterance_id),
                        )

                        async with partial_lock:
                            if pending_partial is None:
                                partial_event.clear()
                                break

            except asyncio.CancelledError:
                logger.info(
                    "[Audio Processor] | Process | ASR partial worker cancelled | Session=%s",
                    session_id,
                )
                raise

            except Exception:
                logger.exception(
                    "[Audio Processor] | Process | ASR partial worker error occurred | Session=%s",
                    session_id,
                )
                raise

            finally:
                logger.info(
                    "[Audio Processor] | Process | ASR partial worker stopped | Session=%s",
                    session_id,
                )

        async def schedule_partial(
            audio_snapshot: np.ndarray,
            utterance_id: int,
        ):
            """
            Store the newest partial snapshot and make sure a partial worker is running.
            """
            nonlocal pending_partial
            nonlocal partial_worker_task

            async with partial_lock:
                pending_partial = (
                    audio_snapshot.copy(),
                    utterance_id,
                )
                partial_event.set()
                if partial_worker_task is None or partial_worker_task.done():
                    partial_worker_task = asyncio.create_task(partial_worker())

        try:
            async for event in audio_stream:
                frame = event.frame
                frame_count += 1

                samples = frame.samples_per_channel
                sample_rate = frame.sample_rate
                channels = frame.num_channels

                if frame_count == 1:
                    logger.info(
                        "[Audio Processor] | Process | First audio frame received | Session=%s | SampleRate=%s | Channels=%s | Samples=%s",
                        session_id,
                        sample_rate,
                        channels,
                        samples,
                    )

                pcm = np.frombuffer(frame.data, dtype=np.int16)

                if channels > 1:
                    pcm = pcm.reshape(-1, channels)
                    pcm = pcm[:, 0]

                audio = torch.from_numpy(pcm.astype(np.float32, copy=False))
                audio = audio / 32768.0
                audio = audio.unsqueeze(0)

                if sample_rate != self.sample_rate:
                    audio = torchaudio.functional.resample(
                        audio, sample_rate, self.sample_rate
                    )

                audio_16k = audio.squeeze(0).numpy()

                vad_buffer = np.concatenate((vad_buffer, audio_16k))
                pre_roll_chunks.append(audio_16k)
                pre_roll_sample_count += len(audio_16k)

                while pre_roll_sample_count > pre_roll_max_samples and pre_roll_chunks:
                    excess = pre_roll_sample_count - pre_roll_max_samples
                    first = pre_roll_chunks[0]

                    if len(first) <= excess:
                        pre_roll_chunks.popleft()
                        pre_roll_sample_count -= len(first)
                    else:
                        pre_roll_chunks[0] = first[excess:]
                        pre_roll_sample_count -= excess

                while len(vad_buffer) >= 512:
                    vad_chunk = vad_buffer[:512]
                    vad_buffer = vad_buffer[512:]

                    vad_result = self.vad_service.process(vad, vad_chunk)

                    speech_started = vad_result is not None and "start" in vad_result

                    speech_ended = vad_result is not None and "end" in vad_result

                    rms_volume = np.sqrt(np.mean(np.square(vad_chunk)))

                    if speech_started and not speech_active:
                        if rms_volume > self.min_rms_threshold:
                            speech_active = True
                            speech_buffer.clear()
                            current_utterance_id += 1
                            utterance_id = current_utterance_id
                            active_utterance_id = utterance_id
                            self.asr_scheduler.begin_utterance(session_id)

                            if pre_roll_chunks:
                                pre_roll_audio = np.concatenate(tuple(pre_roll_chunks))

                                if len(pre_roll_audio) > pre_roll_max_samples:
                                    pre_roll_audio = pre_roll_audio[
                                        -pre_roll_max_samples:
                                    ]

                                speech_buffer.append(pre_roll_audio)

                            if speech_buffer.sample_count() == 0:
                                speech_buffer.append(vad_chunk)

                            last_partial_time = time.monotonic()
                            logger.info(
                                "[Audio Processor] | Process | 🟢 Speech started | Session=%s | Utterance=%s | RMS=%.4f",
                                session_id,
                                utterance_id,
                                rms_volume,
                            )

                        else:
                            vad.reset_states()
                            logger.info(
                                "[Audio Processor] | Process | ❌ Speech rejected | RMS=%.4f",
                                rms_volume,
                            )

                    if speech_active:
                        if not speech_started:
                            speech_buffer.append(vad_chunk)

                        now = time.monotonic()
                        elapsed = now - last_partial_time

                        if elapsed >= self.partial_interval_seconds:
                            last_partial_time = now

                            audio_snapshot = speech_buffer.get_audio(
                                max_duration_seconds=(self.partial_window_seconds)
                            )

                            if len(audio_snapshot) > 0:
                                await schedule_partial(
                                    audio_snapshot=(audio_snapshot),
                                    utterance_id=(utterance_id),
                                )

                    if speech_ended and speech_active:
                        speech_active = False
                        speech_audio = speech_buffer.get_audio()
                        speech_samples = len(speech_audio)
                        speech_duration = speech_samples / self.sample_rate

                        ended_utterance_id = active_utterance_id
                        active_utterance_id = None

                        logger.info(
                            "[Audio Processor] | Process | 🔴 Speech ended | Session=%s | Utterance=%s | Duration=%.2fs",
                            session_id,
                            ended_utterance_id,
                            speech_duration,
                        )

                        pre_roll_chunks.clear()
                        pre_roll_sample_count = 0

                        if speech_samples > 0:
                            final_task = asyncio.create_task(
                                self._submit_final_asr(
                                    session=session,
                                    participant_identity=participant_identity,
                                    room=room,
                                    audio=speech_audio,
                                    utterance_id=ended_utterance_id,
                                )
                            )
                            final_tasks.add(final_task)
                            final_task.add_done_callback(final_tasks.discard)

                        speech_buffer.clear()

        except asyncio.CancelledError:
            logger.info(
                "[Audio Processor] | Process | Audio processing cancelled | Session=%s",
                session_id,
            )
            raise
        except Exception:
            logger.exception(
                "[Audio Processor] | Process | Error occurred | Session=%s", session_id
            )
        finally:

            if partial_worker_task is not None and not partial_worker_task.done():
                partial_worker_task.cancel()

            if partial_worker_task is not None:
                await asyncio.gather(
                    partial_worker_task,
                    return_exceptions=True,
                )

            if final_tasks:
                for task in final_tasks:
                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    *final_tasks,
                    return_exceptions=True,
                )
                final_tasks.clear()

            try:
                await audio_stream.aclose()
            except Exception:
                logger.exception(
                    "[Audio Processor] | Process | Audio stream closing error | Session=%s",
                    session_id,
                )

            logger.info(
                "[Audio Processor] | Process | Audio stream closed | Session=%s",
                session_id,
            )

    async def _submit_partial_asr(
        self,
        session: VoiceSession,
        participant_identity: str,
        room: rtc.Room,
        audio: np.ndarray,
        utterance_id: int,
        active_utterance_id_getter,
    ):
        started = time.monotonic()
        session_id = session.session_id
        asr_service = session.asr_service
        try:
            text = await self.asr_scheduler.submit_partial(
                session_id, audio, asr_service
            )

            elapsed = time.monotonic() - started

            text = (text or "").strip()

            if not text:
                return

            current_id = active_utterance_id_getter()

            utterance_still_active = current_id == utterance_id
            logger.info(
                "[Audio Processor] | Submit Partial ASR | 🟡 Partial | Session=%s | Utterance=%s | %.2fs | Active=%s | Text=%r",
                session_id,
                utterance_id,
                elapsed,
                utterance_still_active,
                text,
            )

            if session.generation_id > 0 and utterance_still_active:
                interrupted, old_generation_id = await session.interrupt_response()
                if interrupted:    

                    await self._publish_llm_response(
                        room=room,
                        participant_identity=participant_identity,
                        message="",
                        status="interrupted",
                        generation_id=old_generation_id,
                    )

                    logger.info(
                        "[Audio Processor] | Submit Partial ASR | 🔇 Partial triggered barge-in | Session=%s | Generation=%s",
                        session_id,
                        session.generation_id,
                    )

            if not utterance_still_active:
                logger.info(
                    "[Audio Processor] | Submit Partial ASR | Suppressing partial after utterance close | Session=%s | Utterance=%s",
                    session_id,
                    utterance_id,
                )
                return

            await self._publish_transcript(
                room=room,
                participant_identity=participant_identity,
                status="partial",
                text=text,
            )

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[Audio Processor] | Submit Partial ASR | Error occurred | Session=%s",
                session_id,
            )

    async def _submit_final_asr(
        self,
        session: VoiceSession,
        participant_identity: str,
        room: rtc.Room,
        audio: np.ndarray,
        utterance_id: int,
    ):
        started = time.monotonic()
        session_id = session.session_id
        asr_service = session.asr_service
        try:
            text = await self.asr_scheduler.submit_final(session_id, audio, asr_service)

            elapsed = time.monotonic() - started

            text = (text or "").strip()

            if not text:
                return

            logger.info(
                "[Audio Processor] | Submit Final ASR | 📝 Final | Session=%s | Utterance=%s | %.2fs | Text=%r",
                session_id,
                utterance_id,
                elapsed,
                text,
            )

            old_generation_id, generation_id = await session.barge_in()

            # -----------------------------------------------------
            # Publish final user transcript
            # -----------------------------------------------------

            await self._publish_transcript(
                room=room,
                participant_identity=participant_identity,
                status="final",
                text=text,
            )

            # -----------------------------------------------------
            # Run LLM independently.
            # -----------------------------------------------------

            llm_task = asyncio.create_task(
                self._run_llm(
                    session=session,
                    participant_identity=participant_identity,
                    room=room,
                    user_text=text,
                    generation_id=generation_id,
                )
            )

            session.llm_task = llm_task

            # -----------------------------------------------------
            # Clean up reference when task finishes.
            # -----------------------------------------------------

            def _llm_done(task: asyncio.Task):
                if session.llm_task is task:
                    session.llm_task = None

                if task.cancelled():
                    return

                try:
                    exception = task.exception()
                    if exception is not None:
                        logger.exception(
                            "[Audio Processor] | Submit Final ASR | LLM task failed | Session=%s | Generation=%s",
                            session_id,
                            generation_id,
                        )

                except asyncio.CancelledError:
                    pass

            llm_task.add_done_callback(_llm_done)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[Audio Processor] | Submit Final ASR | Final transcription error occurred | Session=%s",
                session_id,
            )

    async def _publish_transcript(
        self,
        room: rtc.Room,
        participant_identity: str,
        status: str,
        text: str,
    ):
        message = {
            "type": "transcript",
            "status": status,
            "text": text,
        }

        payload = json.dumps(message)

        try:

            await room.local_participant.publish_data(
                payload,
                reliable=True,
                destination_identities=[participant_identity],
                topic="transcript",
            )
            logger.info(
                "[Audio Processor] | Publish Transcript | Sending transcript to frontend | Status=%s | ParticipantIdentity=%s | Text=%s",
                status,
                participant_identity,
                text,
            )

        except Exception:
            logger.exception(
                "[Audio Processor] | Publish Transcript| Error while sending transcript to frontend | Status=%s | ParticipantIdentity=%s",
                status,
                participant_identity,
            )

    async def _run_llm(
        self,
        session: VoiceSession,
        participant_identity: str,
        room: rtc.Room,
        user_text: str,
        generation_id: int,
    ):
        session_id = session.session_id
        segmenter = TextSegmenter(
            min_chars=40,
            max_chars=180,
        )
        buffer = ""
        last_flush = time.monotonic()

        conversation_session = self.conversation_manager.get_session(session_id)        
        if conversation_session is None:                
            await self._publish_llm_response(
                room=room,
                participant_identity=participant_identity,
                message="Conversation session not found for LLM processing.",
                status="error",
                generation_id=generation_id,
            )
            await session.tts_queue.put("Conversation session not found for LLM processing.")
            return

        await self.llm_service._add_message(
            session_id,
            role="user",
            content=user_text
        )       
       
        selected_stream_func = self.llm_service.stream
        kwargs = {
            "user_text": user_text,
            "is_voice": True,
        }      

        try:
            assistant_text = ""
            async for chunk in selected_stream_func(**kwargs):
                if not session.is_generation_current(generation_id):
                    logger.info(
                        "[Audio Processor] | Run LLM | Stale generation stopped | Session=%s | Generation=%s | CurrentGeneration=%s",
                        session_id,
                        generation_id,
                        session.generation_id,
                    )
                    return

                buffer += chunk
                now = time.monotonic()

                if now - last_flush >= 0.04:
                    if buffer:
                        # -----------------------------------------
                        # Only publish UI text if generation is
                        # still current.
                        # -----------------------------------------
                        if not session.is_generation_current(generation_id):
                            return                        

                        assistant_text += buffer
                        await self._publish_llm_response(
                            room=room,
                            participant_identity=participant_identity,
                            message=buffer,
                            status="delta",
                            generation_id=generation_id,
                        )

                        buffer = ""

                    last_flush = now

                segments = segmenter.add(chunk)

                for segment in segments:

                    # ---------------------------------------------
                    # Check again before sending text to TTS.
                    # ---------------------------------------------
                    if not session.is_generation_current(generation_id):
                        logger.info(
                            "[Audio Processor] | Run LLM | Ignoring stale TTS segment | Session=%s | Generation=%s",
                            session_id,
                            generation_id,
                        )
                        return

                    logger.info(
                        "[Audio Processor] | Run LLM | Text to speech segment | Session=%s | Generation=%s | Text=%s",
                        session_id,
                        generation_id,
                        segment,
                    )
                    await session.tts_queue.put(segment)

            # -----------------------------------------------------
            # Final LLM output
            # -----------------------------------------------------
            if not session.is_generation_current(generation_id):
                return

            # Flush anything remaining.
            if buffer:
                assistant_text += buffer
                await self._publish_llm_response(
                    room=room,
                    participant_identity=participant_identity,
                    message=buffer,
                    status="delta",
                    generation_id=generation_id,
                )

            remaining_segments = segmenter.flush()

            for segment in remaining_segments:
                if not session.is_generation_current(generation_id):
                    return

                logger.info(
                    "[Audio Processor] | Run LLM | Text to speech segment | Session=%s | Generation=%s | Text=%s",
                    session_id,
                    generation_id,
                    segment,
                )

                await session.tts_queue.put(segment)

            # -----------------------------------------------------
            # LLM complete
            # -----------------------------------------------------
            if not session.is_generation_current(generation_id):
                return
            
            await self._publish_llm_response(
                room=room,
                participant_identity=participant_identity,
                message="",
                status="done",
                generation_id=generation_id,
            )

            
            await self.llm_service._add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_text
            )

        except asyncio.CancelledError:
            logger.info(
                "[Audio Processor] | Run LLM | Execution cancelled | Session=%s | Generation=%s",
                session_id,
                generation_id,
            )
            raise

        except Exception as exc:
            logger.exception(
                "[Audio Processor] | Run LLM | Error occurred | Session=%s | Generation=%s",
                session_id,
                generation_id,
            )

            if session.is_generation_current(generation_id):
                await self._publish_llm_response(
                    room=room,
                    participant_identity=participant_identity,
                    message=str(exc),
                    status="error",
                    generation_id=generation_id,
                )

    async def _publish_llm_response(
        self,
        room: rtc.Room,
        participant_identity: str,
        message:str,
        status: str,
        generation_id: int,
    ):
        try:

            message = {
                "type": "llm_response",
                "status": status,
                "text": message,
                "generation_id": generation_id,
            }            

            await room.local_participant.publish_data(
                json.dumps(message).encode("utf-8"),
                reliable=True,
                topic="llm_response",
                destination_identities=[participant_identity],
            )

        except Exception:
            logger.exception(
                "[Audio Processor] | LiveKit | Error occurred while publishing LLM response | Generation=%s | ParticipantIdentity=%s",
                generation_id,
                participant_identity,
            )

    