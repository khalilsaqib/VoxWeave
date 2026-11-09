import asyncio
from livekit import rtc

from app.services.asr.scheduler import AsrScheduler
from app.services.audio_processor.service import AudioProcessor
from app.services.livekit.service import LiveKitService
from app.services.session.voice_service import VoiceSession
from app.services.tts.publisher import TtsAudioPublisher
from app.services.tts.worker import TtsWorker
from app.services.asr.service import AsrService
from app.services.tts.service import TtsService
from app.config.settings import settings
from app.core.logger import logger

class SessionManager:
    def __init__(
        self,
        livekit_service: LiveKitService,
        audio_processor: AudioProcessor,
        asr_scheduler: AsrScheduler
    ):
        self.livekit_service = livekit_service
        self.audio_processor = audio_processor
        self.asr_scheduler = asr_scheduler

        self.sessions: dict[str, VoiceSession] = {}

    async def start_session(
        self,
        session_id: str,
        room_name: str
    ) -> VoiceSession:

        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        session = VoiceSession(
            session_id=session_id,
            room_name=room_name,
        )
        self.sessions[session_id] = session

        task = asyncio.create_task(self._run_session(session))

        session.session_task = task
        return session

    async def _run_session(
        self,
        session: VoiceSession,
    ):
        session_id = session.session_id
        try:

            # ------------------------------------------------
            # 1. Connect to LiveKit
            # ------------------------------------------------
            room = await self.livekit_service.connect(
                session.room_name,
                session.session_id,
            )
            session.room = room

            # ------------------------------------------------
            # 2. Start Audio Publisher
            # ------------------------------------------------
            session.tts_publisher = TtsAudioPublisher(
                sample_rate=settings.tts_sample_rate,
                num_channels=1,
                frame_duration_ms=20,
            )

            await room.local_participant.publish_track(session.tts_publisher.track)

            logger.info("[Session Manager] | RunSession | AI audio track published | Session=%s", session_id)

            # ------------------------------------------------
            # 3. Start TTS Service
            # ------------------------------------------------
            tts_service = TtsService(
                websocket_url=settings.tts_ws_url,
                sample_rate=settings.tts_sample_rate,
            )
            session.tts_service = tts_service

            await session.tts_service.start()

            session.tts_worker = TtsWorker(
                session_id=session_id,
                queue=session.tts_queue,
                tts_service=session.tts_service,
                publisher=session.tts_publisher,
                generation_id_getter=lambda: session.generation_id,
            )

            session.tts_worker.start()

            # ------------------------------------------------
            # 4. Start STT Service
            # ------------------------------------------------
            asr_service = AsrService(
                websocket_url=settings.stt_ws_url,
                language=None,
            )
            session.asr_service = asr_service
            await session.asr_service.start()

            # ------------------------------------------------
            # 5. Register LiveKit Events
            # ------------------------------------------------
            self._register_events(session)           

            session.ready_event.set()
            await asyncio.Event().wait()

        except asyncio.CancelledError:
            logger.error("[Session Manager] | RunSession | Cancelled | Session=%s", session_id)
            raise
        except Exception as e:
            logger.exception("[Session Manager] | RunSession | Error occurred | Session=%s", session_id)            
            
            session.startup_error = e
            session.ready_event.set()

        finally:
            await self.cleanup_session(
                session_id,
                cancel_session_task=False,
            )

    def _register_events(
        self,
        session: VoiceSession,
    ):
        room = session.room

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            
            logger.info(
                "[Session Manager] | RegisterEvents | OnTrackSubscribed | Session=%s | Participant=%s | Track=%s | Kind=%s",
                session.session_id, participant.identity, track.sid, track.kind
            )

            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            
            logger.info("[Session Manager] | RegisterEvents | OnTrackSubscribed | Subscribed to audio from=%s", participant.identity)

            task = asyncio.create_task(
                self.audio_processor.process(session, participant.identity, track, room)
            )

            session.add_audio_task(task)

            logger.info("[Session Manager] | RegisterEvents | OnTrackSubscribed | Processing task started | Session=%s | Track=%s", session.session_id, track.sid)
            

        @room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            logger.info(
                "[Session Manager] | RegisterEvents | OnTrackUnsubscribed | Session=%s | Participant=%s | Track=%s",
                session.session_id, participant.identity, track.sid
            )

        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(
                "[Session Manager] | RegisterEvents | OnParticipantConnected | Connected | Session=%s | Participant=%s",
                session.session_id, participant.identity
            )

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(
                "[Session Manager] | RegisterEvents | OnParticipantDisconnected | Disconnected | Session=%s | Participant=%s",
                session.session_id, participant.identity
            )
            asyncio.create_task(self.cleanup_session(session.session_id))

    async def cleanup_session(
        self,
        session_id: str,
        cancel_session_task: bool = True,
    ):
        session = self.sessions.get(session_id)

        if session is None:
            return

        if session.cleanup_started:
            return

        session.cleanup_started = True

        logger.info("[Session Manager] | CleanupSession | Cleaning up | Session=%s", session_id)

        self.asr_scheduler.cancel_session(session_id)
        current_task = asyncio.current_task()

        if (
            cancel_session_task
            and session.session_task is not None
            and session.session_task is not current_task
            and not session.session_task.done()
        ):
            
            logger.info("[Session Manager] | CleanupSession | Cancelling session task | Session=%s", session_id)

            session.session_task.cancel()

            await asyncio.gather(
                session.session_task,
                return_exceptions=True,
            )

        await session.cancel_audio_tasks()

        if (
            session.llm_task is not None 
            and not session.llm_task.done()
        ):            
            logger.info("[Session Manager] | CleanupSession | Cancelling LLM task | Session=%s", session_id)
            session.llm_task.cancel()    
                
            await asyncio.gather(
                session.llm_task,
                return_exceptions=True,
            )

        if session.tts_worker is not None:
            try:
                await session.tts_worker.stop()

            except Exception:
                logger.exception("[Session Manager] | CleanupSession | Worker cleanup error | Session=%s", session_id)                

            finally:
                session.tts_worker = None

        if session.tts_service is not None:
            try:
                await session.tts_service.close()
            except Exception:
                logger.exception("[Session Manager] | CleanupSession | TTS service cleanup error | Session=%s", session_id)
            finally:
                session.tts_service = None

        if session.tts_publisher is not None:
            try:
                await session.tts_publisher.close()
            except Exception as e:
                logger.exception("[Session Manager] | CleanupSession | TTS publisher cleanup error | Session=%s", session_id)
            finally:
                session.tts_publisher = None

        if session.asr_service is not None:
            try:
                await session.asr_service.close()
            except Exception:
                logger.exception("[Session Manager] | CleanupSession | ASR service cleanup error | Session=%s", session_id)                
            finally:
                session.asr_service = None

        if session.room is not None:
            try:
                await session.room.disconnect()

            except Exception as e:
                logger.exception("[Session Manager] | CleanupSession | Livekit disconnect error | Session=%s", session_id)                

            session.room = None

        self.sessions.pop(
            session_id,
            None,
        )

        logger.info("[Session Manager] | CleanupSession | Cleanup complete | Session=%s", session_id)      

    async def stop_session(
        self,
        session_id: str,
    ):
        if session_id not in self.sessions:
            return            

        await self.cleanup_session(session_id)

    def get_session(
        self,
        session_id: str,
    ) -> VoiceSession | None:

        return self.sessions.get(session_id)

    @property
    def active_count(self) -> int:
        return len(self.sessions)
