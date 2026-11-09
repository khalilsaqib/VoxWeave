import uvicorn
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pathlib import Path
from app.api.routes import create_router
from app.config.settings import settings
from app.services.asr.scheduler import AsrScheduler
from app.services.audio_processor.service import AudioProcessor
from app.services.livekit.service import LiveKitService
from app.services.session.manager_service import SessionManager
from app.services.vad.service import VadService
from app.services.llm.service import LlmService
from app.services.session.conversation_service import ConversationManager
from app.core.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "client"
INDEX_FILE = STATIC_DIR / "index.html"

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")


def create_components():

    logger.info("[Main] | Creating required services for the application")
    logger.info("[Main] | LiveKit WebRTC service url=%s", settings.livekit_url)
    logger.info("[Main] | Speech to text service url=%s", settings.stt_ws_url)
    logger.info("[Main] | Text to speech service url=%s", settings.tts_ws_url)

    # --------------------------------------------
    # LLM
    # --------------------------------------------

    llm_service = LlmService(settings=settings)

    # --------------------------------------------
    # Conversation Manager
    # --------------------------------------------

    conversation_manager = ConversationManager(llm_service=llm_service)

    # --------------------------------------------
    # VAD
    # --------------------------------------------

    vad_service = VadService(
        threshold=0.5, min_silence_duration_ms=750, speech_pad_ms=250
    )

    # --------------------------------------------
    # CENTRAL ASR SCHEDULER
    # --------------------------------------------

    asr_scheduler = AsrScheduler(
        max_workers=1,
        max_queue_size=100,
    )

    # --------------------------------------------
    # AUDIO PROCESSOR
    # --------------------------------------------

    audio_processor = AudioProcessor(
        vad_service=vad_service,
        asr_scheduler=asr_scheduler,
        llm_service=llm_service,
        conversation_manager=conversation_manager
    )

    # --------------------------------------------
    # LIVEKIT
    # --------------------------------------------

    livekit_service = LiveKitService(settings=settings)

    # --------------------------------------------
    # SESSION MANAGER
    # --------------------------------------------

    session_manager = SessionManager(
        livekit_service=livekit_service,
        audio_processor=audio_processor,
        asr_scheduler=asr_scheduler,
    )

    return (
        asr_scheduler,
        session_manager,
        livekit_service,
        conversation_manager
    )


(
    asr_scheduler,
    session_manager,
    livekit_service,
    conversation_manager
) = create_components()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --------------------------------------------
    # STARTUP
    # --------------------------------------------

    await asr_scheduler.start()

    logger.info(
        "[Main] | Application is ready. Open frontend at http://%s:%s",
        settings.worker_host,
        settings.worker_port,
    )

    try:
        yield
    finally:
        logger.info("[Main] | Application is shutting down.")
        session_ids = list(session_manager.sessions.keys())

        for session_id in session_ids:
            try:
                await session_manager.cleanup_session(session_id)
                await conversation_manager.delete_session(session_id)
            except Exception:
                logger.exception(
                    "[Main] | Error occurred while cleaning up session | Session=%s",
                    session_id,
                )

        await asr_scheduler.stop()
        logger.info("[Main] | Application shutdown complete.")


def create_app() -> FastAPI:
    # ----------------------------------------
    # FastAPI
    # ----------------------------------------

    app = FastAPI(
        title="Voice Chat Application",
        lifespan=lifespan,
    )

    app.include_router(
        create_router(session_manager, livekit_service, conversation_manager)
    )

    if not STATIC_DIR.exists():
        logger.info(
            "[Main] | Warning: Static directory not found | Path=%s", STATIC_DIR
        )
    else:
        logger.info("[Main] | Mounting static directory | Path=%s", STATIC_DIR)
        app.mount(
            "/client",
            StaticFiles(directory=STATIC_DIR),
            name="client",
        )

    @app.get("/")
    async def frontend():

        if not INDEX_FILE.exists():
            logger.info("[Main] | Warning: Frontend not found | Expected=%s", INDEX_FILE)
            return {"error": ("Frontend not found. " f"Expected: {INDEX_FILE}")} 

        return FileResponse(
            INDEX_FILE, headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.worker_host,
        port=settings.worker_port,
    )
