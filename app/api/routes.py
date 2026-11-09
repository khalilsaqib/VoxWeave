import uuid
import asyncio
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.services.session.manager_service import SessionManager
from app.services.session.conversation_service import ConversationManager
from app.services.livekit.service import LiveKitService
from app.config.settings import settings
from app.core.logger import logger

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

def create_router(
    session_manager: SessionManager,
    livekit_service: LiveKitService,
    conversation_manager: ConversationManager,
) -> APIRouter:

    router = APIRouter()

    @router.post("/api/session")
    async def create_new_session():
        session = await conversation_manager.create_session()
        logger.info("[Routes] | /api/session | New session created | Session=%s", session.session_id)
        return {
            "sessionId": session.session_id,
        }

    @router.post("/api/sessions/{session_id}/chat")
    async def general_chat(
        session_id: str,
        body: ChatRequest,
        request: Request,
    ):
        logger.info("[Routes] | /api/sessions/chat | New chat request | Session=%s | Data=%s", session_id, body.model_dump_json())
        session = conversation_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation session not found",
            )
        
        prompt = body.message.strip()

        if not prompt:
            raise HTTPException(
                status_code=422,
                detail="Message cannot be empty",
            )
        
        llm_service = conversation_manager.llm_service

        async def generate():
            try:
                async with session.turn_lock:
                    if conversation_manager.get_session(session_id) is not session:
                        yield format_sse({"error": "Conversation session not found"})
                        yield "data: [DONE]\n\n"
                        return                   

                    if await request.is_disconnected():
                        return                 

                    async for chunk in llm_service.stream(
                        user_text=prompt,
                    ):
                        if await request.is_disconnected():
                            return

                        if chunk:
                            yield format_sse({"token": chunk})

                    if await request.is_disconnected():
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[Routes] | /api/sessions/chat | Failed to generate chat response | Session=%s", session_id)
                yield format_sse({"error": "Response generation failed"})

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/api/sessions/{session_id}/voice")
    async def get_livekit_token(
        session_id: str,
    ):
        logger.info("[Routes] | /api/sessions/voice | New voice session request | Session=%s", session_id)
        conversation = conversation_manager.get_session(session_id)

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation session not found",
            )

        if session_id in session_manager.sessions:
            raise HTTPException(
                status_code=409,
                detail="Voice session already exists",
            )

        identity = f"client-{uuid.uuid4().hex}"
        room_name = f"voice-{session_id}"

        logger.info("[Routes] | /api/sessions/voice | Creating new voice session | Session=%s | Identity=%s | Room=%s", session_id, identity, room_name)        

        token = livekit_service.create_client_token(
            room_name=room_name,
            identity=identity,
        )

        try:
            voice_session = await session_manager.start_session(
                session_id=session_id, room_name=room_name
            )

            await voice_session.ready_event.wait()
            if voice_session.startup_error is not None:
                raise RuntimeError(
                    "Voice session initialization failed"
                ) from voice_session.startup_error

        except Exception as exc:
            if session_id in session_manager.sessions:
                await session_manager.cleanup_session(session_id)
            logger.exception("[Routes] | /api/sessions/voice | Failed to create a new voice session | Session=%s", session_id)
            raise HTTPException(
                status_code=503,
                detail="Voice service could not be started",
            ) from exc

        return {
            "url": settings.livekit_url,
            "token": token,
            "sessionId": session_id,
            "room": room_name,
            "identity": identity,
        }

    @router.delete("/api/sessions/{session_id}")
    async def stop_session(
        session_id: str,
    ):
        logger.info("[Routes] | /api/sessions/delete | Stopping existing session | Session=%s", session_id)

        try:
            await session_manager.stop_session(session_id)
            deleted = await conversation_manager.delete_session(session_id)
            if not deleted:
                logger.info("[Routes] | /api/sessions/delete | Conversation session not found | Session=%s", session_id)
                raise HTTPException(
                    status_code=404,
                    detail="Conversation session not found",
                )

        except KeyError:
            logger.exception("[Routes] | /api/sessions/delete | Failed to delete session | Session=%s", session_id)
            raise HTTPException(
                status_code=404,
                detail="Session not found",
            )

        return {
            "success": True,
            "sessionId": session_id,
        }
    
    @router.delete("/api/voice_sessions/{session_id}")
    async def stop_voice_session(
        session_id: str,
    ):
        logger.info("[Routes] | /api/voice_sessions/delete | Stopping existing voice session | Session=%s", session_id)

        try:
            await session_manager.stop_session(session_id)   
        except KeyError:
            logger.exception("[Routes] | /api/voice_sessions/delete | Failed to delete voice session | Session=%s", session_id)
            raise HTTPException(
                status_code=404,
                detail="Session not found",
            )

        return {
            "success": True,
            "sessionId": session_id,
        }
    
    @router.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "activeSessions": session_manager.active_count,
        }

    return router


def format_sse(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"