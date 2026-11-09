import uuid
from app.services.llm.service import LlmService
from app.models.conversation_session import ConversationSession
from app.core.logger import logger


class ConversationManager:
    def __init__(self, llm_service: LlmService):
        self.llm_service = llm_service
        self.sessions: dict[str, ConversationSession] = {}

    async def create_session(self) -> ConversationSession:
        session_id = uuid.uuid4().hex
        session = ConversationSession(
            session_id=session_id,
        )

        self.sessions[session_id] = session
        logger.info("[Conversation Manager] | Session created | Session=%s", session_id)
        return session

    def get_session(
        self,
        session_id: str,
    ) -> ConversationSession | None:
        return self.sessions.get(session_id)

    async def delete_session(
        self,
        session_id: str,
    ) -> bool:
        session = self.sessions.get(session_id)

        if session is None:
            return False

        async with session.turn_lock:
            if self.sessions.get(session_id) is not session:
                return False
            self.sessions.pop(session_id, None)
            await self.llm_service.clear_session(session_id)

        logger.info("[Conversation Manager] | Session deleted | Session=%s", session_id)

        return True
