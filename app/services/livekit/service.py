from livekit import api, rtc
from datetime import timedelta
from app.config.settings import Settings
from app.core.logger import logger

class LiveKitService:
    def __init__(
        self,
        settings: Settings,
    ):
        self.settings = settings

    def create_client_token(
        self,
        room_name: str,
        identity: str,
    ) -> str:
        """
        Create a LiveKit token for the browser/client.
        The browser is allowed to join only the specified room.
        """
        token = (
            api.AccessToken(
                self.settings.livekit_api_key,
                self.settings.livekit_api_secret,
            )
            .with_identity(identity)
            .with_name(identity)
            .with_ttl(timedelta(hours=2))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
            .to_jwt()
        )

        return token

    def create_agent_token(
        self,
        room_name: str,
        session_id: str,
    ) -> str:
        """
        Create a LiveKit token for the AI agent.
        The AI agent gets its own identity but is restricted
        to the same room as the browser.
        """
        agent_identity = (f"ai-agent-{session_id}")

        token = (
            api.AccessToken(
                self.settings.livekit_api_key,
                self.settings.livekit_api_secret,
            )
            .with_identity(agent_identity)
            .with_name("AI Voice Agent")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
            .to_jwt()
        )
        return token

    async def connect(
        self,
        room_name: str,
        session_id: str,
    ) -> rtc.Room:
        """
        Connect the Python AI agent to a LiveKit room.
        """
        token = self.create_agent_token(
            room_name,
            session_id,
        )

        room = rtc.Room()

        logger.info("[LiveKit] | Connecting to URL=%s", self.settings.livekit_url)      

        await room.connect(
            self.settings.livekit_url,
            token,
        )

        logger.info("[LiveKit] | Connected Successfully!")
        
        return room