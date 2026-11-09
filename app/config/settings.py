import os
from dotenv import load_dotenv

class Settings:
    def __init__(self):
        load_dotenv()

        # LiveKit Config
        self.livekit_url = self._required("LIVEKIT_URL")
        self.livekit_api_key = self._required("LIVEKIT_API_KEY")
        self.livekit_api_secret = self._required("LIVEKIT_API_SECRET")

        # App Config
        self.worker_host = self._required("WORKER_HOST")
        self.worker_port = int(self._required("WORKER_PORT"))

        # STT and TTS Config
        self.tts_ws_url = self._required("TTS_WS_URL")
        self.tts_sample_rate = int(self._required("TTS_SAMPLE_RATE"))
        self.stt_ws_url = self._required("STT_WS_URL")
        
        # LLM Config
        self.base_llm_url = self._required("BASE_LLM_URL")
        
        self.chat_endpoint = self._required("CHAT_ENDPOINT")
        self.chat_llm_model = self._required("CHAT_LLM_MODEL")

        self.llm_max_history_messages = int(self._required("LLM_MAX_HISTORY_MESSAGES"))
        self.http_timeout_seconds = float(self._required("HTTP_TIMEOUT_SECONDS"))

    @staticmethod
    def _required(name: str) -> str:
        value = os.getenv(name)

        if not value:
            raise RuntimeError(
                f"{name} is missing from environment variables"
            )
        return value

settings = Settings()