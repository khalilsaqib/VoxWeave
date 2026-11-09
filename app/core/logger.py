import threading
import traceback
from datetime import datetime
from pathlib import Path

class LoggingService:
    def __init__(self, log_dir: str = "app_logs"):
        current_file_dir = Path(__file__).resolve().parents[2]
        self.log_dir = current_file_dir / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_log_file_path(self, level: str) -> Path:
        now = datetime.now()
        day_month_year = now.strftime("%d-%b-%Y").lower()
        filename = f"{day_month_year}_{level.lower()}.log"
        return self.log_dir / filename

    def _write_log(self, level: str, message: str):
        file_path = self._get_log_file_path(level)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        log_line = f"[{timestamp}] [{level.upper()}] | {message}\n"

        with self._lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(log_line)

    def info(self, message: str, *args):
        if args:
            message = message % args
        self._write_log("info", message)

    def error(self, message: str, *args):
        if args:
            message = message % args
        self._write_log("error", message)

    def exception(self, message: str, *args):
        if args:
            message = message % args

        details = traceback.format_exc()
        if details != "NoneType: None\n":
            message = f"{message}\n{details.rstrip()}"

        self._write_log("error", message)

logger = LoggingService()