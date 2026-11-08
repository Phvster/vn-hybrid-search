"""Cost guard for the public demo: per-session and per-day caps on LLM calls."""

import threading
from datetime import UTC, date, datetime

SESSION_LIMIT = 10
DAILY_LIMIT = 300


class UsageLimiter:
    """Thread-safe daily counter. Session counts live in the caller's session state."""

    def __init__(self, daily_limit: int = DAILY_LIMIT, session_limit: int = SESSION_LIMIT):
        self.daily_limit = daily_limit
        self.session_limit = session_limit
        self._lock = threading.Lock()
        self._day: date | None = None
        self._count = 0

    def try_acquire(self, session_count: int, today: date | None = None) -> str | None:
        """Reserve one call. Returns None if allowed, otherwise the reason it is refused."""
        if session_count >= self.session_limit:
            return f"Đã đạt giới hạn {self.session_limit} câu trả lời cho mỗi phiên."
        today = today or datetime.now(UTC).date()
        with self._lock:
            if self._day != today:
                self._day, self._count = today, 0
            if self._count >= self.daily_limit:
                return "Demo đã đạt giới hạn sử dụng trong ngày. Vui lòng quay lại sau."
            self._count += 1
        return None
