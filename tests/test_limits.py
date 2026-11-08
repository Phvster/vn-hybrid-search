from datetime import date

from vnsearch.limits import UsageLimiter


def test_session_limit():
    limiter = UsageLimiter(daily_limit=100, session_limit=2)
    assert limiter.try_acquire(session_count=1) is None
    assert limiter.try_acquire(session_count=2) is not None


def test_daily_limit_resets_next_day():
    limiter = UsageLimiter(daily_limit=2, session_limit=100)
    day1, day2 = date(2026, 11, 8), date(2026, 11, 9)
    assert limiter.try_acquire(0, today=day1) is None
    assert limiter.try_acquire(0, today=day1) is None
    assert limiter.try_acquire(0, today=day1) is not None
    assert limiter.try_acquire(0, today=day2) is None


def test_refused_session_call_does_not_consume_daily_quota():
    limiter = UsageLimiter(daily_limit=1, session_limit=1)
    assert limiter.try_acquire(session_count=5) is not None
    assert limiter.try_acquire(session_count=0) is None
