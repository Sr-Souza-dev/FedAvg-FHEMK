from datetime import datetime

def get_uid_per_minute(year: int=2025) -> int:
    """Get the number of minutes since the given year."""
    now = datetime.now()
    start = datetime(year, 1, 1)
    delta = now - start
    return str(int(delta.total_seconds() / 60))