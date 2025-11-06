from datetime import datetime


def parse_date(s: str | None) -> datetime | None:
    if not s or s.strip() == "":
        return None
    try:
        # Try parsing full ISO format with microseconds
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            # Fallback to ISO format without microseconds
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            # Fallback to date-only
            return datetime.strptime(s, "%Y-%m-%d")
