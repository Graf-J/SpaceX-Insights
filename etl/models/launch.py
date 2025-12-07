from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Launch:
    # Identifiers
    id: str
    flight_number: int
    name: str

    # Timing
    date: datetime
    static_fire: Optional[datetime]
    window: Optional[int]
    net: bool
    tbd: bool
    upcoming: bool

    # Hardware
    rocket_id: str
    launchpad_id: str

    # Mission Outcome
    success: Optional[bool]
    recovery_attempt: bool
    recovered: bool

    # Metadata / additional info
    details: Optional[str]
    patch_url: Optional[str]
    youtube_url: Optional[str]
