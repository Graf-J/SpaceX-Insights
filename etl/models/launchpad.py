from dataclasses import dataclass


@dataclass
class Launchpad:
    id: str
    name: str
    full_name: str
    details: str
    image_url: str
    active: bool
    region: str
    locality: str
    latitude: float
    longitude: float
    launch_attempts: int
    launch_successes: int
