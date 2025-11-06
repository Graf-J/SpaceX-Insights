from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RocketStage:
    # AUTOINCREMENT ID in Database
    rocket_id: str  # foreign key
    stage_type: str  # first_stage, second_stage or whatever
    thrust_vacuum_kN: float  # from thrust_vacuum if first stage, else thrust
    thrust_vacuum_lbf: float  # from thrust_vacuum if first stage, else thrust
    fuel_amount_tons: float
    burn_time_sec: float
    reusable: bool
    engines: int


@dataclass
class RocketEngine:
    # AUTOINCREMENT ID
    rocket_id: str  # Foreign Key
    number: int
    type: str
    version: str
    layout: str
    engine_loss_max: int
    propellant_1: str
    propellant_2: str
    thrust_to_weight: float
    isp_sea_level: float
    isp_vacuum: float
    thrust_sea_level_kN: float
    thrust_sea_level_lbf: float
    thrust_vacuum_kN: float
    thrust_vacuum_lbf: float


@dataclass
class RocketPayload:
    # AUTOINCREMENT ID
    rocket_id: str  # Foreign Key
    name: str
    kg: float


@dataclass
class Rocket:
    id: str
    name: str
    description: str
    wikipedia_url: str
    image_url: str
    active: bool
    boosters: int
    cost_per_launch: int
    success_rate_pct: float
    first_flight: Optional[datetime]
    height_m: float
    diameter_m: float
    mass_kg: float
    landing_legs: int
    landing_legs_material: Optional[str]
    stages: list[RocketStage]
    engines: list[RocketEngine]
    payloads: list[RocketPayload]
