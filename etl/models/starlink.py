from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Starlink:
    # Basic metadata
    id: str
    version: Optional[str]
    launch_id: Optional[str]

    # Position & movement
    longitude: Optional[float]
    latitude: Optional[float]
    height_km: Optional[float]
    velocity_kms: Optional[float]

    # Object identity
    object_name: Optional[str]
    object_id: Optional[str]
    rcs_size: Optional[str]

    # Orbital parameters
    epoch: Optional[str]
    mean_motion: Optional[float]
    eccentricity: Optional[float]
    inclination: Optional[float]
    ra_of_asc_node: Optional[float]
    arg_of_pericenter: Optional[float]
    mean_anomaly: Optional[float]
    ephemeris_type: Optional[int]

    # Additional orbital elements
    mean_motion_dot: Optional[float]
    mean_motion_ddot: Optional[float]
    bstar: Optional[float]
    semimajor_axis: Optional[float]
    period: Optional[float]
    apoapsis: Optional[float]
    periapsis: Optional[float]

    # Identification
    norad_cat_id: Optional[int]
    rev_at_epoch: Optional[int]

    # Launch & decay metadata
    launch_date: Optional[datetime]
    site: Optional[str]
    decay_date: Optional[datetime]
    decayed: Optional[int]
    file_number: Optional[int]
    gp_id: Optional[int]

    # Two-line element set (TLE)
    tle_line0: Optional[str]
    tle_line1: Optional[str]
    tle_line2: Optional[str]
