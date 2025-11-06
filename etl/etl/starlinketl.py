from pathlib import Path

import aiosqlite

from etl import BaseETL
from etl.models.starlink import Starlink
from etl.utils.utils import parse_date

create_table_sql = """
CREATE TABLE IF NOT EXISTS starlink (
    id TEXT PRIMARY KEY,
    version TEXT,
    launch_id TEXT,
    longitude REAL,
    latitude REAL,
    height_km REAL,
    velocity_kms REAL,
    object_name TEXT,
    object_id TEXT,
    rcs_size TEXT,
    epoch TEXT,
    mean_motion REAL,
    eccentricity REAL,
    inclination REAL,
    ra_of_asc_node REAL,
    arg_of_pericenter REAL,
    mean_anomaly REAL,
    ephemeris_type INTEGER,
    mean_motion_dot REAL,
    mean_motion_ddot REAL,
    bstar REAL,
    semimajor_axis REAL,
    period REAL,
    apoapsis REAL,
    periapsis REAL,
    norad_cat_id INTEGER,
    rev_at_epoch INTEGER,
    launch_date TEXT,
    site TEXT,
    decay_date TEXT,
    decayed INTEGER,
    file_number INTEGER,
    gp_id INTEGER,
    tle_line0 TEXT,
    tle_line1 TEXT,
    tle_line2 TEXT
);
"""

insert_sql = """
INSERT INTO starlink (
    id, version, launch_id,
    longitude, latitude, height_km, velocity_kms,
    object_name, object_id, rcs_size,
    epoch, mean_motion, eccentricity, inclination,
    ra_of_asc_node, arg_of_pericenter, mean_anomaly, ephemeris_type,
    mean_motion_dot, mean_motion_ddot, bstar, semimajor_axis, period,
    apoapsis, periapsis, norad_cat_id, rev_at_epoch,
    launch_date, site, decay_date, decayed, file_number, gp_id,
    tle_line0, tle_line1, tle_line2
) VALUES (
    ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?
);
"""


class StarlinkETL(BaseETL):
    def __init__(self, db_path: Path = Path("data", "starlink.db")) -> None:
        super().__init__(db_path)

    def transform(self, raw_data: list[dict]) -> list[Starlink]:
        starlinks = []
        for raw in raw_data:
            st: dict = raw.get("spaceTrack", {})
            flat = {
                "id": raw.get("id"),
                "version": raw.get("version"),
                "launch_id": raw.get("launch"),
                "longitude": raw.get("longitude"),
                "latitude": raw.get("latitude"),
                "height_km": raw.get("height_km"),
                "velocity_kms": raw.get("velocity_kms"),
                "object_name": st.get("OBJECT_NAME"),
                "object_id": st.get("OBJECT_ID"),
                "rcs_size": st.get("RCS_SIZE"),
                "epoch": st.get("EPOCH"),
                "mean_motion": st.get("MEAN_MOTION"),
                "eccentricity": st.get("ECCENTRICITY"),
                "inclination": st.get("INCLINATION"),
                "ra_of_asc_node": st.get("RA_OF_ASC_NODE"),
                "arg_of_pericenter": st.get("ARG_OF_PERICENTER"),
                "mean_anomaly": st.get("MEAN_ANOMALY"),
                "ephemeris_type": st.get("EPHEMERIS_TYPE"),
                "mean_motion_dot": st.get("MEAN_MOTION_DOT"),
                "mean_motion_ddot": st.get("MEAN_MOTION_DDOT"),
                "bstar": st.get("BSTAR"),
                "semimajor_axis": st.get("SEMIMAJOR_AXIS"),
                "period": st.get("PERIOD"),
                "apoapsis": st.get("APOAPSIS"),
                "periapsis": st.get("PERIAPSIS"),
                "norad_cat_id": st.get("NORAD_CAT_ID"),
                "rev_at_epoch": st.get("REV_AT_EPOCH"),
                "launch_date": parse_date(st.get("LAUNCH_DATE")),
                "site": st.get("SITE"),
                "decay_date": parse_date(st.get("DECAY_DATE")),
                "decayed": st.get("DECAYED"),
                "file_number": st.get("FILE"),
                "gp_id": st.get("GP_ID"),
                "tle_line0": st.get("TLE_LINE0"),
                "tle_line1": st.get("TLE_LINE1"),
                "tle_line2": st.get("TLE_LINE2"),
            }

            starlinks.append(Starlink(**flat))

        return starlinks

    async def load(self, starlinks: list[Starlink]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = OFF;")
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")

            # Create table if it doesn't exist
            await db.execute(create_table_sql)

            # Convert dataclass list to list of tuples
            data = [
                (
                    s.id,
                    s.version,
                    s.launch_id,
                    s.longitude,
                    s.latitude,
                    s.height_km,
                    s.velocity_kms,
                    s.object_name,
                    s.object_id,
                    s.rcs_size,
                    s.epoch,
                    s.mean_motion,
                    s.eccentricity,
                    s.inclination,
                    s.ra_of_asc_node,
                    s.arg_of_pericenter,
                    s.mean_anomaly,
                    s.ephemeris_type,
                    s.mean_motion_dot,
                    s.mean_motion_ddot,
                    s.bstar,
                    s.semimajor_axis,
                    s.period,
                    s.apoapsis,
                    s.periapsis,
                    s.norad_cat_id,
                    s.rev_at_epoch,
                    s.launch_date.isoformat() if s.launch_date else None,
                    s.site,
                    s.decay_date.isoformat() if s.decay_date else None,
                    s.decayed,
                    s.file_number,
                    s.gp_id,
                    s.tle_line0,
                    s.tle_line1,
                    s.tle_line2,
                )
                for s in starlinks
            ]

            # Insert Data
            async with db.execute("BEGIN TRANSACTION;"):
                await db.execute("DELETE FROM starlink;")
                await db.executemany(insert_sql, data)
                await db.commit()
