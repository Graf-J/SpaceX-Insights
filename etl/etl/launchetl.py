from etl import BaseETL
from etl.models.launch import Launch
from pathlib import Path
from etl.utils.utils import parse_date
import aiosqlite

create_table_sql = """
CREATE TABLE IF NOT EXISTS launch (
    id TEXT PRIMARY KEY,
    flight_number INTEGER,
    name TEXT,
    date TEXT,
    static_fire TEXT,
    window INTEGER,
    net BOOLEAN,
    tbd BOOLEAN,
    upcoming BOOLEAN,
    rocket_id TEXT,
    launchpad_id TEXT,
    success BOOLEAN,
    recovery_attempt BOOLEAN,
    recovered BOOLEAN,
    details TEXT,
    patch_url TEXT,
    youtube_url TEXT
);
"""

insert_sql = """
INSERT INTO launch (
    id, flight_number, name, date, static_fire, window, net, tbd, upcoming,
    rocket_id, launchpad_id, success, recovery_attempt, recovered,
    details, patch_url, youtube_url
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class LaunchETL(BaseETL):
    name = "Launch"

    def __init__(self, db_path: Path = Path("data", "starlink.db")) -> None:
        super().__init__(db_path)

    def transform(self, raw_data: list[dict]) -> list[Launch]:
        launches = []

        for raw in raw_data:
            fairings = raw.get("fairings") or {}
            links = raw.get("links") or {}

            youtube_id = links.get("youtube_id")
            youtube_url = (
                f"https://www.youtube.com/watch?v={youtube_id}" if youtube_id else None
            )
            launches.append(
                Launch(
                    # Identifiers
                    id=raw["id"],
                    flight_number=raw.get("flight_number"),
                    name=raw.get("name"),
                    # Timing
                    date=parse_date(raw.get("date_utc")),
                    static_fire=parse_date(raw.get("static_fire_date_utc")),
                    window=raw.get("window"),
                    net=raw.get("net"),
                    tbd=raw.get("tbd"),
                    upcoming=raw.get("upcoming"),
                    # Hardware
                    rocket_id=raw.get("rocket"),
                    launchpad_id=raw.get("launchpad"),
                    # Mission Outcome
                    success=raw.get("success"),
                    recovery_attempt=fairings.get("recovery_attempt") or False,
                    recovered=fairings.get("recovered") or False,
                    # Metadata
                    details=raw.get("details"),
                    patch_url=(links.get("patch") or {}).get("small"),
                    youtube_url=youtube_url,
                )
            )

        return launches

    async def load(self, launches: list[Launch]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = OFF;")
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")

            # Create table if it doesn't exist
            await db.execute(create_table_sql)

            # Convert dataclass list to list of tuples
            data = [
                (
                    launch.id,
                    launch.flight_number,
                    launch.name,
                    launch.date.isoformat() if launch.date else None,
                    launch.static_fire.isoformat() if launch.static_fire else None,
                    launch.window,
                    launch.net,
                    launch.tbd,
                    launch.upcoming,
                    launch.rocket_id,
                    launch.launchpad_id,
                    launch.success,
                    launch.recovery_attempt,
                    launch.recovered,
                    launch.details,
                    launch.patch_url,
                    launch.youtube_url,
                )
                for launch in launches
            ]

            # Insert Data
            async with db.execute("BEGIN TRANSACTION;"):
                await db.execute("DELETE FROM launch;")  # Optional: clear table first
                await db.executemany(insert_sql, data)
                await db.commit()
