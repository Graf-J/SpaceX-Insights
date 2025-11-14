from etl import BaseETL
from etl.models.launchpad import Launchpad
from pathlib import Path
import sqlite3

create_launchpad_table_sql = """
CREATE TABLE IF NOT EXISTS launchpad (
    id TEXT PRIMARY KEY,
    name TEXT,
    full_name TEXT,
    details TEXT,
    image_url TEXT,
    active BOOLEAN,
    region TEXT,
    locality TEXT,
    latitude REAL,
    longitude REAL,
    launch_attempts INTEGER,
    launch_successes INTEGER
);
"""

insert_launchpad_sql = """
INSERT INTO launchpad (
    id, name, full_name, details, image_url, active,
    region, locality, latitude, longitude,
    launch_attempts, launch_successes
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class LaunchpadETL(BaseETL):
    name = "Launchpad"

    def __init__(self, db_path: Path = Path("data", "starlink.db")) -> None:
        super().__init__(db_path)

    def transform(self, raw_data: list[dict]) -> list[Launchpad]:
        launchpads = []
        for data in raw_data:
            image_url = None
            images = data.get("images")
            if images:
                large_images = images.get("large")
                if large_images and len(large_images) > 0:
                    image_url = large_images[0]

            status = data.get("status")
            if status is None:
                active = None
            else:
                active = status.lower() == "active"

            launchpad = Launchpad(
                id=data.get("id"),
                name=data.get("name"),
                full_name=data.get("full_name"),
                details=data.get("details"),
                image_url=image_url,
                active=active,
                region=data.get("region"),
                locality=data.get("locality"),
                latitude=data.get("latitude"),
                longitude=data.get("longitude"),
                launch_attempts=data.get("launch_attempts"),
                launch_successes=data.get("launch_successes"),
            )
            launchpads.append(launchpad)

        return launchpads

    def load(self, launchpads: list[Launchpad]) -> None:
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()

            # Database performance tuning
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")

            # Ensure table exists
            cursor.execute(create_launchpad_table_sql)

            # Convert dataclasses to list of tuples
            data = [
                (
                    lp.id,
                    lp.name,
                    lp.full_name,
                    lp.details,
                    lp.image_url,
                    lp.active,
                    lp.region,
                    lp.locality,
                    lp.latitude,
                    lp.longitude,
                    lp.launch_attempts,
                    lp.launch_successes,
                )
                for lp in launchpads
            ]

            # Begin transaction
            cursor.execute("BEGIN TRANSACTION;")
            try:
                # Optional: clear table first
                cursor.execute("DELETE FROM launchpad;")

                # Bulk insert
                cursor.executemany(insert_launchpad_sql, data)

                # Commit
                db.commit()
            except Exception as e:
                db.rollback()
                raise e
