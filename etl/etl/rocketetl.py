from pathlib import Path
import sqlite3

from etl import BaseETL
from etl.models.rocket import Rocket, RocketEngine, RocketPayload, RocketStage
from etl.utils.utils import parse_date

create_rocket_table = """
CREATE TABLE IF NOT EXISTS rocket (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    wikipedia_url TEXT,
    image_url TEXT,
    active BOOLEAN,
    boosters INTEGER,
    cost_per_launch INTEGER,
    success_rate_pct REAL,
    first_flight TEXT,
    height_m REAL,
    diameter_m REAL,
    mass_kg REAL,
    landing_legs INTEGER,
    landing_legs_material TEXT
);
"""

create_stage_table = """
CREATE TABLE IF NOT EXISTS rocket_stage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rocket_id TEXT,
    stage_type TEXT,
    thrust_vacuum_kN REAL,
    thrust_vacuum_lbf REAL,
    fuel_amount_tons REAL,
    burn_time_sec REAL,
    reusable BOOLEAN,
    engines INTEGER,
    FOREIGN KEY (rocket_id) REFERENCES rocket(id)
);
"""

create_engine_table = """
CREATE TABLE IF NOT EXISTS rocket_engine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rocket_id TEXT,
    number INTEGER,
    type TEXT,
    version TEXT,
    layout TEXT,
    engine_loss_max INTEGER,
    propellant_1 TEXT,
    propellant_2 TEXT,
    thrust_to_weight REAL,
    isp_sea_level REAL,
    isp_vacuum REAL,
    thrust_sea_level_kN REAL,
    thrust_sea_level_lbf REAL,
    thrust_vacuum_kN REAL,
    thrust_vacuum_lbf REAL,
    FOREIGN KEY (rocket_id) REFERENCES rocket(id)
);
"""

create_payload_table = """
CREATE TABLE IF NOT EXISTS rocket_payload (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rocket_id TEXT,
    name TEXT,
    kg REAL,
    FOREIGN KEY (rocket_id) REFERENCES rocket(id)
);
"""

insert_rocket_sql = """
INSERT INTO rocket (
    id, name, description, wikipedia_url, image_url, active,
    boosters, cost_per_launch, success_rate_pct, first_flight,
    height_m, diameter_m, mass_kg, landing_legs, landing_legs_material
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

insert_stage_sql = """
INSERT INTO rocket_stage (
    rocket_id, stage_type, thrust_vacuum_kN, thrust_vacuum_lbf,
    fuel_amount_tons, burn_time_sec, reusable, engines
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

insert_engine_sql = """
INSERT INTO rocket_engine (
    rocket_id, number, type, version, layout, engine_loss_max,
    propellant_1, propellant_2, thrust_to_weight,
    isp_sea_level, isp_vacuum,
    thrust_sea_level_kN, thrust_sea_level_lbf,
    thrust_vacuum_kN, thrust_vacuum_lbf
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

insert_payload_sql = """
INSERT INTO rocket_payload (
    rocket_id, name, kg
) VALUES (?, ?, ?)
"""


class RocketETL(BaseETL):
    name = "Rocket"

    def __init__(self, db_path: Path = Path("data", "starlink.db")) -> None:
        super().__init__(db_path)

    def transform(self, raw_data: list[dict]) -> list[Rocket]:
        rockets = []
        for data in raw_data:
            # Parse stages
            stages: list[RocketStage] = []

            # First stage
            first_stage_data = data["first_stage"]
            stages.append(
                RocketStage(
                    rocket_id=data["id"],
                    stage_type="first_stage",
                    thrust_vacuum_kN=first_stage_data["thrust_vacuum"]["kN"],
                    thrust_vacuum_lbf=first_stage_data["thrust_vacuum"]["lbf"],
                    fuel_amount_tons=first_stage_data["fuel_amount_tons"],
                    burn_time_sec=first_stage_data["burn_time_sec"],
                    reusable=first_stage_data["reusable"],
                    engines=first_stage_data["engines"],
                )
            )

            # Second stage
            second_stage_data = data["second_stage"]
            stages.append(
                RocketStage(
                    rocket_id=data["id"],
                    stage_type="second_stage",
                    thrust_vacuum_kN=second_stage_data["thrust"]["kN"],
                    thrust_vacuum_lbf=second_stage_data["thrust"]["lbf"],
                    fuel_amount_tons=second_stage_data["fuel_amount_tons"],
                    burn_time_sec=second_stage_data["burn_time_sec"],
                    reusable=second_stage_data["reusable"],
                    engines=second_stage_data["engines"],
                )
            )

            # Parse engine
            eng = data["engines"]
            engines = [
                RocketEngine(
                    rocket_id=data["id"],
                    number=eng["number"],
                    type=eng["type"],
                    version=eng["version"],
                    layout=eng["layout"],
                    engine_loss_max=eng["engine_loss_max"],
                    propellant_1=eng["propellant_1"],
                    propellant_2=eng["propellant_2"],
                    thrust_to_weight=eng["thrust_to_weight"],
                    isp_sea_level=eng["isp"]["sea_level"],
                    isp_vacuum=eng["isp"]["vacuum"],
                    thrust_sea_level_kN=eng["thrust_sea_level"]["kN"],
                    thrust_sea_level_lbf=eng["thrust_sea_level"]["lbf"],
                    thrust_vacuum_kN=eng["thrust_vacuum"]["kN"],
                    thrust_vacuum_lbf=eng["thrust_vacuum"]["lbf"],
                )
            ]

            # Parse payloads
            payloads = [
                RocketPayload(
                    rocket_id=data["id"],
                    name=p["name"],
                    kg=p["kg"],
                )
                for p in data["payload_weights"]
            ]

            # Parse Rocket
            rocket = Rocket(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                wikipedia_url=data["wikipedia"],
                image_url=data["flickr_images"][0],
                active=data["active"],
                boosters=data["boosters"],
                cost_per_launch=data["cost_per_launch"],
                success_rate_pct=data["success_rate_pct"],
                first_flight=parse_date(data["first_flight"]),
                height_m=data["height"]["meters"],
                diameter_m=data["diameter"]["meters"],
                mass_kg=data["mass"]["kg"],
                landing_legs=data["landing_legs"]["number"],
                landing_legs_material=data["landing_legs"].get("material"),
                stages=stages,
                engines=engines,
                payloads=payloads,
            )

            rockets.append(rocket)

        return rockets

    def load(self, rockets: list[Rocket]) -> None:
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()

            # Database performance tuning
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")

            # Ensure schema exists
            cursor.execute(create_rocket_table)
            cursor.execute(create_stage_table)
            cursor.execute(create_engine_table)
            cursor.execute(create_payload_table)

            # Begin explicit transaction
            cursor.execute("BEGIN TRANSACTION;")

            try:
                # Cleanup old data
                cursor.execute("DELETE FROM rocket_stage;")
                cursor.execute("DELETE FROM rocket_engine;")
                cursor.execute("DELETE FROM rocket_payload;")
                cursor.execute("DELETE FROM rocket;")

                # Prepare bulk data
                rocket_data = [
                    (
                        r.id,
                        r.name,
                        r.description,
                        r.wikipedia_url,
                        r.image_url,
                        r.active,
                        r.boosters,
                        r.cost_per_launch,
                        r.success_rate_pct,
                        r.first_flight.isoformat() if r.first_flight else None,
                        r.height_m,
                        r.diameter_m,
                        r.mass_kg,
                        r.landing_legs,
                        r.landing_legs_material,
                    )
                    for r in rockets
                ]

                stage_data = [
                    (
                        s.rocket_id,
                        s.stage_type,
                        s.thrust_vacuum_kN,
                        s.thrust_vacuum_lbf,
                        s.fuel_amount_tons,
                        s.burn_time_sec,
                        s.reusable,
                        s.engines,
                    )
                    for r in rockets
                    for s in r.stages
                ]

                engine_data = [
                    (
                        e.rocket_id,
                        e.number,
                        e.type,
                        e.version,
                        e.layout,
                        e.engine_loss_max,
                        e.propellant_1,
                        e.propellant_2,
                        e.thrust_to_weight,
                        e.isp_sea_level,
                        e.isp_vacuum,
                        e.thrust_sea_level_kN,
                        e.thrust_sea_level_lbf,
                        e.thrust_vacuum_kN,
                        e.thrust_vacuum_lbf,
                    )
                    for r in rockets
                    for e in r.engines
                ]

                payload_data = [
                    (p.rocket_id, p.name, p.kg) for r in rockets for p in r.payloads
                ]

                # Bulk insert operations
                cursor.executemany(insert_rocket_sql, rocket_data)
                cursor.executemany(insert_stage_sql, stage_data)
                cursor.executemany(insert_engine_sql, engine_data)
                cursor.executemany(insert_payload_sql, payload_data)

                # Commit transaction
                db.commit()

            except Exception as e:
                # Rollback on error for safety
                db.rollback()
                raise e
