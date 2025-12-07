from datetime import datetime

from prefect.runtime import task_run

from etl import BaseETL


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


def generate_task_run_name(step_name: str):
    def _generate_name():
        etl: BaseETL = task_run.get_parameters()["etl"]
        return f"{etl.name} - {step_name}"

    return _generate_name
