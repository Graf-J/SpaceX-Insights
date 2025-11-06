import asyncio

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from etl import BaseETL, LaunchETL, StarlinkETL


@task
async def extract_task(etl: BaseETL, url: str) -> list[dict]:
    return await etl.extract(url)


@task
def transform_task(etl: BaseETL, raw_data: list[dict]) -> list:
    return etl.transform(raw_data)


@task
async def load_task(etl: BaseETL, transformed_data: list) -> None:
    await etl.load(transformed_data)


async def run_etl(etl, url: str):
    raw_data = await extract_task(etl, url)
    transformed_data = transform_task(etl, raw_data)
    await load_task(etl, transformed_data)


@flow(task_runner=ConcurrentTaskRunner())
async def main_pipeline():
    etls = [
        (LaunchETL(), "https://api.spacexdata.com/v4/launches"),
        (StarlinkETL(), "https://api.spacexdata.com/v4/starlink"),
        # Rocket
        # LaunchPad
        # LandingPad
    ]

    await asyncio.gather(*(run_etl(etl, url) for etl, url in etls))


# TODO: Make etl_flow possible to run individually by passing a parameter in the UI

if __name__ == "__main__":
    asyncio.run(main_pipeline())
