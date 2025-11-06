import asyncio
from typing import Type

from prefect import flow, task

from etl import BaseETL, StarlinkETL


@task
async def extract_task(etl: BaseETL, url: str) -> list[dict]:
    return await etl.extract(url)


@task
def transform_task(etl: BaseETL, raw_data: list[dict]) -> list:
    return etl.transform(raw_data)


@task
async def load_task(etl: BaseETL, transformed_data: list) -> None:
    await etl.load(transformed_data)


@flow
async def etl_flow(etl_class: Type[BaseETL], url: str) -> None:
    etl: BaseETL = etl_class()
    raw_data = await extract_task(etl, url)
    transformed_data = transform_task(etl, raw_data)
    await load_task(etl, transformed_data)


@flow
async def main_pipeline():
    # Define ETLs to run
    etls = [
        (StarlinkETL, "https://api.spacexdata.com/v4/starlink"),
        # (RocketETL, "https://api.spacexdata.com/v4/rockets"),
        # (LaunchETL, "https://api.spacexdata.com/v4/launches"),
    ]

    # Create coroutine objects for each ETL flow
    coros = [etl_flow(etl_cls, url) for etl_cls, url in etls]

    # Run them concurrently
    await asyncio.gather(*coros)


if __name__ == "__main__":
    asyncio.run(main_pipeline())
