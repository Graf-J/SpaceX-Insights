import asyncio

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from prefect.runtime import task_run

from etl import BaseETL, LaunchETL, StarlinkETL


def generate_task_run_name(step_name: str):
    def _generate_name():
        etl: BaseETL = task_run.get_parameters()["etl"]
        return f"{etl.name} - {step_name}"

    return _generate_name


@task(name="Extract", task_run_name=generate_task_run_name("Extract"))
async def extract_task(etl: BaseETL, url: str) -> list[dict]:
    return await etl.extract(url)


@task(name="Transform", task_run_name=generate_task_run_name("Transform"))
def transform_task(etl: BaseETL, raw_data: list[dict]) -> list:
    return etl.transform(raw_data)


@task(name="Load", task_run_name=generate_task_run_name("Load"))
async def load_task(etl: BaseETL, transformed_data: list) -> None:
    await etl.load(transformed_data)


async def run_etl(etl, url: str):
    raw_data = await extract_task(etl, url)
    transformed_data = transform_task(etl, raw_data)
    await load_task(etl, transformed_data)


@flow(task_runner=ConcurrentTaskRunner())
async def main_pipeline(selected_etls: list[str] | None = None):
    etls = [
        (LaunchETL(), "https://api.spacexdata.com/v4/launches"),
        (StarlinkETL(), "https://api.spacexdata.com/v4/starlink"),
    ]
    valid_names = [etl.name for etl, _ in etls]

    # Filter ETLs if names are provided
    if selected_etls is not None:
        etls = [(etl, url) for etl, url in etls if etl.name in selected_etls]

    # Raise exception if no ETLs match
    if not etls:
        raise ValueError(
            f"No matching ETLs found for {selected_etls}. "
            f"Valid ETL names are: {valid_names}"
        )

    # Run selected ETLs concurrently
    await asyncio.gather(*(run_etl(etl, url) for etl, url in etls))


if __name__ == "__main__":
    # asyncio.run(main_pipeline())
    main_pipeline.serve(name="ETL-Pipeline")
