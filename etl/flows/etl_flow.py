from typing import Literal
from prefect import flow, task
from prefect.futures import wait

from etl import BaseETL, LaunchETL, StarlinkETL, RocketETL, LaunchpadETL
from etl.utils.utils import generate_task_run_name


@task(name="Extract", task_run_name=generate_task_run_name("Extract"))
def extract_task(etl: BaseETL, url: str) -> list[dict]:
    return etl.extract(url)


@task(name="Transform", task_run_name=generate_task_run_name("Transform"))
def transform_task(etl: BaseETL, raw_data: list[dict]) -> list:
    return etl.transform(raw_data)


@task(name="Load", task_run_name=generate_task_run_name("Load"))
def load_task(etl: BaseETL, transformed_data: list) -> None:
    etl.load(transformed_data)


@flow
def main_pipeline(etl_param: None | Literal["Launch", "Rocket", "Starlink"] = None):
    etls = {
        "Rocket": (RocketETL(), "https://api.spacexdata.com/v4/rockets"),
        "Launchpad": (LaunchpadETL(), "https://api.spacexdata.com/v4/launchpads"),
        "Launch": (LaunchETL(), "https://api.spacexdata.com/v4/launches"),
        "Starlink": (StarlinkETL(), "https://api.spacexdata.com/v4/starlink"),
    }

    # Filter if ETL specified
    if etl_param:
        etls = {etl_param: etls[etl_param]}

    # Extract in Parallel
    futures_extract = extract_task.map(
        [etl for etl, _ in etls.values()],
        [url for _, url in etls.values()],
    )
    wait(futures_extract)
    results_raw_data = [f.result() for f in futures_extract]

    # Transform in Parallel
    futures_transform = transform_task.map(
        [etl for etl, _ in etls.values()], [raw_data for raw_data in results_raw_data]
    )
    wait(futures_transform)
    results_transformed_data = [f.result() for f in futures_transform]

    # Load Data
    if etl_param:
        load_task(etls[etl_param][0], results_transformed_data[0])
    else:
        rocket_future = load_task.submit(
            etls["Rocket"][0],
            results_transformed_data[list(etls.keys()).index("Rocket")],
        )
        launchpad_future = load_task.submit(
            etls["Launchpad"][0],
            results_transformed_data[list(etls.keys()).index("Launchpad")],
        )
        launch_future = load_task.submit(
            etls["Launch"][0],
            results_transformed_data[list(etls.keys()).index("Launch")],
            wait_for=[rocket_future, launchpad_future],
        )
        starlink_future = load_task.submit(
            etls["Starlink"][0],
            results_transformed_data[list(etls.keys()).index("Starlink")],
            wait_for=[launch_future],
        )

        wait([rocket_future, launchpad_future, launch_future, starlink_future])



if __name__ == "__main__":
    main_pipeline()  
