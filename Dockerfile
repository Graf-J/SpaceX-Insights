
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml .
COPY uv.lock .


RUN pip install --no-cache-dir \
    prefect \
    requests \
    numpy \
    pandas \
    pyproj


COPY etl ./etl
COPY analysis ./analysis
COPY data ./data


RUN mkdir -p /app/data


CMD ["python", "-m", "etl.flows.etl_flow"]

