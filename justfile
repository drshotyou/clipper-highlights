set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

default:
    @just --list

build:
    docker compose build

build-gpu:
    docker compose -f compose.yaml -f compose.gpu.yaml build

test:
    pytest -q tests

venv:
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e '.[dev]'

cli *args:
    PYTHONPATH=src python -m clipper_highlights {{args}}

docker *args:
    docker compose run --rm clipper-highlights {{args}}

docker-gpu *args:
    docker compose -f compose.yaml -f compose.gpu.yaml run --rm clipper-highlights {{args}}

init-config destination="config.yaml":
    PYTHONPATH=src python -m clipper_highlights init-config {{destination}}

docker-init-config destination="/workspace/config.yaml":
    docker compose run --rm clipper-highlights init-config {{destination}}
