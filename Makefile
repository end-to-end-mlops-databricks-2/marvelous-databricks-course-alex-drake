create-venv:
	@echo "Create virtual environment using uv"
	uv venv -p 3.11 .venv

activate-venv:
	@echo "Activating virtual environment"
	source .venv/bin/activate

sync-venv:
	@echo "Syncing python environment"
	uv pip install -r pyproject.toml --all-extras --link-mode=copy
	uv lock

lint:
	pre-commit run --all-files

clean:
	rm -rf __pycache__ dist

test:
	pytest

build:
	@echo "Building hotel_reservations package"
	uv build

copy-whl-to-databricks:
	@echo "Copying package to Databricks"
	databricks fs cp ./dist/*.whl dbfs:/Volumes/mlops_dev/aldrake8/packages/hotel_reservations-latest-py3-none-any.whl --overwrite