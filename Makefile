SRC = src/
PY = python3

run:
	uv run $(SRC)

lint:
	flake8 $(SRC)
	$(PY) -m mypy $(SRC)
