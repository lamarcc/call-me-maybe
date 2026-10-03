NAME        = codexion
UV          = uv
PYTHON      = python3
RM          = rm -rf
MYPYFLAGS   = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	$(UV) sync

run:
	$(UV) run src

debug:
	$(UV) run $(PYTHON) -m pdb -m src

lint:
	$(UV) run flake8 . --exclude=.venv,llm_sdk
	$(UV) run $(PYTHON) -m mypy . $(MYPYFLAGS) --exclude llm_sdk --exclude .venv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	$(RM) .mypy_cache

.PHONY: install run debug lint clean
