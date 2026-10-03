NAME        = codexion
UV          = uv
PYTHON      = python3 -m
RM          = rm -rf
MYPYFLAGS   = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) src

debug:
	$(UV) run $(PYTHON) pdb -m src

lint:
	$(UV) run flake8 . --exclude=.venv,llm_sdk
	$(UV) run $(PYTHON) -m mypy . $(MYPYFLAGS) --exclude llm_sdk --exclude .venv

clean:
	find . -type d -name "__pycache__" -exec $(RM) {} +
	$(RM) .mypy_cache

fclean: clean
	rm -rf .venv .penv
	rm -rf data/output

.PHONY: install run debug lint clean fclean
