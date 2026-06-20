PYTHON ?= python3

ifneq (,$(wildcard .env))
include .env
export
endif

HOST ?= $(PE_MAS_HOST)
PORT ?= $(PE_MAS_PORT)
APP_MODE ?= studio

.PHONY: bootstrap run run-api clean

bootstrap:
	$(PYTHON) -m pip install -r requirements.txt

run:
	@test -n "$(HOST)" || (echo "Set PE_MAS_HOST locally before running." && exit 1)
	@test -n "$(PORT)" || (echo "Set PE_MAS_PORT locally before running." && exit 1)
	@test "$(HOST)" != "<bind-host>" || (echo "Set PE_MAS_HOST to a real local bind host." && exit 1)
	@test "$(PORT)" != "<port>" || (echo "Set PE_MAS_PORT to a real local port." && exit 1)
	PE_MAS_APP_MODE=$(APP_MODE) PE_MAS_HOST=$(HOST) PE_MAS_PORT=$(PORT) $(PYTHON) server.py

run-api:
	@test -n "$(HOST)" || (echo "Set PE_MAS_HOST locally before running." && exit 1)
	@test -n "$(PORT)" || (echo "Set PE_MAS_PORT locally before running." && exit 1)
	@test "$(HOST)" != "<bind-host>" || (echo "Set PE_MAS_HOST to a real local bind host." && exit 1)
	@test "$(PORT)" != "<port>" || (echo "Set PE_MAS_PORT to a real local port." && exit 1)
	PE_MAS_APP_MODE=api PE_MAS_HOST=$(HOST) PE_MAS_PORT=$(PORT) $(PYTHON) server.py

clean:
	find . -type d -name '__pycache__' -prune -print0 | xargs -0 rm -rf
	find . -name '*.pyc' -delete
	rm -rf .pe_mas_runtime .files .vscode report.log
