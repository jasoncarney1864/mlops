.PHONY: setup train run-api docker-build docker-run clean

PY=python
UVICORN=uvicorn

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	. .venv/bin/activate && $(PY) src/train.py

run-api:
	. .venv/bin/activate && $(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t housing-api:latest .

docker-run:
	docker run --rm -p 8000:8000 housing-api:latest

clean:
	rm -rf __pycache__ .pytest_cache .venv