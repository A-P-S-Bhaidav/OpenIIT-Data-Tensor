PYTHON ?= python3
export PYTHONPATH := src

.PHONY: install train test benchmark replay dashboard serve all

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) scripts/train.py

test:
	$(PYTHON) -m unittest discover -s tests -v

benchmark:
	$(PYTHON) scripts/benchmark_latency.py

replay:
	$(PYTHON) scripts/replay_call.py --ptp PTP000162

dashboard:
	$(PYTHON) scripts/build_dashboard.py

serve:
	$(PYTHON) -m uvicorn ptp_credibility.service.api:app --host 0.0.0.0 --port 8000

all: train test benchmark dashboard
