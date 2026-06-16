install:
	python -m pip install -r requirements.txt

generate-data:
	python generate_synthetic_data.py --days 30

run-app:
	streamlit run app.py

train:
	python main.py

test:
	pytest

help:
	@echo "Available tasks: install, generate-data, run-app, train, test"
