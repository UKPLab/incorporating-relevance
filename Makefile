install:
	pip install --upgrade pip
	pip install -r requirements

install-dev:
	pip install --upgrade pip
	pip install -r requirements.dev.txt
	pre-commit install

format:
	black .
	
download:
	python download.py
