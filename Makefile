include .env

all : install install-dev format download index
.PHONY: all

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.dev.txt
	pre-commit install

format:
	black .
	
download:
	python download.py

index:
	docker run -d \
		-v $(INC_REL_ES_DATA_DIR):/usr/share/elasticsearch/data \
		-p 9200:9200 \
		-e "discovery.type=single-node" \
		-e "indices.query.bool.max_clause_count=16384" \
		--name inc-rel-es \
		elasticsearch:7.11.2
	bash -c "until curl -s -o /dev/null http://localhost:9200; do echo 'Waiting for Elasticsearch.'; sleep 3; done"

	docker stop inc-rel-es && docker rm inc-rel-es
