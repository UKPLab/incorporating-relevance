include .env

.PHONY: install install-dev format download index first-stage zero-shot knn query-fine-tune meta-query-fine-tune rank-fusion help
.DEFAULT_GOAL := help
DOCKER_AVAILABLE := $(shell docker -v 2>/dev/null)
ES_DIR := ./elasticsearch-7.11.2

## 		Install dependencies. Note: Activate virtualenv before running this command.
install:
	pip install --upgrade pip
	pip install -r requirements.txt

## 		Install development dependencies. Note: Activate virtualenv before running this command.
install-dev: 
	pip install --upgrade pip
	pip install -r requirements.dev.txt
	pre-commit install

## 		Format code base.
format: 
	isort --profile black .
	black .
	
## 		Download datasets according to the configuration in .env.
download: 
	@echo env-file=$(env-file)
	python inc_rel/download.py --env-file $(env-file)

es-up: 
ifdef DOCKER_AVAILABLE
	@echo Runnig Elasticsearch in docker
	docker run -d \
		-v $(INC_REL_ES_DATA_DIR):/usr/share/elasticsearch/data \
		-p 9200:9200 \
		-e "discovery.type=single-node" \
		-e "indices.query.bool.max_clause_count=16384" \
		--name inc-rel-es \
		elasticsearch:7.11.2
else
	@echo Running Elasticsearch on host
	$(ES_DIR)/bin/elasticsearch -Ediscovery.type=single-node -Eindices.query.bool.max_clause_count=16438 -d -p $(PWD)/es.pid
endif

	bash -c "until curl -s -o /dev/null http://localhost:9200; do echo 'Waiting for Elasticsearch'; sleep 3; done"

es-down:
ifdef DOCKER_AVAILABLE
	docker rm -s inc-rel-es
else
	kill `cat es.pid`
endif

genreate-first-stage:
	python inc_rel/first_stage.py --dataset $(dataset) || $(MAKE) es-down
	
first-stage: dataset:=$(dataset)
## 		Get first stage retrieval results using BM25.
first-stage: es-up generate-first-stage es-down

generate-few-shot:
	python inc_rel/generate_few_shot.py --dataset $(dataset) || $(MAKE) es-down
index: dataset:=$(dataset)
## 		Create an elasticsearch index; perform first and second stage retrieval and generate the few-shot dataset.
index: es-up generate-few-shot es-down


## 		Evaluate zero-shot re-ranking.
zero-shot: 
	python inc_rel/zero_shot.py $(args)

knn-index: 
	python inc_rel/knn_index.py $(args)
knn-similarities: 
	python inc_rel/knn_similarities.py $(args)
knn-eval: 
	python inc_rel/knn_eval.py $(args)
knn: args:=$(args)
## 			Evaluate knn re-ranking.
knn: knn-index knn-similarities knn-eval
	@echo args=$(args)

## 		Fine-tune the re-ranker per query on the few-shot examples.
query-ft:
	python inc_rel/query_fine_tune.py $(args)

## 	Pre-train the re-ranker using supervised or meta learning, then Fine-tune per query on the few-shot examples.
pre-train-query-ft:
	python inc_rel/pre_train.py $(args)

rank-fusion:
	@echo dataset=$(dataset)

# COLORS
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

## 			Show this help.	
help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
