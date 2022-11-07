import time
from typing import Callable, Dict, List, Union

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers
from tqdm.auto import tqdm


class Index:
    def __init__(self) -> None:

        self.es = Elasticsearch(["localhost"], timeout=3600, retry_after_timeout=True)

    def index_exists(self, index: str) -> bool:
        return self.es.indices.exists(index)

    def create_index(self, index: str):
        if self.index_exists(index):
            raise RuntimeError("Index already exists.")
        self.es.indices.create(
            index,
            body={
                "mappings": {
                    "properties": {"text": {"type": "text", "term_vector": "yes"}}
                }
            },
        )

    def index_corpus(
        self,
        index: str,
        corpus: Union[Callable, Dict[str, str]],
        corpus_size: Union[float, None] = None,
    ):
        self.create_index(index)
        chunk_size = 1024
        with tqdm(desc="indexing", ncols=100, total=corpus_size) as pbar:
            if isinstance(corpus, Callable):
                chunk = []
                for doc in corpus(verbose=False):
                    chunk.append(
                        {
                            "_index": index,
                            "_id": doc["id"],
                            "_source": {"text": doc["text"]},
                        }
                    )
                    if len(chunk) == chunk_size:
                        es_helpers.bulk(self.es, chunk)
                        pbar.update(len(chunk))
                        chunk = []
                es_helpers.bulk(self.es, chunk)
                pbar.update(len(chunk))
                time.sleep(0.1)
            else:
                for start_idx in range(0, len(corpus), chunk_size):
                    end_idx = start_idx + chunk_size
                    chunk = [
                        {
                            "_index": index,
                            "_id": doc_id,
                            "_source": {"text": corpus[doc_id]},
                        }
                        for doc_id in list(corpus.keys())[start_idx:end_idx]
                    ]
                    es_helpers.bulk(self.es, chunk)
                    pbar.update(chunk_size)
                    time.sleep(0.1)

        for _ in tqdm(range(120), ncols=100, desc="Granting ES some beauty sleep."):
            time.sleep(1)

    def get_doc_by_id(self, index: str, doc_id: str):
        return self.es.get(index=index, id=doc_id)["_source"]["text"]

    def query(self, index, body, time_it: bool = False):
        if time_it:
            self.es.indices.clear_cache(
                index=index,
            )
            time.sleep(1)
        result = self.es.search(index=index, body=body)
        return result

    def filter_labeled_docs(
        self, remove_topics_from_annotatoins: List[Dict], result: List, size: int
    ):
        labeled_doc_ids = list(
            map(
                lambda a: a["doc_id"],
                remove_topics_from_annotatoins,
            )
        )
        result = [r for r in result if r["_id"] not in labeled_doc_ids][:size]
        return result

    def bm25_query(
        self, index: str, topics: Dict[str, str], size: int, time_it: bool = False
    ):
        bm25_results = {}
        es_query = {
            "query": {
                "multi_match": {
                    "query": None,
                    "fields": ["text"],
                }
            },
            "size": size,
        }

        bm25_results, doc_id2text, topic_id2took = {}, {}, {}
        for topic_id, query in tqdm(
            topics.items(), total=len(topics), desc="caching bm25", ncols=100
        ):
            es_query["query"]["multi_match"]["query"] = query
            result = self.query(index, body=es_query, time_it=time_it)
            hits = result["hits"]["hits"]
            topic_id2took[topic_id] = result["took"]

            doc_id2score = {}
            for r in hits:
                doc_id2score[r["_id"]] = r["_score"]
                doc_id2text[r["_id"]] = r["_source"]["text"]
            bm25_results[topic_id] = doc_id2score

        times = list(topic_id2took.values())
        topic_id2took["avg"] = np.mean(times)
        topic_id2took["std"] = np.std(times)

        return bm25_results, doc_id2text, topic_id2took

    def more_like_this_expansion(
        self,
        index: str,
        topics: Dict[str, str],
        annotations: Dict[str, List[Dict]],
        rm_annotations: Dict[str, List[Dict]],
        size: int,
        unlike: bool = False,
        max_query_terms: int = 25,
        time_it: bool = False,
    ):
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-mlt-query.html#query-dsl-mlt-query
        es_query = {
            "query": {
                "more_like_this": {
                    "fields": ["text"],
                    "max_query_terms": max_query_terms,
                }
            },
            "size": size + len(rm_annotations[list(rm_annotations.keys())[0]]),
        }
        mlt_results, doc_id2text, topic_id2took = {}, {}, {}
        for topic_id, annotation in tqdm(
            annotations.items(),
            total=len(annotations),
            desc=f"caching MLT expansion (terms={max_query_terms:04d}, negatives={unlike})",
            ncols=100,
        ):

            query = topics[topic_id]
            like_query, unlike_query = (
                [],
                [],
            )
            for a in annotation:
                if a["label"] >= 1:
                    like_query.append({"_index": index, "_id": a["doc_id"]})
                else:
                    unlike_query.append({"_index": index, "_id": a["doc_id"]})

            es_query["query"]["more_like_this"]["like"] = like_query + [query]
            if unlike:
                es_query["query"]["more_like_this"]["unlike"] = unlike_query

            result = self.query(index, body=es_query, time_it=time_it)
            hits = result["hits"]["hits"]
            topic_id2took[topic_id] = result["took"]
            hits = self.filter_labeled_docs(rm_annotations[topic_id], hits, size)
            doc_id2score = {}
            for r in hits:
                doc_id2score[r["_id"]] = r["_score"]
                doc_id2text[r["_id"]] = r["_source"]["text"]
            mlt_results[topic_id] = doc_id2score

        times = list(topic_id2took.values())
        topic_id2took["avg"] = np.mean(times)
        topic_id2took["std"] = np.std(times)

        return mlt_results, doc_id2text, topic_id2took

    def bm25_query_expansion(
        self,
        index: str,
        topics: Dict[str, str],
        annotations: Dict[str, List[Dict]],
        rm_annotations: Dict[str, List[Dict]],
        size: int,
        time_it: bool = False,
    ):
        es_query = {
            "query": {
                "bool": {
                    "must": None,
                    "should": None,
                }
            },
            "size": size + len(rm_annotations[list(rm_annotations.keys())[0]]),
        }

        bm25_results, doc_id2text, topic_id2took = {}, {}, {}
        for topic_id, annotation in tqdm(
            annotations.items(),
            total=len(annotations),
            desc="caching bm25 expansion",
            ncols=100,
        ):

            query = topics[topic_id]

            relevant_docs = []
            for a in annotation:
                if a["label"] >= 1:
                    relevant_docs.append(a["document"])
            # add query
            es_query["query"]["bool"]["must"] = [
                {
                    "match": {
                        "text": {
                            "query": query,
                        },
                    }
                },
            ]
            # add relevant docs
            es_query["query"]["bool"]["should"] = [
                {
                    "match": {
                        "text": {
                            "query": doc,
                            "boost": 1,
                        },
                    }
                }
                for doc in relevant_docs
            ]
            result = self.query(index, body=es_query, time_it=time_it)
            hits = result["hits"]["hits"]
            topic_id2took[topic_id] = result["took"]
            hits = self.filter_labeled_docs(rm_annotations[topic_id], hits, size)
            doc_id2score = {}
            for r in hits:
                doc_id2score[r["_id"]] = r["_score"]
                doc_id2text[r["_id"]] = r["_source"]["text"]
            bm25_results[topic_id] = doc_id2score

        times = list(topic_id2took.values())
        topic_id2took["avg"] = np.mean(times)
        topic_id2took["std"] = np.std(times)

        return bm25_results, doc_id2text, topic_id2took

    def more_like_this_bool_expansion(
        self,
        index: str,
        topics: Dict[str, str],
        annotations: Dict[str, List[Dict]],
        rm_annotations: Dict[str, List[Dict]],
        size: int,
        max_query_terms: int = 25,
        time_it: bool = False,
    ):
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-mlt-query.html#query-dsl-mlt-query
        def get_mlt_query(max_query_terms=25):
            return {
                "more_like_this": {
                    "fields": ["text"],
                    "max_query_terms": max_query_terms,
                }
            }

        def get_query(size):
            return {
                "query": {
                    "bool": {
                        "must": None,
                        "should": [],
                    }
                },
                "size": size + len(rm_annotations[list(rm_annotations.keys())[0]]),
            }

        mlt_results, doc_id2text, topic_id2took = {}, {}, {}
        for topic_id, annotation in tqdm(
            annotations.items(),
            total=len(annotations),
            desc=f"caching MLT expansion (terms={max_query_terms:04d})",
            ncols=100,
        ):

            query = topics[topic_id]
            es_query = get_query(size)
            es_query["query"]["bool"]["must"] = {
                "match": {
                    "text": {
                        "query": query,
                    },
                }
            }
            for a in annotation:
                if a["label"] >= 1:
                    es_query["query"]["bool"]["should"].append(
                        get_mlt_query(max_query_terms)
                    )
                    es_query["query"]["bool"]["should"][-1]["more_like_this"][
                        "like"
                    ] = {"_index": index, "_id": a["doc_id"]}

            result = self.query(index, body=es_query, time_it=time_it)
            hits = result["hits"]["hits"]
            topic_id2took[topic_id] = result["took"]
            hits = self.filter_labeled_docs(rm_annotations[topic_id], hits, size)
            doc_id2score = {}
            for r in hits:
                doc_id2score[r["_id"]] = r["_score"]
                doc_id2text[r["_id"]] = r["_source"]["text"]
            mlt_results[topic_id] = doc_id2score

        times = list(topic_id2took.values())
        topic_id2took["avg"] = np.mean(times)
        topic_id2took["std"] = np.std(times)

        return mlt_results, doc_id2text, topic_id2took
