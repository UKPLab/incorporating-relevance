import json
import os
import time
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import Levenshtein as lev
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_best_experiment(results, selection_metric: str = "ndcg_cut_20"):
    for i, r in enumerate(results):
        for metric, score in r["metrics"][list(r["metrics"].keys())[0]].items():
            results[i][metric] = score
        r.pop("metrics")
        r.pop("run")
    df = pd.DataFrame(results)
    epoch, learning_rate = (
        df.groupby(["epoch", "learning_rate"])
        .mean(numeric_only=True)[selection_metric]
        .idxmax()
    )
    return epoch, learning_rate


def create_split(
    topic_ids: List[str], seed: int, split_sizes: List[float]
) -> Tuple[List[str], List[str], List[str]]:

    if not sum(split_sizes) == 1:
        raise ValueError(
            f"Expected `split_sizes` to sum up to 1, but got {sum(split_sizes)=}"
        )

    split_sizes = list(map(lambda s: int(s * len(topic_ids)), split_sizes))
    while sum(split_sizes) < len(topic_ids):
        split_sizes[0] += 1

    train_topic_ids, remaining_topic_ids = train_test_split(
        topic_ids, train_size=split_sizes[0], random_state=seed
    )
    valid_topic_ids, test_topic_ids = train_test_split(
        remaining_topic_ids, train_size=split_sizes[1], random_state=seed
    )

    return train_topic_ids, valid_topic_ids, test_topic_ids


def get_similarity(docs: List[str], similarity_fn: Callable) -> np.ndarray:

    t1 = time.time()
    ratios = np.zeros((len(docs), (len(docs))))
    for i, i_doc in enumerate(docs):
        for j, j_doc in enumerate(docs[i:], start=i):
            if i == j:
                continue
            ratios[i, j] = similarity_fn(i_doc.lower(), j_doc.lower())
    t2 = time.time()
    return ratios


def sorted_annotated_retrieved_docs(
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    bm25_results: Dict[str, Dict[str, float]],
    min_relevant_relevance: int,
    remove_duplicates: bool = False,
    enrich_negatives_from_bm25: Path = None,
    num_non_relevant: int = None,
) -> Dict[str, Dict[str, List[str]]]:

    if enrich_negatives_from_bm25:
        with open(enrich_negatives_from_bm25) as fh:
            full_bm25 = json.load(fh)
        bm25_negatives = {}
        for topic_id, doc2score in full_bm25.items():
            bm25_negatives[topic_id] = list(doc2score.keys())[100:]

    candidates = {}

    for topic_id, annotated_doc_relevance in qrels.items():
        # annotated docs that were retrieved with bm25
        doc_ids = set(annotated_doc_relevance.keys()) & set(
            bm25_results[topic_id].keys()
        )
        if remove_duplicates:
            docs = [corpus[doc_id] for doc_id in doc_ids]

            similarity_ratio = get_similarity(docs, lambda x, y: int(x == y))
            duplicates = np.argwhere(similarity_ratio > 0.9)
            docs_to_remove = sorted(set(duplicates[:, 1].tolist()), reverse=True)
            for index in docs_to_remove:
                del docs[index]

        # get all docs that were retrieved by bm25 and annotated with at least
        # `min_relveant_relevance` or 0 for non relevant
        relevant_docs = sorted(
            filter(
                lambda doc_id: qrels[topic_id][doc_id] >= min_relevant_relevance,
                doc_ids,
            ),
            key=lambda doc_id: bm25_results[topic_id][doc_id],
            reverse=True,
        )
        non_relevant_docs = sorted(
            filter(
                lambda doc_id: qrels[topic_id][doc_id] <= 0,
                doc_ids,
            ),
            key=lambda doc_id: bm25_results[topic_id][doc_id],
            reverse=True,
        )
        if len(non_relevant_docs) < num_non_relevant and enrich_negatives_from_bm25:
            num_to_append = num_non_relevant - len(non_relevant_docs)
            print(f"Appended {num_to_append} bm25 negatives to {topic_id=}.")
            non_relevant_docs.extend(bm25_negatives[topic_id[:num_to_append]])

        candidates[topic_id] = {
            "relevant": relevant_docs,
            "non_relevant": non_relevant_docs,
        }

    return candidates


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
