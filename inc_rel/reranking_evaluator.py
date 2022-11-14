from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pytrec_eval import RelevanceEvaluator
from sentence_transformers import util
from sentence_transformers import util as st_util
from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.SentenceTransformer import SentenceTransformer
from tqdm.auto import tqdm


# https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(model):
    """Temporarily switch to evaluation mode."""
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


class RerankingEvaluator:
    def __init__(self, qrels: Dict[str, Dict[str, int]]):

        self.qrels = qrels
        at_k = [1, 5, 10, 20, 50, 100, 1000]
        map = "map_cut." + ",".join([str(k) for k in at_k])
        ndcg = "ndcg_cut." + ",".join([str(k) for k in at_k])
        recall = "recall." + ",".join([str(k) for k in at_k])
        precision = "P." + ",".join([str(k) for k in at_k])
        rprecision = "Rprec"
        self.measures = {map, ndcg, recall, precision, rprecision}

    def __call__(
        self,
        model: Union[CrossEncoder, SentenceTransformer],
        queries: Dict[str, str],
        docs: Dict[str, str],
        inital_ranking: Dict[str, Dict[str, float]],
        model_ctx: SentenceTransformer = None,
        batch_size: int = 32,
        tokenizer=None,
        show_progress_bar=False,
        scoring_fn="cos",
    ):

        if tokenizer is None:
            tokenizer = model.tokenizer

        if scoring_fn == "cos":
            scoring_fn = st_util.cos_sim
        elif scoring_fn == "dot":
            scoring_fn = st_util.dot_score

        run = defaultdict(lambda: defaultdict(dict))
        with torch.no_grad(), evaluating(
            model if isinstance(model, SentenceTransformer) else model.model
        ):
            for topic_id, query in tqdm(
                queries.items(),
                desc="eval queries",
                ncols=100,
                disable=not show_progress_bar,
            ):

                doc_ids = inital_ranking[topic_id].keys()
                query_docs = [docs[doc_id] for doc_id in doc_ids]
                if isinstance(model, CrossEncoder):
                    sentences = [[query, doc] for doc in query_docs]
                    scores = model.predict(
                        sentences,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                    )
                elif isinstance(model, SentenceTransformer):
                    query_embedding = model.encode(
                        query,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                    )
                    if model_ctx:
                        doc_model = model_ctx
                    else:
                        doc_model = model
                    doc_embeddings = doc_model.encode(
                        query_docs,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                    )
                    scores = scoring_fn(query_embedding, doc_embeddings)[0]
                # sort results
                run[topic_id] = {
                    doc_id: float(score)
                    for doc_id, score in sorted(
                        zip(doc_ids, scores),
                        key=lambda doc_score: doc_score[1],
                        reverse=True,
                    )
                }

        # RelevanceEvaluator seems to swallow some metrics when re-used
        # re-initializing fixes this.
        results = self.eval(run)

        return results, run

    def eval(self, run):
        return RelevanceEvaluator(self.qrels, self.measures).evaluate(run)
