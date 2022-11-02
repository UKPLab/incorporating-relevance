from collections import defaultdict
from typing import Dict, List

import numpy as np
from pytrec_eval import RelevanceEvaluator

at_k = [1, 5, 10, 20, 50, 100, 1000]
map = "map_cut." + ",".join([str(k) for k in at_k])
ndcg = "ndcg_cut." + ",".join([str(k) for k in at_k])
recall = "recall." + ",".join([str(k) for k in at_k])
precision = "P." + ",".join([str(k) for k in at_k])
rprecision = "Rprec." + ",".join([str(k) for k in at_k])
measures = {map, ndcg, recall, precision, rprecision}


def eval_bm25(
    qrels: Dict[str, Dict[str, float]],
    run: Dict[str, Dict[str, float]],
    exclude_annotations: Dict[str, List[Dict]] = None,
) -> Dict[str, Dict[str, float]]:

    if exclude_annotations is not None:
        filtered_qrels, filtered_run = {}, {}
        for topic_id, annotation in exclude_annotations.items():
            annotated_doc_ids = [a["doc_id"] for a in annotation]

            filtered_qrels[topic_id] = dict(
                filter(
                    lambda doc_rel: doc_rel[0] not in annotated_doc_ids,
                    qrels[topic_id].items(),
                )
            )
            filtered_run[topic_id] = dict(
                filter(
                    lambda doc_rel: doc_rel[0] not in annotated_doc_ids,
                    run[topic_id].items(),
                )
            )

        run = filtered_run
        qrels = filtered_qrels

    evaluator = RelevanceEvaluator(qrels, measures)
    results = evaluator.evaluate(run)

    return results


def accumulate_results(
    bm25_eval: Dict[str, Dict[str, float]], topic_ids: List[str] = None
) -> Dict[str, float]:
    acc_results = defaultdict(list)
    for topic_id, measure2score in bm25_eval.items():
        if topic_ids is not None and not topic_id in topic_ids:
            continue
        for measure, score in measure2score.items():
            acc_results[measure].append(score)
    mean = {k: np.mean(v) for k, v in acc_results.items()}
    std = {k: np.std(v) for k, v in acc_results.items()}

    return {"mean": mean, "std": std}
