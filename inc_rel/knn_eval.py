import ast
import json
import operator
import os
from collections import defaultdict
from typing import Dict, List

import eval
import simple_parsing
from args import Experiment
from reranking_evaluator import RerankingEvaluator


def collect_similarities(
    qrels,
    annotations,
    similarities,
    expansion_results,
    query_sim: bool,
    annot_sim: bool,
    relevant_or_non_relevant: str,
):
    sims = defaultdict(lambda: defaultdict(list))
    if relevant_or_non_relevant == "relevant":
        op = operator.gt
    elif relevant_or_non_relevant == "non-relevant":
        op = operator.le
    else:
        raise ValueError
    for topic_id in qrels.keys():
        if query_sim:
            # add query:doc similarity score
            for doc_id in expansion_results[topic_id].keys():
                sims[topic_id][doc_id].append(similarities[(topic_id, doc_id)])
        if annot_sim:
            # add relevant-annotation:doc similarity score
            for annotation in annotations[topic_id]:
                if op(annotation["label"], 0):
                    for doc_id in expansion_results[topic_id].keys():
                        sims[topic_id][doc_id].append(
                            similarities[(annotation["doc_id"], doc_id)]
                        )
    return sims


def aggregate_sim_collection(sims, fn):
    run = defaultdict(dict)
    for topic_id, docids2scores in sims.items():
        for doc_id, scores in docids2scores.items():
            run[topic_id][doc_id] = fn(scores)
    return run


def main(args):
    with open(
        os.path.join(
            args.data_path,
            f"knn_similarities_{args.prefix}.json",
        )
    ) as fh:
        similarities = json.load(fh)
    similarities = {ast.literal_eval(k): v for k, v in similarities.items()}

    with open(os.path.join(args.data_path, "qrels.json")) as fh:
        qrels = json.load(fh)

    for k in [2, 4, 8]:
        with open(
            os.path.join(args.data_path, f"k{k}", f"expansion_results_16.json")
        ) as fh:
            expansion_results: Dict[str, Dict] = json.load(fh)

        with open(os.path.join(args.data_path, f"annotations_{k}.json")) as fh:
            annotations: Dict[str, List[Dict]] = json.load(fh)

        query_only_sims = collect_similarities(
            qrels,
            annotations,
            similarities,
            expansion_results,
            query_sim=True,
            annot_sim=False,
            relevant_or_non_relevant="relevant",
        )
        annot_only_sims = collect_similarities(
            qrels,
            annotations,
            similarities,
            expansion_results,
            query_sim=False,
            annot_sim=True,
            relevant_or_non_relevant="relevant",
        )
        query_annot_sims = collect_similarities(
            qrels,
            annotations,
            similarities,
            expansion_results,
            query_sim=True,
            annot_sim=True,
            relevant_or_non_relevant="relevant",
        )

        results = {}
        full_result = defaultdict(dict)
        for sim, sim_name in zip(
            [query_only_sims, annot_only_sims, query_annot_sims],
            ["query", "annot", "query_annot"],
        ):
            for fn, fn_name in zip([sum, max], ["sum", "max"]):
                sims_agg = aggregate_sim_collection(sim, fn=fn)
                results[(k, sim_name, fn_name)] = RerankingEvaluator(qrels).eval(
                    sims_agg
                )

                full_result[(k, sim_name, fn_name)] = [
                    {
                        "topic_id": topic_id,
                        "metrics": results[(k, sim_name, fn_name)][topic_id],
                        "run": {
                            topic_id: dict(
                                sorted(
                                    sims_agg[topic_id].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                        },
                    }
                    for topic_id in sim.keys()
                ]

        annot_non_rel_sims = collect_similarities(
            qrels,
            annotations,
            similarities,
            expansion_results,
            query_sim=False,
            annot_sim=True,
            relevant_or_non_relevant="non-relevant",
        )
        annot_non_rel_agg = aggregate_sim_collection(sim, fn=sum)
        sim_name = "annot_non_rel"
        for fn, fn_name in zip([sum, max], ["sum", "max"]):
            sims_agg = aggregate_sim_collection(sim, fn=fn)
            for topic_id in annot_non_rel_agg.keys():
                for doc_id in annot_non_rel_agg[topic_id].keys():
                    sims_agg[topic_id][doc_id] -= annot_non_rel_agg[topic_id][doc_id]
            results[(k, sim_name, fn_name)] = RerankingEvaluator(qrels).eval(sims_agg)
            full_result[(k, sim_name, fn_name)] = [
                {
                    "topic_id": topic_id,
                    "metrics": results[(k, sim_name, fn_name)][topic_id],
                    "run": {
                        topic_id: dict(
                            sorted(
                                sims_agg[topic_id].items(),
                                key=lambda item: item[1],
                                reverse=True,
                            )
                        )
                    },
                }
                for topic_id in sim.keys()
            ]
            # results_acc = eval.accumulate_results(results)

        for key in full_result.keys():
            _k, sim_name, fn_name = key
            with open(
                os.path.join(
                    args.data_path,
                    f"k{_k}",
                    f"knn_{args.prefix}_{sim_name}-{fn_name}_eval.json",
                ),
                "w",
            ) as fh:
                json.dump(full_result[key], fh, indent=4)

        split2metric = defaultdict(list)
        for seed in args.seeds:
            for split in args.splits:
                with open(
                    os.path.join(args.data_path, f"k{k}", f"s{seed}", f"{split}.json")
                ) as fh:
                    split_seed = json.load(fh)

                for exp_name, result in results.items():
                    split_seed_eval_acc = eval.accumulate_results(
                        result, topic_ids=list(split_seed.keys())
                    )
                    _, sim_name, fn_name = exp_name

                    with open(
                        os.path.join(
                            args.data_path,
                            f"k{k}",
                            f"s{seed}",
                            f"{split}_knn_{args.prefix}_{sim_name}-{fn_name}_eval_acc.json",
                        ),
                        "w",
                    ) as fh:
                        json.dump(split_seed_eval_acc, fh, indent=4)
                    if sim_name == "query_annot" and fn_name == "sum":
                        # works best
                        split2metric[split].append(
                            split_seed_eval_acc["mean"][args.metric]
                        )
                        print(
                            f"k={k:02d} split={split:5s} seed={seed:02d} "
                            f"{args.metric}={split_seed_eval_acc['mean'][args.metric]:.4f}"
                        )

        print("---MEAN---")
        for split in args.splits:
            print(
                f"k={k:02d} split={split:5s} "
                f"{args.metric}={sum(split2metric[split]) / len(split2metric[split]):.4f}"
            )


if __name__ == "__main__":
    args = simple_parsing.parse(Experiment)
    main(args)
