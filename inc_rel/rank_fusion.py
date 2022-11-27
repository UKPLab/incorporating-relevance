import json
import os
from collections import defaultdict

import simple_parsing
from args import RankFusion
from eval import accumulate_results
from reranking_evaluator import RerankingEvaluator


def rank_fusion(rank: int, c: int = 60):
    return 1 / (c + rank)


def main(args):

    # load result files
    results = []
    for file in args.result_files:
        with open(file) as fh:
            results.append(json.load(fh))

    # compute rank fusion scores
    query_to_doc_to_scores = defaultdict(lambda: defaultdict(float))
    for result in results:
        for query_id, doc_to_score in result.items():
            for rank, doc_id in enumerate(doc_to_score.keys(), start=1):
                query_to_doc_to_scores[query_id][doc_id] += rank_fusion(rank)

    with open(os.path.join(args.data_path, "qrels.json")) as fh:
        qrels = json.load(fh)
    ranking_evaluator = RerankingEvaluator(qrels)
    result = ranking_evaluator.eval(query_to_doc_to_scores)

    split2metric = defaultdict(list)
    for seed in args.seeds:
        print(f"---seed={seed}---")
        for split in args.splits:
            with open(
                os.path.join(
                    args.data_path, f"k{args.num_samples}", f"s{seed}", f"{split}.json"
                )
            ) as fh:
                split_seed = json.load(fh)

            split_seed_eval_acc = accumulate_results(
                result, topic_ids=list(split_seed.keys())
            )

            with open(
                os.path.join(
                    args.data_path,
                    f"k{args.num_samples}",
                    f"s{seed}",
                    f"{split}_rank-fusion_{args.prefix}_eval_acc.json",
                ),
                "w",
            ) as fh:
                json.dump(split_seed_eval_acc, fh, indent=4)
            split2metric[split].append(split_seed_eval_acc["mean"][args.metric])
            print(
                f"k={args.num_samples:02d} split={split:5s} seed={seed:02d} "
                f"{args.metric}={split_seed_eval_acc['mean'][args.metric]:.4f}"
            )

    print("---MEAN---")
    for split in args.splits:
        print(
            f"k={args.num_samples:02d} split={split:5s} "
            f"{args.metric}={sum(split2metric[split]) / len(split2metric[split]):.4f}"
        )


if __name__ == "__main__":
    args = simple_parsing.parse(RankFusion)
    print(args)
    main(args)
