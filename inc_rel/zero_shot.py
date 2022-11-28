import json
import os
from collections import defaultdict

import simple_parsing
from args import ZeroShot
from eval import accumulate_results
from reranking_evaluator import RerankingEvaluator


def main(args):

    model = args.model_class(args.model)
    if args.model_ctx is not None:
        model_ctx = args.model_class(args.model_ctx)
    else:
        model_ctx = None

    eval_results, run = RerankingEvaluator(args.qrels)(
        model=model,
        model_ctx=model_ctx,
        queries=args.topics,
        docs=args.bm25_docs,
        inital_ranking=args.bm25_results,
        batch_size=128,
        show_progress_bar=False,
        scoring_fn=args.scoring_fn,
    )

    eval_file = os.path.join(args.exp_path, f"k{args.num_samples}_eval.json")
    with open(eval_file, "w") as fh:
        json.dump(eval_results, fh, indent=4)

    eval_acc_file = os.path.join(args.exp_path, f"k{args.num_samples}_eval_acc.json")
    with open(
        eval_acc_file,
        "w",
    ) as fh:
        json.dump(accumulate_results(eval_results), fh, indent=4)

    results_file = os.path.join(args.exp_path, f"k{args.num_samples}_results.json")
    with open(results_file, "w") as fh:
        json.dump(run, fh, indent=4)

    split2metric = defaultdict(list)
    for seed in args.seeds:
        for split in args.splits:
            topic_ids = list(args.topic_ids_split_seed[split, seed].keys())
            split_seed_eval_acc = accumulate_results(eval_results, topic_ids=topic_ids)

            split_seed_eval_acc_file = os.path.join(
                args.exp_path, f"k{args.num_samples}_s{seed}_{split}_eval_acc.json"
            )
            with open(split_seed_eval_acc_file, "w") as fh:
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
    args = simple_parsing.parse(ZeroShot)
    main(args)
