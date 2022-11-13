import json
import os
from collections import defaultdict

import simple_parsing
from args import ZeroShot
from eval import accumulate_results
from reranking_evaluator import RerankingEvaluator
from sentence_transformers import CrossEncoder, SentenceTransformer


def main(args):

    base_path = os.path.join(args.data_path, f"k{args.num_samples}")
    results_file = os.path.join(base_path, f"expansion_results_16.json")
    docs_file = os.path.join(base_path, f"expansion_docs_16.json")

    with open(results_file) as fh:
        bm25_results = json.load(fh)
    with open(docs_file) as fh:
        bm25_docs = json.load(fh)
    with open(os.path.join(args.data_path, "qrels.json")) as fh:
        qrels = json.load(fh)
    with open(os.path.join(args.data_path, "topics.json")) as fh:
        topics = json.load(fh)

    assert len(topics) == len(qrels) == len(bm25_results), (
        len(topics),
        len(qrels),
        len(bm25_results),
    )

    if args.model_class == "ce":
        model_class = CrossEncoder
    elif args.model_class == "bi":
        model_class = SentenceTransformer
    model = model_class(args.model)
    if args.model_ctx is not None:
        model_ctx = model_class(args.model_ctx)
    else:
        model_ctx = None

    eval_results, run = RerankingEvaluator(qrels)(
        model,
        model_ctx=model_ctx,
        queries=topics,
        docs=bm25_docs,
        inital_ranking=bm25_results,
        batch_size=128,
        show_progress_bar=False,
        scoring_fn=args.scoring_fn,
    )

    # assert len(eval_results) == len(qrels)

    exp_name = f"{args.prefix}_{args.model_class}"
    with open(
        os.path.join(
            args.data_path,
            f"k{args.num_samples}",
            f"zero_shot_{exp_name}_eval.json",
        ),
        "w",
    ) as fh:
        json.dump(eval_results, fh, indent=4)
    with open(
        os.path.join(
            args.data_path,
            f"k{args.num_samples}",
            f"zero_shot_{exp_name}_eval_acc.json",
        ),
        "w",
    ) as fh:
        json.dump(accumulate_results(eval_results), fh, indent=4)
    with open(
        os.path.join(
            args.data_path,
            f"k{args.num_samples}",
            f"zero_shot_{exp_name}_results.json",
        ),
        "w",
    ) as fh:
        json.dump(run, fh, indent=4)

    splits = ["train", "valid", "test"]
    for seed in args.seeds:
        split2metric = defaultdict(list)
        for split in splits:
            with open(
                os.path.join(
                    args.data_path,
                    f"k{args.num_samples}",
                    f"s{seed}",
                    f"{split}.json",
                )
            ) as fh:
                split_seed = json.load(fh)

            split_seed_eval_acc = accumulate_results(
                eval_results, topic_ids=list(split_seed.keys())
            )

            with open(
                os.path.join(
                    args.data_path,
                    f"k{args.num_samples}",
                    f"s{seed}",
                    f"{split}_zero_shot_{exp_name}_eval_acc.json",
                ),
                "w",
            ) as fh:
                json.dump(split_seed_eval_acc, fh, indent=4)

            split2metric[split].append(split_seed_eval_acc["mean"][args.metric])
            print(
                f"split={split:5s} seed={seed:03d} "
                f"{args.metric}={split_seed_eval_acc['mean'][args.metric]:.4f}"
            )

    print("---MEAN---")
    for split in splits:
        print(
            f"split={split:5s} {args.metric}={sum(split2metric[split]) / len(split2metric[split]):.4f}"
        )


if __name__ == "__main__":
    args = simple_parsing.parse(ZeroShot)
    main(args)
