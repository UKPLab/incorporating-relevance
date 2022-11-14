import json
import os
from collections import defaultdict

import simple_parsing
import tqdm
from args import FineTuneExperiment
from few_shot_trainer import FewShotTrainer
from reranking_evaluator import RerankingEvaluator
from utils import get_best_experiment


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

    # only keep topics that ended up in the dataset
    topics = dict(
        filter(
            lambda topic_id2query: topic_id2query[0] in bm25_results.keys(),
            topics.items(),
        )
    )

    ranking_evaluator = RerankingEvaluator(qrels)
    split2metric = defaultdict(list)
    for seed in args.seeds:
        print(f"---seed={seed}---")
        annotations = defaultdict(dict)
        for split in args.splits:
            with open(
                os.path.join(
                    args.data_path, f"k{args.num_samples}", f"s{seed}", f"{split}.json"
                )
            ) as fh:
                annotations[split] = json.load(fh)

        search_steps = (
            args.epochs * len(args.learning_rates) * len(annotations["valid"])
        )

        with tqdm.tqdm(
            total=search_steps,
            ncols=100,
            desc=f"few-shot h-param search seed={seed}",
        ) as pbar:

            few_shot_trainer = FewShotTrainer(
                args.model,
                args.ft_params,
                bm25_docs,
                bm25_results,
                ranking_evaluator,
                pbar=pbar,
            )

            results = []
            for learning_rate in args.learning_rates:
                exp_results = few_shot_trainer.train(
                    annotations["valid"], args.epochs, learning_rate
                )
                results.extend(exp_results)
                # write out results after each experiment
                with open(args.out_file.format(seed=seed), "w") as fh:
                    json.dump(results, fh, indent=4)

            best_epochs, best_learning_rate = get_best_experiment(
                results, "ndcg_cut_20"
            )
            print(f"Best Epoch={best_epochs}. Best LR={best_learning_rate}")
            for split in args.splits:
                few_shot_results = few_shot_trainer.train(
                    annotations[split],
                    best_epochs,
                    best_learning_rate,
                    iter_epochs=False,
                    update_pbar=False,
                )
                with open(
                    os.path.join(
                        args.data_path,
                        f"k{args.num_samples}",
                        f"s{seed}",
                        f"{split}_{args.model_class}_{args.ft_params}_16_few_shot.json",
                    ),
                    "w",
                ) as fh:
                    json.dump(few_shot_results, fh, indent=4)

                mean_metric = 0
                for result in few_shot_results:
                    mean_metric += (
                        1
                        / len(few_shot_results)
                        * result["metrics"][list(result["metrics"].keys())[0]][
                            args.metric
                        ]
                    )
                print(
                    f"k={args.num_samples:02d} split={split:5s} "
                    f"{args.metric}={mean_metric:.4f}"
                )
                split2metric[split].append(mean_metric)

    print("---MEAN---")
    for split in args.splits:
        print(
            f"k={args.num_samples:02d} split={split:5s} "
            f"{args.metric}={sum(split2metric[split]) / len(split2metric[split]):.4f}"
        )


if __name__ == "__main__":
    args = simple_parsing.parse(FineTuneExperiment)
    main(args)
