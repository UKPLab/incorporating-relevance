import json
import os
import shutil
from collections import defaultdict

import simple_parsing
import tqdm
from args import PreTrain
from few_shot_trainer import FewShotTrainer
from pre_train_trainer import PreTrainTrainer
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

    annotations = defaultdict(dict)
    for split in ["train", "valid", "test"]:
        with open(
            os.path.join(
                args.data_path, f"k{args.num_samples}", f"s{args.seed}", f"{split}.json"
            )
        ) as fh:
            annotations[split] = json.load(fh)

    # PRE-TRAIN model on training set
    results = []
    lr2best_metric = {}
    search_steps = args.epochs * len(args.learning_rates)
    with tqdm.tqdm(
        total=search_steps,
        ncols=100,
        desc="pre-train h-param search",
    ) as pbar:

        pre_train_trainer = PreTrainTrainer(
            args.model,
            args.ft_params,
            bm25_docs,
            bm25_results,
            ranking_evaluator,
            meta=args.pretrain_method == "meta",
            pbar=pbar,
        )

        for learning_rate in args.learning_rates:
            exp_dir = os.path.join(
                os.path.dirname(args.out_file),
                f"pt-exp_{args.model_class}"
                f"_{args.ft_params}_{args.pretrain_method}"
                f"_lr-{learning_rate:.8f}",
            )
            result, best_epoch, best_metric = pre_train_trainer.train(
                train_annotations=annotations["train"],
                eval_annotations=annotations["valid"],
                epochs=args.epochs,
                learning_rate=learning_rate,
                exp_dir=exp_dir,
                selection_metric="ndcg_cut_20",
            )
            lr2best_metric[learning_rate] = best_metric
            results.append({"result": result, "best_epoch": best_epoch})
            # write out results after each experiment
            with open(args.out_file, "w") as fh:
                json.dump(results, fh, indent=4)

    best_lr = max(lr2best_metric, key=lr2best_metric.get)
    for learning_rate in args.learning_rates:
        exp_dir = os.path.join(
            os.path.dirname(args.out_file),
            f"pt-exp_{args.model_class}"
            f"_{args.ft_params}_{args.pretrain_method}"
            f"_lr-{learning_rate:.8f}",
        )
        if learning_rate == best_lr:
            best_exp_dir = os.path.join(
                os.path.dirname(args.out_file),
                f"pt-exp_{args.model_class}"
                f"_{args.ft_params}_{args.pretrain_method}",
            )
            os.rename(exp_dir, best_exp_dir)
        else:
            shutil.rmtree(exp_dir)

    # FINE-TUNE model on valid set
    search_steps = args.epochs * len(args.learning_rates) * len(annotations["valid"])
    with tqdm.tqdm(
        total=search_steps,
        ncols=100,
        desc="few-shot h-param search",
    ) as pbar:
        few_shot_trainer = FewShotTrainer(
            model_name=args.model,
            ft_params=args.ft_params,
            docs=bm25_docs,
            inital_ranking=bm25_results,
            ranking_evaluator=ranking_evaluator,
            pbar=pbar,
            model_path=os.path.join(best_exp_dir, "model"),
        )

        args.out_file = (
            args.data_path,
            f"k{args.num_samples}",
            f"s{args.seed}",
            (
                f"valid_{args.model_class}_{args.ft_params}"
                f"_{args.pretrain_method}_pre_train_few_shot_hpsearch.json"
            ),
        )

        results = []
        for learning_rate in args.learning_rates:
            exp_results = few_shot_trainer.train(
                annotations["valid"], epochs=args.epochs, learning_rate=learning_rate
            )
            results.extend(exp_results)
            # write out results after each experiment
            with open(args.out_file, "w") as fh:
                json.dump(results, fh, indent=4)

    # Get best model and eval on train/valid/test set
    best_epochs, best_learning_rate = get_best_experiment(results, "ndcg_cut_20")
    for split in ["train", "valid", "test"]:
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
                f"s{args.seed}",
                (
                    f"{split}_{args.model_class}_{args.ft_params}"
                    f"_{args.pretrain_method}_pre_train_few_shot.json"
                ),
            ),
            "w",
        ) as fh:
            json.dump(few_shot_results, fh, indent=4)


if __name__ == "__main__":
    args = simple_parsing.parse(PreTrain)
    main(args)
