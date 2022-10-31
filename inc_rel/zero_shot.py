import argparse
import json
import os

from sentence_transformers import CrossEncoder, SentenceTransformer

from eval import accumulate_results
from reranking_evaluator import RerankingEvaluator
from settings import ZeroShotSettings, dataset_settings_cls


def main(dataset_settings, zero_shot_settings, num_samples):

    data_path = os.path.join(
        dataset_settings.data_path, str(dataset_settings.bm25_size)
    )
    base_path = os.path.join(data_path, f"k{num_samples}")
    results_file = os.path.join(
        base_path, f"expansion_results_{zero_shot_settings.rerank}.json"
    )
    docs_file = os.path.join(
        base_path, f"expansion_docs_{zero_shot_settings.rerank}.json"
    )

    with open(results_file) as fh:
        bm25_results = json.load(fh)
    with open(docs_file) as fh:
        bm25_docs = json.load(fh)
    with open(os.path.join(data_path, "qrels.json")) as fh:
        qrels = json.load(fh)
    with open(os.path.join(data_path, "topics.json")) as fh:
        topics = json.load(fh)

    assert len(topics) == len(qrels) == len(bm25_results), (
        len(topics),
        len(qrels),
        len(bm25_results),
    )

    if zero_shot_settings.model_class == "ce":
        model_class = CrossEncoder
    elif zero_shot_settings.model_class == "bi":
        model_class = SentenceTransformer
    model = model_class(zero_shot_settings.model)
    if zero_shot_settings.model_ctx:
        model_ctx = model_class(zero_shot_settings.model_ctx)
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
        scoring_fn=zero_shot_settings.scoring_fn,
    )

    # assert len(eval_results) == len(qrels)

    exp_name = f"{zero_shot_settings.prefix}_{zero_shot_settings.model_class}_{zero_shot_settings.rerank}"
    with open(
        os.path.join(
            data_path,
            f"k{num_samples}",
            f"zero_shot_{exp_name}_eval.json",
        ),
        "w",
    ) as fh:
        json.dump(eval_results, fh, indent=4)
    with open(
        os.path.join(
            data_path,
            f"k{num_samples}",
            f"zero_shot_{exp_name}_eval_acc.json",
        ),
        "w",
    ) as fh:
        json.dump(accumulate_results(eval_results), fh, indent=4)
    with open(
        os.path.join(
            data_path,
            f"k{num_samples}",
            f"zero_shot_{exp_name}_results.json",
        ),
        "w",
    ) as fh:
        json.dump(run, fh, indent=4)

    for seed in zero_shot_settings.seeds:
        for split in ["train", "valid", "test"]:
            with open(
                os.path.join(
                    data_path,
                    f"k{num_samples}",
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
                    data_path,
                    f"k{num_samples}",
                    f"s{seed}",
                    f"{split}_zero_shot_{exp_name}_eval_acc.json",
                ),
                "w",
            ) as fh:
                json.dump(split_seed_eval_acc, fh, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-k", "--num-samples", type=int)

    args = parser.parse_args()
    dataset_settings = dataset_settings_cls[args.dataset]()
    zero_shot_settings = ZeroShotSettings()

    main(dataset_settings, zero_shot_settings, num_samples=args.num_samples)
