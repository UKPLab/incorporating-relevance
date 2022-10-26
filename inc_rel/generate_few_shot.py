import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

import utils
from settings import dataset_settings_cls
from datasets import datasets_cls
from index import Index


def main(args):

    index = Index()

    dataset = datasets_cls[args.name](
        corpus_path=args.corpus_path,
        topics_path=args.topics_path,
        qrels_path=args.qrels_path,
        load_corpus=not index.index_exists(index=args.name)
        or args.name in ["trec-news"],
        remove_topics_with_few_negatives=args.enrich_bm25_path is None,
    )

    if not index.index_exists(index=args.name):
        index.index_corpus(index=args.name, corpus=dataset.corpus)

        for _ in tqdm(range(120), ncols=100, desc="Granting ES some beauty sleep."):
            time.sleep(1)

    # do first bm25 retrieval
    bm25_results, bm25_docs, _ = index.bm25_query(
        index=args.name, topics=dataset.topics, size=args.bm25_size
    )

    # we can use doc_id2text as corpus here because the bm25 docs are a superset of
    # the candidates
    max_k = max(args.num_samples)
    candidates = utils.sorted_annotated_retrieved_docs(
        corpus=bm25_docs,
        qrels=dataset.qrels,
        bm25_results=bm25_results,
        min_relevant_relevance=dataset.min_relevant_relevancy,
        remove_duplicates=args.remove_duplicates,
        enrich_negatives_from_bm25=args.enrich_bm25_path,
        num_non_relevant=max_k,
    )

    dataset_path = args.output_path / args.name / str(args.bm25_size)
    os.makedirs(dataset_path, exist_ok=True)

    # check if each topic has enough candidates in the bm25 results
    annotations = defaultdict(lambda: defaultdict(list))
    for topic_id, query in dataset.topics.copy().items():
        relevant_candidates = candidates[topic_id]["relevant"][:max_k]
        non_relevant_candidates = candidates[topic_id]["non_relevant"][:max_k]
        if not max_k == len(relevant_candidates) == len(non_relevant_candidates):

            print(
                f"Could not find enough candidates for {topic_id=}. "
                f"{len(relevant_candidates)=} {len(non_relevant_candidates)=} "
            )
            candidates.pop(topic_id)
            dataset.remove_topic(topic_id)

    # re-run bm25-query, since topics might have been removed
    bm25_results, bm25_docs, bm25_topics2time = index.bm25_query(
        index=args.name, topics=dataset.topics, size=args.bm25_size
    )
    bm25_eval = eval.eval_bm25(dataset.qrels, bm25_results)
    bm25_eval_acc = eval.accumulate_bm25_results(bm25_eval)

    # find annotations
    for topic_id, query in dataset.topics.items():
        relevant_candidates = candidates[topic_id]["relevant"][:max_k]
        non_relevant_candidates = candidates[topic_id]["non_relevant"][:max_k]
        for rel_doc_id, non_doc_id in zip(relevant_candidates, non_relevant_candidates):
            # if datsets has been enriched with bm25 negatives, qrels might not have
            # a doc, therefore we label it as 0
            rel_label = dataset.qrels[topic_id][rel_doc_id]
            rel_document = bm25_docs[rel_doc_id]

            non_label = dataset.qrels[topic_id].get(non_doc_id, 0)
            # incase of bm25 negatives, negatives might not be in bm_docs
            non_document = bm25_docs.get(
                non_doc_id, index.get_doc_by_id(args.name, non_doc_id)
            )
            for k in args.num_samples:
                # has to be 2k, since there are k relevant and k non-relevant
                if len(annotations[k][topic_id]) < 2 * k:
                    for doc_id, label, document in zip(
                        [rel_doc_id, non_doc_id],
                        [rel_label, non_label],
                        [rel_document, non_document],
                    ):
                        annotations[k][topic_id].append(
                            {
                                "topic_id": topic_id,
                                "doc_id": doc_id,
                                "label": label,
                                "query": query,
                                "document": document,
                            }
                        )
    for k, annotation in annotations.items():
        with open(dataset_path / f"annotations_{k}.json", "w") as fh:
            json.dump(annotation, fh, indent=4)

    dataset.remove_annotations_from_qrels(annotations[max_k])

    # exclude annotations here, since bm25_results contain annotated docs
    bm25_removed_eval = eval.eval_bm25(
        dataset.qrels, bm25_results, exclude_annotations=annotations[max_k]
    )
    bm25_removed_eval_acc = eval.accumulate_bm25_results(bm25_removed_eval)

    for obj, name in zip(
        [
            dataset.topics,
            dataset.qrels,
            bm25_results,
            bm25_docs,
            bm25_eval,
            bm25_eval_acc,
            bm25_removed_eval,
            bm25_removed_eval_acc,
            bm25_topics2time,
        ],
        [
            "topics",
            "qrels",
            "bm25_results",
            "bm25_docs",
            "bm25_eval",
            "bm25_eval_accumulated",
            "bm25_annot_removed_eval",
            "bm25_annot_removed_eval_accumulated",
            "bm25_topics2time",
        ],
    ):
        with open(dataset_path / f"{name}.json", "w") as fh:
            json.dump(obj, fh, indent=4)

    for k in args.num_samples:
        expan_results, expan_docs, expan_eval, expan_acc, expan_topic2time = (
            {},
            {},
            {},
            {},
            {},
        )
        # expand bm25 query with annotated docs
        bm25_key = "full"
        (
            expan_results[bm25_key],
            expan_docs[bm25_key],
            expan_topic2time[bm25_key],
        ) = index.bm25_query_expansion(
            index=args.name,
            topics=dataset.topics,
            annotations=annotations[k],
            rm_annotations=annotations[max_k],
            size=args.bm25_size,
        )
        expan_eval[bm25_key] = eval.eval_bm25(
            qrels=dataset.qrels,
            run=expan_results[bm25_key],
        )
        expan_acc[bm25_key] = eval.accumulate_bm25_results(expan_eval[bm25_key])

        for max_query_terms in [4, 8, 16, 32, 64]:
            mlt_key = max_query_terms
            (
                expan_results[mlt_key],
                expan_docs[mlt_key],
                expan_topic2time[mlt_key],
            ) = index.more_like_this_bool_expansion(
                index=args.name,
                topics=dataset.topics,
                annotations=annotations[k],
                rm_annotations=annotations[max_k],
                size=args.bm25_size,
                max_query_terms=max_query_terms,
            )
            expan_eval[mlt_key] = eval.eval_bm25(
                qrels=dataset.qrels,
                run=expan_results[mlt_key],
            )
            expan_acc[mlt_key] = eval.accumulate_bm25_results(expan_eval[mlt_key])

        sample_path = args.output_path / args.name / str(args.bm25_size) / f"k{k}"
        os.makedirs(sample_path)
        for key in expan_results.keys():
            for name, obj in zip(
                ["results", "docs", "eval", "eval_acc", "times"],
                [expan_results, expan_docs, expan_eval, expan_acc, expan_topic2time],
            ):
                with open(sample_path / f"expansion_{name}_{key}.json", "w") as fh:
                    json.dump(obj[key], fh, indent=4)

        for split_seed in args.split_seeds:
            split_path = sample_path / f"s{str(split_seed)}"
            os.makedirs(split_path)

            train_topics, valid_topics, test_topics = utils.create_split(
                topic_ids=list(annotations[k].keys()),
                seed=split_seed,
                split_sizes=args.split_sizes,
            )

            splits = {}
            for split_name, topic_ids in zip(
                ["train", "valid", "test"], [train_topics, valid_topics, test_topics]
            ):
                splits[split_name] = {
                    topic_id: v
                    for topic_id, v in annotations[k].items()
                    if topic_id in topic_ids
                }
                split_eval = {}

                split_eval[("bm25", split_name)] = eval.accumulate_bm25_results(
                    bm25_eval, list(splits[split_name].keys())
                )
                split_eval[("bm25-removed", split_name)] = eval.accumulate_bm25_results(
                    bm25_removed_eval, list(splits[split_name].keys())
                )

                for key in expan_results.keys():
                    split_eval[(key, split_name)] = eval.accumulate_bm25_results(
                        expan_eval[key], list(splits[split_name].keys())
                    )

                for split_name, split in splits.items():
                    with open(split_path / f"{split_name}.json", "w") as fh:
                        json.dump(split, fh, indent=4)

                for key in split_eval.keys():
                    if isinstance(key[0], str):
                        file_name = f"{split_name}_{key[0]}"
                    else:
                        file_name = f"{split_name}_expansion_{key[0]}"
                    with open(split_path / f"{file_name}_eval_acc.json", "w") as fh:
                        json.dump(split_eval[key], fh, indent=4)

            config = vars(args)
            config["k"] = k
            config["seed"] = split_seed
            config["min_annotations"] = dataset.min_annotations
            config["min_relevant_relevancy"] = dataset.min_relevant_relevancy
            config = {
                config_k: str(v) if isinstance(v, Path) else v
                for config_k, v in config.items()
            }
            with open(split_path / "config.json", "w") as fh:
                json.dump(config, fh, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()

    dataset_settings = dataset_settings_cls[args.dataset]()

    main(dataset_settings)
