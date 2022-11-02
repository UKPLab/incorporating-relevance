import json
import os
from typing import Dict, List

import numpy as np
import simple_parsing
from args import KNNSimilarities
from sentence_transformers import util
from tqdm.auto import tqdm


def main(args):

    if args.scoring_fn == "cos":
        scoring_fn = util.cos_sim
    elif args.scoring_fn == "dot":
        scoring_fn = util.dot_score

    embeddings = {}
    for name in ["queries", "annotations", "docs"]:
        with open(
            os.path.join(
                args.data_path,
                f"knn_{name}_{args.prefix}_embeddings.json",
            )
        ) as fh:
            embeddings[name] = json.load(fh)

    with open(os.path.join(args.data_path, "annotations_8.json")) as fh:
        annotations: Dict[Dict[List]] = json.load(fh)

    similarites = {}
    for k in [2, 4, 8]:
        with open(
            os.path.join(args.data_path, f"k{k}", f"expansion_results.json")
        ) as fh:
            expansion_results: Dict[Dict] = json.load(fh)
        for topic_id, doc_id2score in tqdm(
            expansion_results.items(),
            total=len(expansion_results),
            desc="expansion_results",
        ):
            query_embedding = np.asarray(embeddings["queries"][topic_id]).reshape(1, -1)
            doc_ids = doc_id2score.keys()
            doc_embeddings = []
            new_keys = []
            for doc_id in doc_ids:
                key = (topic_id, doc_id)
                if key not in similarites:
                    new_keys.append(key)
                    doc_embeddings.append(embeddings["docs"][doc_id])

            if doc_embeddings:
                doc_embeddings = np.asarray(doc_embeddings)
                cos_sim = scoring_fn(query_embedding, doc_embeddings)
                for j, key in enumerate(new_keys):
                    similarites[key] = cos_sim[0, j].item()
            else:
                print(f"No new query-doc pairs found for topic={topic_id} at k={k}")

        for topic_id, annotation in tqdm(
            annotations.items(), total=len(annotations), desc="Annotations"
        ):
            annotation_doc_ids = [a["doc_id"] for a in annotation]
            annotation_embeddings = [
                embeddings["annotations"][doc_id] for doc_id in annotation_doc_ids
            ]

            for i, annotation_doc_id in enumerate(annotation_doc_ids):
                doc_embeddings = []
                new_keys = []
                for doc_id in expansion_results[topic_id].keys():
                    key = (annotation_doc_id, doc_id)
                    if key not in similarites:
                        new_keys.append(key)
                        doc_embeddings.append(embeddings["docs"][doc_id])
                if doc_embeddings:
                    doc_embeddings = np.asarray(doc_embeddings)
                    annotation_embedding = np.asarray(annotation_embeddings[i]).reshape(
                        1, -1
                    )
                    cos_sim = util.cos_sim(annotation_embedding, doc_embeddings)
                    for j, key in enumerate(new_keys):
                        similarites[key] = cos_sim[0, j].item()
                else:
                    print(
                        f"No new annot-doc pairs found for topic={topic_id} at k={k} for annot={i, annotation_doc_id}"
                    )

    with open(
        os.path.join(
            args.data_path,
            f"knn_similarities_{args.prefix}.json",
        ),
        "w",
    ) as fh:
        json.dump({str(k): v for k, v in similarites.items()}, fh, indent=4)


if __name__ == "__main__":
    args = simple_parsing.parse(KNNSimilarities)
    main(args)
