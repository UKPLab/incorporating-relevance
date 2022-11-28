import json
import os
from typing import Dict, List

import simple_parsing
import torch
from args import KNNIndex
from sentence_transformers import SentenceTransformer


def main(args):

    docs = {}
    for k in [2, 4, 8]:
        base_path = os.path.join(args.data_path, f"k{k}")
        docs_file = os.path.join(base_path, f"expansion_docs_16.json")
        with open(docs_file) as fh:
            docs.update(json.load(fh))

    with open(os.path.join(args.data_path, "topics.json")) as fh:
        queries = json.load(fh)

    with open(os.path.join(args.data_path, "annotations_8.json")) as fh:
        annotations: Dict[List[Dict]] = json.load(fh)
    annotations = {
        a["doc_id"]: a["document"]
        for annotation in annotations.values()
        for a in annotation
    }

    print(
        f"Loaded {len(queries)} queries "
        f"{len(annotations)} annotation docs "
        f"and {len(docs)} retrieval docs."
    )

    model = SentenceTransformer(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for name, id2doc in zip(
            ["queries", "annotations", "docs"], [queries, annotations, docs]
        ):
            embeddings_path = os.path.join(args.exp_path, f"{name}_embeddings.json")
            if os.path.exists(embeddings_path):
                print(f"{name} embeddings already exists.")
            else:
                texts = list(id2doc.values())
                texts_embeddings = model.encode(texts, show_progress_bar=True)
                texts_embeddings_mapped = {}
                for i, _id in enumerate(id2doc.keys()):
                    texts_embeddings_mapped[_id] = texts_embeddings[i].tolist()
                with open(embeddings_path, "w") as fh:
                    json.dump(texts_embeddings_mapped, fh, indent=4)


if __name__ == "__main__":
    args = simple_parsing.parse(KNNIndex)
    main(args)
