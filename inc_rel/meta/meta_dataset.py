import random
from collections import defaultdict
from typing import Dict, List

from torch.utils.data import Dataset


def make_train_query_annotations(annotations):
    topic_ids = list(annotations.keys())
    random.shuffle(topic_ids)
    n = len(topic_ids) // 2
    train_topic_ids, query_topic_ids = topic_ids[:n], topic_ids[-n:]
    train_annotations = {
        topic_id: annotations[topic_id] for topic_id in train_topic_ids
    }
    query_annotations = {
        topic_id: annotations[topic_id] for topic_id in query_topic_ids
    }
    return train_annotations, query_annotations


class MetaDataset(Dataset):
    def __init__(self, annotations: Dict[str, List[Dict]]):
        self.annotations = annotations
        self.idx2topic_id = {i: topic_id for i, topic_id in enumerate(annotations)}
        self.data = []
        for topic_id, topic_annotations in self.annotations.items():
            documents, doc_ids, labels = [], [], []
            for annot in topic_annotations:
                query = annot["query"]
                documents.append(annot["document"])
                doc_ids.append(annot["doc_id"])
                labels.append(annot["label"])

            self.data.append(
                {
                    "topic_id": [topic_id] * len(documents),
                    "query": [query] * len(documents),
                    "documents": documents,
                    "doc_ids": doc_ids,
                    "labels": labels,
                }
            )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
