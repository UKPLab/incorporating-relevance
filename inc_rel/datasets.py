import csv
import itertools
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Union

import ir_datasets
import utils
from tqdm.auto import tqdm

import utils


class Dataset:
    def __init__(
        self,
        corpus_path: Path,
        topics_path: Path,
        qrels_path: Path,
        load_corpus: bool,
        remove_topics_with_few_annotations: bool = True,
        remove_topics_with_few_negatives: bool = True,
    ):
        if load_corpus:
            self.corpus = self.read_corpus(corpus_path)
        self.topics = self.read_topics(topics_path)
        self.qrels = self.read_qrels(qrels_path)
        if remove_topics_with_few_annotations:
            self.remove_topics_with_few_annotations(remove_topics_with_few_negatives)

    def remove_topic(self, topic_id):
        self.qrels.pop(topic_id, None)
        self.topics.pop(topic_id, None)

    def read_corpus(self, corpus_path: Path) -> Union[Dict, Callable]:
        raise NotImplementedError()

    def read_topics(self, topics_path: Path) -> Dict[str, str]:
        raise NotImplementedError()

    def read_qrels(self, qrels_path: Path) -> Dict[str, str]:
        raise NotImplementedError()

    def remove_topics_with_few_annotations(
        self, remove_topics_with_few_negatives: bool
    ):

        topics_to_remove = []
        for topic_id, doc_id2relevance in self.qrels.items():
            num_relevant = len(
                list(filter(lambda r: r >= 1, doc_id2relevance.values()))
            )
            if remove_topics_with_few_negatives:
                num_non_relevant = len(
                    list(filter(lambda r: r <= 0, doc_id2relevance.values()))
                )
            else:
                num_non_relevant = self.min_annotations
            if self.min_annotations > min(num_relevant, num_non_relevant):
                topics_to_remove.append((topic_id, num_relevant, num_non_relevant))

        for topic_id in self.topics.keys():
            if topic_id not in self.qrels:
                topics_to_remove.append((topic_id, 0, 0))

        for topic_id, num_relevant, num_non_relevant in topics_to_remove:
            print(
                f"Removing {topic_id=} due to too few annotations "
                f"({num_relevant=}, {num_non_relevant=})."
            )
            self.remove_topic(topic_id)

        if not len(self.qrels) == len(self.topics):
            raise RuntimeError(
                "Expected to have same amount of qrels and topics, "
                f"but got {len(self.qrels)=} {len(self.topics)=}"
            )
        print(f"Remaining Topics={len(self.qrels)}")

    def remove_annotations_from_qrels(self, annotations: Dict[str, List[Dict]]):
        for topic_id, annotation in annotations.items():
            topic_id = annotation[0]["topic_id"]
            for a in annotation:
                doc_id = a["doc_id"]
                r = self.qrels[topic_id].pop(doc_id, None)
                if r is None:
                    print(
                        f"Could not find document to remove from qrels. {topic_id=} {doc_id=} label={a['label']}"
                    )


class TrecCovid(Dataset):
    def __init__(self, *args, **kwargs):
        self.min_annotations = 32
        self.min_relevant_relevancy = 2
        super().__init__(*args, **kwargs)

    def read_corpus(self, corpus_path: Path) -> Dict[str, str]:

        fields = ["title", "abstract"]

        corpus = {}
        with open(corpus_path, encoding="utf8") as fh:
            reader = csv.DictReader(fh, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                cord_uid = row["cord_uid"]
                row["title"] = row["title"].strip()
                if row["title"] and row["title"][-1] not in "!?.":
                    row["title"] += "."

                text = " ".join([row[field] for field in fields]).strip()
                if cord_uid and text:
                    corpus[cord_uid] = text

        return corpus

    def read_topics(self, topics_path: Path) -> Dict[str, str]:

        query_type = "question"

        queries = {}
        root = ET.parse(topics_path).getroot()
        for topic in root:
            for query in topic:
                if query.tag == query_type:
                    queries[topic.attrib["number"]] = query.text

        return queries

    def read_qrels(self, qrels_path: Path) -> Dict[str, Dict[str, int]]:
        qrels = defaultdict(dict)
        with open(qrels_path, "r") as fh:
            for line in fh:
                query_id, _, doc_id, relevance = line.strip().strip("\n").split(" ")
                qrels[str(query_id)][doc_id] = int(relevance)

        return qrels


class TrecNews(Dataset):
    def __init__(self, *args, **kwargs):
        self.min_annotations = 32
        self.min_relevant_relevancy = 1
        super().__init__(*args, **kwargs)

    def read_topics(self, topics_path: Path) -> Dict[str, str]:

        topics = {}
        with open(topics_path) as fh:
            it = itertools.chain("<root>", fh, "</root>")
            root = ET.fromstringlist(it)

        docid2num = {}
        for topic in root.findall("top"):
            num = topic[0].text.split(":")[1].strip()
            docid = topic[1].text
            docid2num[docid] = num
        for doc in tqdm(
            self.corpus(doc_ids=list(docid2num.keys()), verbose=False),
            desc="extracting titles",
            total=len(docid2num),
            ncols=100,
        ):
            num = docid2num[doc["id"]]
            topics[num] = doc["title"]

        return topics

    def read_qrels(self, qrels_path: Path) -> Dict[str, Dict[str, int]]:
        qrels = defaultdict(dict)
        with open(qrels_path) as fh:
            for line in fh.readlines():
                qid, _, doc_id, relevance = line.split(" ")
                qrels[str(qid)].update({doc_id: int(relevance)})
        return qrels

    def read_corpus(self, corpus_path: Path) -> Callable:
        def corpus_iter(doc_ids: List[str] = None, verbose=True):
            if doc_ids is not None:
                doc_ids = list(set(doc_ids))
            with open(corpus_path) as fh:
                for line in tqdm(fh, total=595037, disable=not verbose, ncols=100):
                    doc = json.loads(line)
                    if doc_ids is None or doc["id"] in doc_ids:
                        full_text = []
                        title = ""
                        for content in doc.get("contents", []):
                            if content is None:
                                continue
                            if content["type"] == "title":
                                title = content["content"]
                            elif (
                                content.get("type", None) == "sanitized_html"
                                and content.get("subtype", None) == "paragraph"
                            ):
                                full_text.append(utils.strip_tags(content["content"]))
                        full_text = " ".join(full_text)

                        yield {"id": doc["id"], "title": title, "text": full_text}

                        if doc_ids is not None:
                            doc_ids.remove(doc["id"])
                            if len(doc_ids) == 0:
                                break

        return corpus_iter

    def read_topics(self, queries_file_path: Path) -> Dict[str, str]:
        queries = {}
        with open(queries_file_path) as fh:
            for line in fh:
                line = json.loads(line)
                queries[str(line["qid"])] = line["query"]
        return queries

    def read_qrels(self, qrels_path: Path) -> Dict[str, Dict[str, int]]:

        qrels = defaultdict(dict)
        with open(qrels_path, "r") as fh:
            for line in fh:
                query_id, _, doc_id, relevance = line.strip().strip("\n").split(" ")
                qrels[str(query_id)][doc_id] = int(relevance)

        return qrels


class Robust04(Dataset):
    def __init__(self, *args, **kwargs):
        self.min_annotations = 32
        self.min_relevant_relevancy = 1
        self.dataset = ir_datasets.load("trec-robust04")
        super().__init__(*args, **kwargs)

    def read_corpus(self, *args, **kwargs) -> Dict[str, str]:
        corpus = {}
        for doc in tqdm(
            self.dataset.docs_iter(), total=528155, desc="loading dataset", ncols=100
        ):
            corpus[doc.doc_id] = doc.text
        return corpus

    def read_topics(self, *args, **kwargs) -> Dict[str, str]:
        topics = {}
        for topic in self.dataset.queries_iter():
            topics[topic.query_id] = topic.description
        return topics

    def read_qrels(self, *args, **kwargs) -> Dict[str, Dict[str, int]]:
        qrels = defaultdict(dict)
        for qrel in self.dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        return qrels


class Touche(Dataset):
    def __init__(self, *args, **kwargs):
        self.min_annotations = 32
        self.min_relevant_relevancy = 3
        self.dataset = ir_datasets.load("beir/webis-touche2020")
        super().__init__(*args, **kwargs)

    def read_corpus(self, *args, **kwargs) -> Dict[str, str]:
        corpus = {}
        for doc in tqdm(
            self.dataset.docs_iter(), total=382545, desc="loading dataset", ncols=100
        ):
            corpus[doc.doc_id] = doc.text
        return corpus

    def read_topics(self, *args, **kwargs) -> Dict[str, str]:
        topics = {}
        for topic in self.dataset.queries_iter():
            topics[topic.query_id] = topic.text
        return topics

    def read_qrels(self, *args, **kwargs) -> Dict[str, Dict[str, int]]:
        qrels = defaultdict(dict)
        for qrel in self.dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        return qrels


datasets_cls = {
    "trec-covid": TrecCovid,
    "trec-news": TrecNews,
    "robust04": Robust04,
    "touche": Touche,
}
