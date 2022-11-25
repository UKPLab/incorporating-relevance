import json
from collections import defaultdict

import simple_parsing
from args import RankFusion


def rank_fusion(rank: int, c: int = 60):
    return 1 / (c + rank)


def main(args):
    # load result files
    results = []
    for file in args.result_files:
        with open(file) as fh:
            results.append(json.load(fh))

    # compute rank fusion scores
    query_to_doc_to_scores = defaultdict(lambda: defaultdict(float))
    for result in results:
        for query_id, doc_to_score in result.items():
            for rank, doc_id in enumerate(doc_to_score.keys(), start=1):
                query_to_doc_to_scores[query_id][doc_id] += rank_fusion(rank)


if __name__ == "__main__":
    args = simple_parsing.parse(RankFusion)
    main(args)
