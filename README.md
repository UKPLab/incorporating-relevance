# Incorporating Relevance Feedback for Information-Seeking Retrieval using Few-Shot Document Re-Ranking
<p align="center">
<a href="https://aclanthology.org/2022.emnlp-main.614/">
    <img alt="Paper Badge" src="https://img.shields.io/badge/Paper-b31b1b.svg">
</a>
<a href="https://public.ukp.informatik.tu-darmstadt.de/baumgaertner/emnlp-2022/inc-rel-slides-emnlp-2022.pdf">
    <img alt="Slides Badge" src="https://img.shields.io/badge/Slides-b31b1b.svg">
</a>
<a href="https://public.ukp.informatik.tu-darmstadt.de/baumgaertner/emnlp-2022/inc-rel-poster-emnlp-2022.pdf">
    <img alt="Slides Badge" src="https://img.shields.io/badge/Poster-b31b1b.svg">
</a>
</p>

This repository contains the code for our EMNLP 2022 paper [_Incorporating Relevance Feedback for Information-Seeking Retrieval using Few-Shot Document Re-Ranking_](https://aclanthology.org/2022.emnlp-main.614).

## Overview
<p align="center">
<img src="https://user-images.githubusercontent.com/11020443/214001736-5aa508ac-56cb-4c89-8bf9-ba879693f868.svg">
</p>

> Pairing a lexical retriever with a neural re-ranking model has set state-of-the-art performance on large-scale information retrieval datasets. This pipeline covers scenarios like question answering or navigational queries, however, for information-seeking scenarios, users often provide information on whether a document is relevant to their query in form of clicks or explicit feedback. Therefore, in this work, we explore how relevance feedback can be directly integrated into neural re-ranking models by adopting few-shot and parameter-efficient learning techniques. Specifically, we introduce a kNN approach that re-ranks documents based on their similarity with the query and the documents the user considers relevant. Further, we explore Cross-Encoder models that we pre-train using meta-learning and subsequently fine-tune for each query, training only on the feedback documents. To evaluate our different integration strategies, we transform four existing information retrieval datasets into the relevance feedback scenario. Extensive experiments demonstrate that integrating relevance feedback directly in neural re-ranking models improves their performance, and fusing lexical ranking with our best performing neural re-ranker outperforms all other methods by 5.2 nDCG@20.

## Requirements
- python 3.10+
- if docker is availbale, elasticsearch will be run in a container, else the bare metal version will be started.
## Setup
### Virtual Environment and Dependencies
1. Setup your python virtual environment, for example:
    ```shell
    python -m venv .venv
    ```
2. Install dependencies
    ```shell
    make install
    ```
    If you want to contribute to the repository, please also install the dev dependencies:
    ```shell
    make install-dev
    ```
### Datasets and Preprocessing
The paths where to store raw, preprocessed and experiments data can be modified in the `.env` file. Once these are set, run the following commands:
1. Download Datasets
    ```shell
    make download
    ```
2. Index corpora in elasticsearch.
    ```
    make index
    ```


## Experiments
The following commands run the experiments using the default parameters. For the full set of available arguments, see [inc_rel/args.py](inc_rel/args.py).
### 2nd Stage Retrieval and Query Expansion
Creating the index the last step in the [Setup](#setup) also runs the experiments over the second stage retrieval. By default, the query is expanded with `[4, 8, 16, 32, 64]` terms using ElasticSearchs [MoreLikeThis](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-mlt-query.html) query.

### kNN
```
make knn args="--dataset <dataset>"
```

### Zero-Shot
```
make knn args="--dataset <dataset> --num_samples <num_samples>"
```

### Query Fine-Tuning
```
make args="--dataset <dataset> --num_samples <num_samples>"
```

### Meta-Learning + Query Fine-Tuning
```
make args="--dataset <dataset> --num_samples <num_samples>"
```

### Rank-Fusion
```
make args="--dataset <dataset> --num_samples <num_samples> --result_files <path to first result file> <path to second result file> 
```

## Contact
This project is maintained by [Tim Baumgärtner](https://github.com/timbmg).
- [Mail](mailto:baumgaertner.t@gmail.com) 
- [Twitter](https://twitter.com/timbmg) 
- [UKP Lab](http://www.ukp.tu-darmstadt.de/)
- [TU Darmstadt](http://www.tu-darmstadt.de/)

## Citation
If you find this work useful, please considering citing the following [paper](https://aclanthology.org/2022.emnlp-main.614):
```bibtex
@inproceedings{baumgartner-etal-2022-incorporating,
    title = "Incorporating Relevance Feedback for Information-Seeking Retrieval using Few-Shot Document Re-Ranking",
    author = {Baumg{\"a}rtner, Tim  and
      Ribeiro, Leonardo F. R.  and
      Reimers, Nils  and
      Gurevych, Iryna},
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.614",
    pages = "8988--9005",
}
```

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
