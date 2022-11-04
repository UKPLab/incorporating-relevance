import os
from typing import List, Union

from pydantic import BaseSettings, Field, SecretStr


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_"

    ir_datasets_home: str = Field("~/.ir_datasets", env="IR_DATASETS_HOME")

    base_path: str
    raw_dir: str
    data_dir: str

    @property
    def raw_path(self) -> str:
        return os.path.join(self.base_path, self.raw_dir)

    @property
    def data_path(self) -> str:
        return os.path.join(self.base_path, self.data_dir)


class DatasetSettings(BaseSettings):

    split_seeds: List[int] = Field(
        [0, 1, 2], description="Random seeds for splitting the dataset."
    )
    split_sizes: List[float] = Field(
        [0.6, 0.2, 0.2], description="Train, dev, test split sizes."
    )
    num_samples: List[int] = Field(
        [2, 4, 8], description="Amount of relevance feedback per topic (k)."
    )
    bm25_size: int = Field(
        1000,
        description="Number of documents to retrieve in the first stage with BM25.",
    )
    remove_duplicates: bool = Field(False, description="Remove duplicate documents.")

    @property
    def corpus_path(self) -> str:
        # datasets managed by ir_datasets do not have a corpus_path
        return "not implemented"

    @property
    def topics_path(self) -> str:
        return "not implemented"

    @property
    def qrels_path(self) -> str:
        return "not implemented"

    @property
    def enrich_bm25_path(self) -> Union[str, None]:
        return None


class TRECCovidDatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_COVID_"

    name: str = "trec-covid"
    raw_path: str
    data_path: str
    corpus_url: str
    topics_url: str
    qrels_url: str

    remove_duplicates = True

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.raw_path, "2020-07-16", "metadata.csv")

    @property
    def topics_path(self) -> str:
        return os.path.join(self.raw_path, "topics-rnd5.xml")

    @property
    def qrels_path(self) -> str:
        return os.path.join(self.raw_path, "qrels-covid_d5_j0.5-5.txt")


class WebisTouche2020DatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TOUCHE_"

    name: str = "touche"
    ir_datasets_name: str
    data_path: str

    @property
    def enrich_bm25_path(self) -> str:
        return os.path.join(
            os.path.join(self.data_path, "1000", "full_bm25_results.json")
        )


class TRECRobustDatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_ROBUST_"

    name: str = "robust04"
    raw_path: str
    data_path: str
    username: str
    password: SecretStr
    corpus_disk4_url: str
    corpus_disk5_url: str


class TRECNewsDatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_NEWS_"

    name: str = "trec-news"
    raw_path: str
    data_path: str
    username: str
    password: SecretStr
    corpus_url: str

    @property
    def corpus_path(self) -> str:
        return os.path.join(
            self.raw_path,
            "WashingtonPost.v2",
            "data",
            "TREC_Washington_Post_collection.v2.jl",
        )


dataset_settings_cls = {
    "trec-covid": TRECCovidDatasetSettings,
    "robust04": TRECRobustDatasetSettings,
    "trec-news": TRECNewsDatasetSettings,
    "touche": WebisTouche2020DatasetSettings,
}
