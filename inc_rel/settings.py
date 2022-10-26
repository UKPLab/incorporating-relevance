import os
from typing import List
from pydantic import BaseSettings, Field, SecretStr


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_"

    data_path: str


class DatasetSettings(BaseSettings):

    base_path: str = Field(..., env="INC_REL_DATASET_BASE_PATH")

    seeds: List[int] = Field(
        [0, 1, 2], description="Random seeds for splitting the dataset."
    )
    split_sizes: List[float] = Field(
        [0.6, 0.2, 0.2], description="Train, dev, test split sizes."
    )
    num_samples: List[int] = Field(
        [2, 4, 8], description="Amount of relevance feedback per topic (k)."
    )
    bm25_size: int = Field(
        1000, description="Amount of documents to retrieve with BM25."
    )
    remove_duplicates: bool = Field(False, description="Remove duplicate documents.")
    enrich_bm25_path: str = Field(
        None, description="If provided, use negatives from BM25."
    )


class TRECCovidDatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_COVID_"

    name: str = "trec-covid"
    data_path: str
    corpus_url: str
    topics_url: str
    qrels_url: str

    remove_duplicates = True

    @property
    def corpus_path(self) -> str:
        return os.path.join(
            self.base_path, self.data_path, "2020-07-16", "metadata.csv"
        )

    @property
    def topics_path(self) -> str:
        return os.path.join(self.base_path, self.data_path, "topics-rnd5.xml")

    @property
    def qrels_path(self) -> str:
        return os.path.join(self.base_path, self.data_path, "qrels-covid_d5_j0.5-5.txt")


class WebisTouche2020DatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_WEBIS_TOUCHE_2020_"

    name: str = "touche"
    ir_datasets_name: str


class TRECRobustDatasetSettings(DatasetSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_ROBUST_"

    name: str = "robust"
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
    data_path: str
    username: str
    password: SecretStr
    corpus_url: str


dataset_settings_cls = {
    "trec-covid": TRECCovidDatasetSettings,
    "trec-robust": TRECRobustDatasetSettings,
    "trec-news": TRECNewsDatasetSettings,
    "webis-touche-2020": WebisTouche2020DatasetSettings,
}
