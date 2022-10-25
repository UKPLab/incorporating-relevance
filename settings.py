from typing import List, Union
from pydantic import BaseSettings, Field, SecretStr


class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_"

    data_path: str


class TRECCovidDatasetSettings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_COVID_"

    data_path: str
    corpus_url: str
    topics_url: str
    qrels_url: str


class WebisTouche2020DatasetSettings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_WEBIS_TOUCHE_2020_"

    ir_datasets_name: str


class TRECRobustDatasetSettings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_ROBUST_"

    data_path: str
    username: str
    password: SecretStr
    corpus_disk4_url: str
    corpus_disk5_url: str


class TRECNewsDatasetSettings(BaseSettings):
    class Config:
        env_file = ".env"
        env_prefix = "INC_REL_TREC_NEWS_"

    data_path: str
    username: str
    password: SecretStr
    corpus_url: str
