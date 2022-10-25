import os
import shutil
import tarfile
from urllib.parse import urlparse

import ir_datasets
import requests
from requests.auth import HTTPBasicAuth
from tqdm.auto import tqdm

from settings import (
    Settings,
    TRECCovidDatasetSettings,
    TRECNewsDatasetSettings,
    TRECRobustDatasetSettings,
    WebisTouche2020DatasetSettings,
)

UNSET_USERNAME = "<NONE>"


def download_file(
    url: str,
    file_path: str,
    username: str = None,
    password: str = None,
    pbar_desc: str = "",
) -> None:
    """Download a file from a URL to a file path."""
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, os.path.basename(urlparse(url).path))

    if not os.path.exists(file_path):
        request_kwargs = {}
        if all([username, password]):
            request_kwargs["auth"] = HTTPBasicAuth(username, password)

        response = requests.get(url, stream=True, **request_kwargs)
        response.raise_for_status()

        total_length = int(response.headers.get("Content-Length", 0))
        with tqdm.wrapattr(
            response.raw,
            "read",
            total=total_length,
            ncols=100,
            desc=pbar_desc,
        ) as pbar:
            with open(file_path, "wb") as file:
                shutil.copyfileobj(pbar, file)
    else:
        print(f"File already exists: {file_path}")

    return file_path


def stip_extension(p: str) -> str:
    """Strip extension from a file path. Even with double extensions like .tar.gz."""
    p = os.path.basename(p)
    p = os.path.splitext(p)
    print(p, isinstance(p, tuple))
    while isinstance(p, tuple) and len(p[1]):
        p = p[0]
        p = os.path.splitext(p)
    return p[0]


def untar_gzip(file_path: str, output_dir: str) -> None:
    """Untar a gzip file."""
    p, f = os.path.dirname(file_path), os.path.basename(file_path)
    f = stip_extension(f)
    if not os.path.exists(os.path.join(output_dir, f)):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(output_dir)
    else:
        print(f"File already extracted: {os.path.join(output_dir, f)}")


# Setup data dir
settings = Settings()
os.makedirs(settings.data_path, exist_ok=True)

# Download TREC-COVID
trec_covid_settings = TRECCovidDatasetSettings()
trec_covid_data_path = os.path.join(settings.data_path, trec_covid_settings.data_path)
os.makedirs(trec_covid_data_path, exist_ok=True)
out_file = download_file(
    trec_covid_settings.corpus_url, trec_covid_data_path, pbar_desc="trec-covid/corpus"
)
untar_gzip(out_file, trec_covid_data_path)
download_file(
    trec_covid_settings.topics_url, trec_covid_data_path, pbar_desc="trec-covid/topcis"
)
download_file(
    trec_covid_settings.qrels_url, trec_covid_data_path, pbar_desc="trec-covid/qrels"
)

# Download Webis Touche 2020
webis_touche_2020_settings = WebisTouche2020DatasetSettings()
ir_datasets.load(webis_touche_2020_settings.ir_datasets_name)

# Download TREC-Robust
trec_robust_settings = TRECRobustDatasetSettings()
if trec_robust_settings.username == UNSET_USERNAME:
    print("Skipping TREC-Robust download. Get credentials from TREC for this dataset.")
else:
    trec_robust_data_path = os.path.join(
        settings.data_path, trec_robust_settings.data_path
    )
    os.makedirs(trec_robust_data_path, exist_ok=True)
    out_file = download_file(
        trec_robust_settings.corpus_disk4_url,
        trec_robust_data_path,
        trec_robust_settings.username,
        trec_robust_settings.password.get_secret_value(),
        pbar_desc="trec-robust/corpus-disk4",
    )
    untar_gzip(out_file, trec_robust_data_path)

    out_file = download_file(
        trec_robust_settings.corpus_disk5_url,
        trec_robust_data_path,
        trec_robust_settings.username,
        trec_robust_settings.password.get_secret_value(),
        pbar_desc="trec-robust/corpus-disk5",
    )
    untar_gzip(out_file, trec_robust_data_path)

# Download TREC-News
trec_news_settings = TRECNewsDatasetSettings()
if trec_news_settings.username == UNSET_USERNAME:
    print("Skipping TREC-News download. Get credentials from TREC for this dataset.")
else:
    trec_news_data_path = os.path.join(settings.data_path, trec_news_settings.data_path)
    os.makedirs(trec_news_data_path, exist_ok=True)
    out_file = download_file(
        trec_news_settings.corpus_url,
        trec_news_data_path,
        trec_news_settings.username,
        trec_news_settings.password.get_secret_value(),
        pbar_desc="trec-news/corpus",
    )
    untar_gzip(out_file, trec_news_data_path)
