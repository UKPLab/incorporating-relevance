import os
import shutil
import tarfile
from urllib.parse import urlparse

import ir_datasets
import requests
from requests.auth import HTTPBasicAuth
from settings import (
    Settings,
    TRECCovidDatasetSettings,
    TRECNewsDatasetSettings,
    TRECRobustDatasetSettings,
    WebisTouche2020DatasetSettings,
)
from tqdm.auto import tqdm

UNSET_USERNAME = "<NONE>"


def download_file(
    url: str,
    file_path: str,
    username: str = None,
    password: str = None,
    pbar_desc: str = "",
    response_attr: str = "raw",
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

        if response_attr == "raw":
            total_length = int(response.headers.get("Content-Length", 0))
            with tqdm.wrapattr(
                getattr(response, response_attr),
                "read",
                total=total_length,
                ncols=100,
                desc=pbar_desc,
            ) as pbar:
                with open(file_path, "wb") as file:
                    shutil.copyfileobj(pbar, file)
        elif response_attr == "content":
            with open(file_path, "wb") as file:
                file.write(response.content)
        else:
            raise ValueError(f"Invalid response_attr: {response_attr}")

    else:
        print(f"File already exists: {file_path}")

    return file_path


def untar_gzip(file_path: str, output_dir: str) -> None:
    """Untar a gzip file."""
    with tarfile.open(file_path, "r:gz") as tar:
        for name in tar.getnames():
            if not os.path.exists(os.path.join(output_dir, name)):
                tar.extract(name, path=output_dir)


def main(env_file: str = ".env") -> None:
    # Setup data dir
    settings = Settings(_env_file=env_file)
    os.makedirs(settings.raw_path, exist_ok=True)

    # Download TREC-COVID
    trec_covid_settings = TRECCovidDatasetSettings()
    os.makedirs(trec_covid_settings.raw_path, exist_ok=True)
    out_file = download_file(
        trec_covid_settings.corpus_url,
        trec_covid_settings.raw_path,
        pbar_desc="trec-covid/corpus",
    )
    untar_gzip(out_file, trec_covid_settings.raw_path)
    download_file(
        trec_covid_settings.topics_url,
        trec_covid_settings.raw_path,
        response_attr="content",
        pbar_desc="trec-covid/topcis",
    )
    download_file(
        trec_covid_settings.qrels_url,
        trec_covid_settings.raw_path,
        response_attr="content",
        pbar_desc="trec-covid/qrels",
    )

    # Download Webis Touche 2020
    webis_touche_2020_settings = WebisTouche2020DatasetSettings(_env_file=env_file)
    ir_datasets.load(webis_touche_2020_settings.ir_datasets_name)

    # Download TREC-Robust
    trec_robust_settings = TRECRobustDatasetSettings(_env_file=env_file)
    if trec_robust_settings.username == UNSET_USERNAME:
        print(
            "Skipping TREC-Robust download. Get credentials from TREC for this dataset."
        )
    else:
        os.makedirs(trec_robust_settings.raw_path, exist_ok=True)
        out_file = download_file(
            trec_robust_settings.corpus_disk4_url,
            trec_robust_settings.raw_path,
            trec_robust_settings.username,
            trec_robust_settings.password.get_secret_value(),
            pbar_desc="trec-robust/corpus-disk4",
        )
        untar_gzip(out_file, trec_robust_settings.raw_path)

        out_file = download_file(
            trec_robust_settings.corpus_disk5_url,
            trec_robust_settings.raw_path,
            trec_robust_settings.username,
            trec_robust_settings.password.get_secret_value(),
            pbar_desc="trec-robust/corpus-disk5",
        )
        untar_gzip(out_file, trec_robust_settings.raw_path)

        # create symlinks to disk4 and disk5 in ir_datasets
        ir_datasets_robust_home_path = os.path.join(
            settings.ir_datasets_home, "trec-robust04", "trec45"
        )
        os.makedirs(ir_datasets_robust_home_path, exist_ok=True)
        for disk, robust_dataset in [
            ("TREC-Disk-4", "FR94"),
            ("TREC-Disk-4", "FT"),
            ("TREC-Disk-5", "FBIS"),
            ("TREC-Disk-5", "LATIMES"),
        ]:
            os.symlink(
                os.path.join(trec_robust_settings.raw_path, disk, robust_dataset),
                os.path.join(ir_datasets_robust_home_path, robust_dataset),
            )

    # Download TREC-News
    trec_news_settings = TRECNewsDatasetSettings(_env_file=env_file)
    if trec_news_settings.username == UNSET_USERNAME:
        print(
            "Skipping TREC-News download. Get credentials from TREC for this dataset."
        )
    else:
        os.makedirs(trec_news_settings.raw_path, exist_ok=True)
        out_file = download_file(
            trec_news_settings.corpus_url,
            trec_news_settings.raw_path,
            trec_news_settings.username,
            trec_news_settings.password.get_secret_value(),
            pbar_desc="trec-news/corpus",
        )
        untar_gzip(out_file, trec_news_settings.raw_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-file", type=str, default=".env")
    args = parser.parse_args()
    main(env_file=args.env_file)
