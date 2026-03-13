import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/ai4i2020.csv"


def download_dataset(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("Dataset already exists at %s", output_path)
        return

    for url in [DATASET_URL, FALLBACK_URL]:
        try:
            logger.info("Downloading from %s", url)
            urllib.request.urlretrieve(url, output_path)
            logger.info("Saved to %s", output_path)
            return
        except Exception as e:
            logger.warning("Failed to download from %s: %s", url, e)

    raise RuntimeError("Failed to download dataset from all sources")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    download_dataset(Path("data/raw/ai4i2020.csv"))
