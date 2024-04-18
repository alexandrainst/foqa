"""Build, save and publish the Faroese question answering dataset."""

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

from foqa.openai import get_json_generation

logger = logging.getLogger(__name__)


def build_dataset(config: DictConfig) -> None:
    """Build the Faroese question answering dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    logger.info("Loading the Faroese Wikipedia dataset...")
    dataset = load_dataset("alexandrainst/scandi-wiki", "fo", split="train")

    records: list[dict[str, str]] = list()
    for sample in tqdm(dataset, desc=f"Generating samples with {config.model}"):
        generated_samples = get_json_generation(article=sample["text"], config=config)
        for generated_sample in generated_samples:
            generated_sample["url"] = sample["url"]
        records.extend(generated_samples)

    logger.info("Converting the records to a Hugging Face dataset...")
    df = pd.DataFrame.from_records(records)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    logger.info("Saving the dataset to disk...")
    dataset_path = Path(config.dirs.data) / config.dirs.raw / "foqa"
    dataset.save_to_disk(dataset_path)
    logger.info(f"Dataset saved to {dataset_path}.")

    if config.push_to_hub:
        logger.info("Pushing the dataset to the Hugging Face Hub...")
        dataset.push_to_hub(config.hub_id)

    logger.info("All done!")
