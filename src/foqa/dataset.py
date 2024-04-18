"""Build, save and publish the Faroese question answering dataset."""

import json
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

    records_path = Path(config.dirs.data) / config.dirs.raw / "records.jsonl"
    if records_path.exists():
        with records_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = list()

    for sample in tqdm(dataset, desc=f"Generating samples with {config.model}"):
        sample_exists = any(record["id"] == sample["url"] for record in records)
        if sample_exists:
            continue

        sample_is_too_short = len(sample["text"]) < config.min_article_length
        if sample_is_too_short:
            continue

        try:
            generated_samples = get_json_generation(
                article=sample["text"], config=config
            )
        except Exception as e:
            logger.info(
                f"Failed to generate samples for {sample['url']} with error message "
                f"{str(e)}. Skipping."
            )
            continue

        with records_path.open("a") as f:
            for generated_sample in generated_samples:
                record = dict(
                    id=sample["url"],
                    title=sample["title"],
                    context=sample["text"],
                    question=generated_sample["question"],
                    answers=dict(
                        text=[generated_sample["answer"]],
                        answer_start=sample["text"].find(generated_sample["answer"]),
                    ),
                )
                records.append(record)
                f.write(json.dumps(record) + "\n")

    logger.info("Converting the records to a Hugging Face dataset...")
    df = pd.DataFrame.from_records(records)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    logger.info("Saving the dataset to disk...")
    dataset_path = Path(config.dirs.data) / config.dirs.processed / "foqa"
    dataset.save_to_disk(dataset_path)
    logger.info(f"Dataset saved to {dataset_path}.")

    if config.push_to_hub:
        logger.info("Pushing the dataset to the Hugging Face Hub...")
        dataset.push_to_hub(config.hub_id)

    logger.info("All done!")
