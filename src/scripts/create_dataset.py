"""Create, save and publish the Faroese question answering dataset.

Usage:
    python src/scripts/create_dataset.py <config_key>=<config_value> ...
"""

import hydra
from foqa import build_dataset
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Create, save and publish the Faroese question answering dataset.

    Args:
        config:
            The Hydra config for your project.
    """
    build_dataset(config=config)


if __name__ == "__main__":
    main()
