"""Interact with OpenAI models."""

import json
import logging
import os

from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat

load_dotenv()


logger = logging.getLogger(__name__)


def get_json_generation(article: str, config: DictConfig) -> list[dict[str, str]]:
    """Generate a list of (context, question, answer) dictionaries from an article.

    Args:
        article:
            The article to generate questions from.
        config:
            The Hydra config for your project.

    Returns:
        A list of dictionaries containing the generated text.

    Raises:
        ValueError:
            If the OPENAI_API_KEY environment variable is not set.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=openai_api_key, max_retries=60)

    model_output = client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(
                role="system", content=config.system_prompt
            ),
            ChatCompletionUserMessageParam(
                role="user", content=config.prompt.format(article=article)
            ),
        ],
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        response_format=ResponseFormat(type="json_object"),
    )
    generation_output = model_output.choices[0].message.content
    assert isinstance(generation_output, str)
    json_obj = json.loads(generation_output)
    if isinstance(json_obj, dict):
        json_obj = list(json_obj.values())[0]
    return json_obj
