import logging
import os
from time import perf_counter
from typing import Dict

import toml
from dotenv import load_dotenv
from openai import OpenAI

from ..utils import find_closest

logger = logging.getLogger(__name__)
load_dotenv()


class GptSimilarityFinder:
    """
    class the handler open AI gpt retriever with embeddings to look up
    closer sense for each target word.
    """

    def __init__(self):
        # Configure the gpt 4 instance:
        self.config = toml.load(find_closest("diachronic_hwe/ai_finder/config.toml"))
        self.model = self.config["openai"]["model"]
        self.prompt = os.getenv("GPT_PROMPT")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_prompt(
        self,
        target_word: str,
        text: str,
        senses: Dict,
    ) -> str:
        """
        Function to get the prompt for the GPT-3 model.
        :param target_word: str
        :param text: str
        :param senses: Dict
        """
        if os.path.isfile(self.prompt):
            # Read the prompt from the file
            # that allows changing the prompt without restarting the server
            # use it only for development
            with open(self.prompt) as f:
                prompt = f.read()
        else:
            prompt = self.prompt
        return prompt.format(text=text, word=target_word, senses=senses)

    def get_sense_from_target_word(
        self,
        target_word: str,
        text: str,
        senses: Dict,
    ):
        """
        Function to get the sense of the target word based on the context.
        :param target_word: str
        :param text: str
        :param senses: Dict
        """
        prompt = self._get_prompt(target_word, text, senses)
        # Call OpenAI's API to create a chat completion using the GPT-3 model
        logger.info("Calling OpenAI API....")
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],  # The 'user' role is assigned to the prompt
            stop=None,  # There are no specific stop sequences
            temperature=self.config["openai"]["temperature"],
        )
        logger.info(f"OpenAI API call took {perf_counter() - start} seconds.")
        response_text = response.choices[0].message.content.strip()
        return response_text
