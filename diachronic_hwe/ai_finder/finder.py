import logging
import os
from time import perf_counter
from typing import Dict, Union

import toml
import json
from dotenv import load_dotenv
from openai import OpenAI

from ..utils import find_closest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


class GptSimilarityFinder:
    """
    class the handler open AI gpt retriever with embeddings to look up
    closer sense for each target word.
    """

    def __init__(self, prompt_path: str):
        # Configure the gpt 4 instance:
        self.config = toml.load(find_closest("diachronic_hwe/ai_finder/config.toml"))
        self.cache_file = find_closest("diachronic_hwe/ai_finder/cache.json")
        self.model = self.config["openai"]["model"]
        self.prompt = prompt_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache = {}

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def _get_prompt(
        self,
        target_word: str,
        text: str,
        senses: Union[Dict, str],
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

    def call_gpt(
        self,
        target_word: str,
        text: str,
        senses: Union[Dict, str],
    ):
        """
        Function to get the sense of the target word based on the context.
        :param target_word: str
        :param text: str
        :param senses: Dict
        """
        # Serialize the target_word, text, and senses to create a unique cache key
        self.load_cache()
        cache_key = f"{target_word}_{text}"
        # Check if the query is already in the cache
        if cache_key in self.cache:
            logging.info("Retrieving response from cache.")
            return self.cache[cache_key]

        prompt = self._get_prompt(target_word, text, senses)
        # Call OpenAI's API to create a chat completion using the GPT-3 model
        logging.info("Calling OpenAI API....")
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],  # The 'user' role is assigned to the prompt
            stop=None,  # There are no specific stop sequences
            temperature=self.config["openai"]["temperature"],
        )
        logging.info(f"OpenAI API call took {perf_counter() - start} seconds.")
        response_text = response.choices[0].message.content.strip()

        # Store the response in cache before returning
        self.cache[cache_key] = response_text
        self.save_cache()
        return response_text
