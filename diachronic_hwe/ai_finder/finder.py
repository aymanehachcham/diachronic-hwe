import logging
import os
from time import perf_counter
from typing import Dict, Union, List

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
    _format = """
    target_word.n.01 - sense.n.01
    sense.n.01 - sub-sense.n.01
    context_word.n.01 - sub-sense.n.01
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
        doc: str,
        target: str,
        senses: Dict,
        sub_senses: List,
        output_format: str
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
        return prompt.format(
            context=doc,
            target=target,
            senses=senses,
            sub_senses=sub_senses,
            format=output_format
        )

    def call_gpt(
        self,
        target: str,
        doc: str,
        senses: Dict,
        sub_senses: List,
    ):
        """
        Function to get the sense of the target word based on the context.
        :param target: str
        :param doc: str
        :param senses: List
        :param sub_senses: List
        """
        prompt = self._get_prompt(
            doc=doc,
            target=target,
            senses=senses,
            sub_senses=sub_senses,
            output_format=f'{self._format}'
        )
        # Call OpenAI's API to create a chat completion using the GPT-3 model
        logging.info("Calling OpenAI API....")
        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],  # The 'user' role is assigned to the prompt
            stop=None,  # There are no specific stop sequences
            temperature=self.config["openai"]["temperature"],
            max_tokens=self.config["openai"]["max_tokens"],
        )
        logging.info(f"OpenAI API call took {perf_counter() - start} seconds.")
        response_text = response.choices[0].message.content.strip()
        # convert the response text to a list of strings
        response_list = response_text.split("\n")
        return response_list
