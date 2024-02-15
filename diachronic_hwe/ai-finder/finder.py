import os

import openai
import tom

class GptFinder:
    """
    class the handler open AI gpt retriever with embeddings to look up
    closer sense for each target word.
    """
    def __init__(self):
        # Configure the gpt 4 instance:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.config =




