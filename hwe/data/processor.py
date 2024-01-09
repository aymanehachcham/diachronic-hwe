import json
import logging
import os
import re
from collections import Counter
from typing import List, Optional, Union

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import RegexpTokenizer

from .extraction import NewsPaperExtractorXml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Class that takes the extracted raw documents and analyses the word frequencies,
    and other statistics relevant for further analysis.
    """

    _openai_token_limit: int = 150_000

    def __init__(self, file_path: Union[str, os.PathLike]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        self.file_path = file_path
        self.file_name = os.path.basename(file_path).split(".")[0]
        self.docs = []
        # Detect if the file path comes from the post_process dir:
        if "postprocess_data" not in file_path:
            logging.warning(
                f"File path {file_path} does not come from the postprocess_data directory."
                f"The file will be processed first."
            )
            NewsPaperExtractorXml.from_xml(
                file_path=file_path,
            )
            file_name = self.file_name.replace(".xml", ".json")
            self.file_path = os.path.join(os.getenv("COMPILED_DOCS_PATH"), file_name)

    @staticmethod
    def __word_frequency(text: str, top_freq: int = 20) -> List[str]:
        # Remove stopwords
        stopwords_set = set(stopwords.words("english"))
        tokenizer = RegexpTokenizer(r"\b\w{3,}\b")

        # Tokenize the text
        tokens = tokenizer.tokenize(text.lower())
        word_counts = Counter(tokens)
        for stopword in stopwords_set:
            del word_counts[stopword]

        # Get words with frequency higher equal top_freq
        return [word for word, count in word_counts.items() if count == top_freq]

    def __compile_docs(self, save: bool = False) -> Optional[str]:
        """
        Compile all full texts into one big chunk.
        """
        pattern = r"(?<=\.)\s+$"

        try:
            with open(self.file_path) as f:
                articles = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(e)
            raise ValueError(f"Error: {self.file_path} is not a valid json file.") from e

        self.docs = [art["fulltext"][: self._openai_token_limit] for art in articles]

        txt = ".".join(self.docs)
        txt = re.sub(pattern, "", txt, flags=re.MULTILINE)
        if save:
            path = str(self.file_path).rsplit("/", maxsplit=1)[-1].replace(".json", ".txt")
            with open(os.path.join(os.getenv("COMPILED_DOCS_PATH"), path), "w") as f:
                f.write(txt)
            return None

        return txt

    def extract_hyponyms(self, word: str) -> List[str]:
        """
        Extract hyponyms for a given word.
        """
        synsets = wn.synsets(word)
        hyponyms = []
        for synset in synsets:
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    hyponyms += [lemma.name()]

        return hyponyms

    def retrieve_docs(self, query: str) -> Optional[List[str]]:
        """
        Retrieve documents that contain a given word.
        """
        self.__compile_docs(save=True)
