import json
import logging
import os
import re
from collections import Counter
from typing import List, Optional, Union

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import RegexpTokenizer

from ..rag.manager import RAGManager
from .extraction import NewsPaperExtractorXml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Class that takes the extracted raw documents and analyses the word frequencies,
    and other statistics relevant for further analysis.
    """

    _openai_token_limit: int = 150_000

    def __init__(self, json_file_path: Union[str, os.PathLike]):
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File {json_file_path} does not exist.")

        self.file_path = json_file_path
        self.file_name = os.path.basename(json_file_path).split(".")[0]
        self.docs = []
        # Detect if the file path comes from the post_process dir:
        if "postprocess_data" not in json_file_path:
            file_name = self.file_name + ".json"
            self.file_path = os.path.join(os.getenv("COMPILED_DOCS_PATH"), file_name)
            if not os.path.exists(self.file_path):
                logging.warning(
                    f"File path {json_file_path} does not come from the postprocess_data directory."
                    f"The file will be processed first."
                )
                NewsPaperExtractorXml.from_xml(
                    xml_file_path=self.file_path,
                )

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

    def __compile_docs(self, num_articles: int = 100, save: bool = False) -> Optional[str]:
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

        self.docs = [art["fulltext"] for art in articles[:num_articles]]

        txt = ".".join(self.docs)
        txt = re.sub(pattern, "", txt, flags=re.MULTILINE)
        if save:
            txt_path = str(self.file_path).rsplit("/", maxsplit=1)[-1].replace(".json", ".txt")
            with open(os.path.join(os.getenv("COMPILED_DOCS_PATH"), txt_path), "w") as f:
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

    def retrieve_context(
        self,
        query: str,
    ) -> Optional[List[str]]:
        """
        Retrieve documents that contain a given word.
        """
        self.__compile_docs(save=True)
        context = []
        rag = RAGManager.from_file(file_path=os.path.join(os.getenv("COMPILED_DOCS_PATH"), self.file_name + ".txt"))
        docs = rag.retrieve_docs(query=query)
        logging.info(f"Retrieved documents: {len(docs)}.")
        for doc in docs:
            context += [doc.page_content]
        return context
