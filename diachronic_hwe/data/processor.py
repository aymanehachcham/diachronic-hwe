import json
import logging
import os
import re
from collections import Counter
from typing import List, Optional, Union

import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
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

        self.nlp = spacy.load("en_core_web_md")
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

    def __compile_docs(self, num_articles: int = 1000, save: bool = False) -> Optional[str]:
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

    def lookup_word(self, word: str) -> None:
        """
        Look up a word in the documents.
        """
        txt_path = str(self.file_path).rsplit("/", maxsplit=1)[-1].replace(".json", ".txt")
        file = os.path.join(os.getenv("COMPILED_DOCS_PATH"), txt_path)
        # if os.path.exists(file):
        #     logging.info(f"File {file} already exists.")
        #     return

        _ = self.__compile_docs()
        logging.info(f"Length of docs: {len(self.docs)}")
        matching_paragraphs = []
        target_lemma = self.nlp(word)[0].lemma_.lower()
        for doc in tqdm(self.docs, desc=f"Looking up word {word} in documents..."):
            paragraphs = doc.split("\n")
            # Process each paragraph with spaCy
            for chunk in paragraphs:
                if chunk.strip():  # Ensure paragraph is not just whitespace
                    doc = self.nlp(chunk)
                    # Check if any word in the paragraph has the same lemma as the target word
                    if any(word.lemma_.lower() == target_lemma for word in doc):
                        matching_paragraphs += [self.__preprocess_text(chunk)]

        txt = "\n".join(matching_paragraphs)
        with open(file, "w") as f:
            f.write(txt)

    @staticmethod
    def __preprocess_text(txt: str) -> str:
        """
        Preprocesses the input string by removing extra spaces and non-printable characters.

        :param input_string: The string to preprocess.
        :return: A cleaned and preprocessed version of the input string.
        """
        # Remove non-printable characters
        cleaned_string = re.sub(r"[^\x20-\x7E]+", "", txt)

        # Replace multiple spaces with a single space
        cleaned_string = re.sub(r"\s+", " ", cleaned_string)

        # Trim spaces at the beginning and end of the string
        cleaned_string = cleaned_string.strip()

        return cleaned_string

    def retrieve_context_docs(self, target_word: str) -> List[str]:
        """
        Retrieve the documents from the file.
        """
        self.lookup_word(target_word)
        with open(os.path.join(os.getenv("COMPILED_DOCS_PATH"), self.file_name + ".txt")) as f:
            txt = f.readlines()
        return txt

    def retrieve_context(
        self,
        target_word: str,
        query: str,
    ) -> Optional[List[str]]:
        """
        Retrieve documents that contain a given word.
        """
        self.lookup_word(target_word)
        context = []
        rag = RAGManager.from_file(file_path=os.path.join(os.getenv("COMPILED_DOCS_PATH"), self.file_name + ".txt"))
        docs = rag.retrieve_docs(query=query)
        logging.info(f"Retrieved documents: {len(docs)}.")
        for doc in docs:
            context += [doc.page_content]
        return context
