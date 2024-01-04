import os
import logging
import json
import re
from .extraction import NewsPaperExtractorXml
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DocumentProcessor:
    """
    Class that takes the extracted raw documents and analyses the word frequencies,
    and other statistics relevant for further analysis.
    """
    def __init__(
            self,
            file_path: os.PathLike,
    ):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'File {file_path} does not exist.'
            )
        # Detect if the file path comes from the post_process dir:
        if not 'post_process_data' in file_path:
            logging.warning(
                f'File path {file_path} does not come from the post_process_data directory.'
                f'The file will be processed first.'
            )
            NewsPaperExtractorXml.from_xml(
                file_path=file_path,
            )

        self.file_path = file_path
        self.docs = []

    @staticmethod
    def __word_frequency(
            text: str,
            top_freq: int = 20
    ) -> dict:

        # Remove stopwords
        stopwords_list = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\b\w{3,}\b')

        # Tokenize the text
        tokens = tokenizer.tokenize(text.lower())
        word_counts = Counter(tokens)
        for stopword in stopwords_list:
            del word_counts[stopword]

        # Get words with frequency higher than top_freq
        return [word for word, count in word_counts.items() if count > top_freq]

    def __compile_docs(self):
        """
        Compile all full texts into one big chunk.
        """
        pattern = r'(?<=\.)\s+$'

        with open(self.file_path, 'r') as f:
            articles = json.load(f)
        self.docs = [art['fulltext'] for art in articles]

        txt =  '.'.join(self.docs)
        txt = re.sub(pattern, '', txt, flags=re.MULTILINE)

        return txt

    def most_frequent_w(self):
        """
        Get words with a min frequency of 10 that are not stopwords.
        """
        txt = self.__compile_docs()
        return self.__word_frequency(txt)



