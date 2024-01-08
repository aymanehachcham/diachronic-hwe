import os
import logging
import json
import re
from .extraction import NewsPaperExtractorXml
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from typing import List
from nltk.corpus import wordnet as wn

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

        self.file_path = file_path
        self.docs = []
        # Detect if the file path comes from the post_process dir:
        if 'post_process_data' not in file_path:
            logging.warning(
                f'File path {file_path} does not come from the post_process_data directory.'
                f'The file will be processed first.'
            )
            NewsPaperExtractorXml.from_xml(
                file_path=file_path,
            )
            file_name = os.path.basename(file_path).replace('.xml', '.json')
            self.file_path = os.path.join(os.getenv('COMPILED_DOCS_PATH'), file_name)

    @staticmethod
    def __word_frequency(
            text: str,
            top_freq: int = 20
    ) -> List[str]:

        # Remove stopwords
        stopwords_set = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\b\w{3,}\b')

        # Tokenize the text
        tokens = tokenizer.tokenize(text.lower())
        word_counts = Counter(tokens)
        for stopword in stopwords_set:
            del word_counts[stopword]

        # Get words with frequency higher equal top_freq
        return [word for word, count in word_counts.items() if count == top_freq]

    def __compile_docs(self) -> str:
        """
        Compile all full texts into one big chunk.
        """
        pattern = r'(?<=\.)\s+$'

        try:
            with open(self.file_path, 'r') as f:
                articles = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(e)
            raise ValueError(
                f'Error: {self.file_path} is not a valid json file.'
            )

        self.docs = [art['fulltext'] for art in articles]

        txt = '.'.join(self.docs)
        txt = re.sub(pattern, '', txt, flags=re.MULTILINE)

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
    
    def most_frequent_w(self):
        """
        Get words with a min frequency of 10 that are not stopwords.
        """
        txt = self.__compile_docs()
        return self.__word_frequency(txt)

    




        



        





    

    
    


