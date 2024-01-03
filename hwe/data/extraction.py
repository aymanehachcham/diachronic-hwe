
import requests
from .schemas import NewsPaper, DetailedIssue, CompiledDoc
from itertools import islice
from typing import List, Optional
import os
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class NewsPaperExtractorOcr:
    """
    This class is used to extract data from the Chronicling America API.
    """

    __url_unifier = "https://chroniclingamerica.loc.gov/"
    def __init__(self, id_lcnn: str):
        """
        Initialize the class with the id_lcnn of the newspaper.

        Args:
            id_lcnn (str): The id_lcnn of the newspaper.

        Raises:
            ValueError: If the id_lcnn is not valid.
        """
        self.id_url = f'{self.__url_unifier}lccn/{id_lcnn}.json'
        try:
            resp = requests.get(self.id_url)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise ValueError(
                f'Error: {err}\n'
                f'Please check the id_lcnn: {id_lcnn}'
            )

        self.target = NewsPaper(**resp.json())

    def __extract_issue(self) -> DetailedIssue:
        """
        Yield back an instance of DetailedIssue for each issue of the newspaper target

        Yields:
            DetailedIssue: An instance of DetailedIssue
        """
        for issue in self.target.issues:
            try:
                resp = requests.get(issue.url)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise ValueError(
                    f'Error: {err}\n'
                    f'Please check the issue url: {issue.url}'
                )
            yield DetailedIssue(**resp.json())

    def page_to_image(self, page_url: str) -> str:
        # replace .json with .jp2 at the end of the url
        return page_url[:-4] + 'jp2'

    def pages(self, limit: int = 2, save:bool = False) -> List[str]:
        """
        Return a list of issues for the newspaper target

        Args:
            limit (int): The number of issues to return.

        Returns:
            List[str]: A list of issues.
        """
        pages = {}
        for issue in islice(self.__extract_issue(), limit):
            pages[issue.date_issued] = list(map(lambda x: x[:-4] + 'jp2', [page.url for page in issue.pages]))

        if save:
            # Create a directory for the newspaper if it doesn't exist
            if not os.path.exists(self.target.name):
                os.mkdir(self.target.name)

            # Download the images
            for date, pages in pages.items():
                for page in pages:
                    try:
                        resp = requests.get(page)
                        resp.raise_for_status()
                    except requests.exceptions.HTTPError as err:
                        raise ValueError(
                            f'Error: {err}\n'
                            f'Please check the page url: {page}'
                        )
                    with open(f'{self.target.name}/{date}-{page.split("/")[-1]}', 'wb') as f:
                        f.write(resp.content)
            return

        return pages

class NewsPaperExtractorXml:
    def __init__(self):
        raise NotImplementedError(
            'Use the class method from_xml to initialize this class'
        )

    @staticmethod
    def __format_text(text: str) -> str:
        # Remove line breaks
        text = text.replace('\n', ' ')

        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9,. -]', '', text)
        return text

    @staticmethod
    def __word_frequency(
            text: str,
            top_k: int = 10
    ) -> dict:

        # Remove stopwords
        stopwords_list = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\b\w{3,}\b')

        # Tokenize the text
        tokens = tokenizer.tokenize(text.lower())
        word_counts = Counter(tokens)
        for stopword in stopwords_list:
            del word_counts[stopword]

        # Get the top k words
        top_words = word_counts.most_common(top_k)
        return [w[0] for w in top_words]

    @classmethod
    def from_xml(
            cls,
            file_path: os.PathLike
    ) -> None:

        if not os.path.exists(file_path):
            raise ValueError(f'Error: {file_path} does not exist')

        root = ET.parse(file_path).getroot()
        docs = []
        elements = root.findall('.//*[title][fulltext]')
        for elem in tqdm(elements, desc=f'Extracting documents from: {file_path}'):
            if elem.find('title').text is not None and elem.find('fulltext').text is not None:
                title = cls.__format_text(elem.find('title').text)
                fulltext = cls.__format_text(elem.find('fulltext').text)
                top_words = cls.__word_frequency(fulltext)

                docs += [CompiledDoc(
                    title=title,
                    fulltext=fulltext,
                    top_words=top_words
                ).model_dump()]

        save_dir = os.getenv('COMPILED_DOCS_PATH')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        saved_file = file_path.__str__().split('/')[-1].split('.')[0] + '.json'
        with open(os.path.join(save_dir,saved_file), 'w') as f:
            json.dump(docs, f, indent=4)

        return

    def doc_retrieval(
            self,
            target_word: str,
            top_k: int = 10
    ) -> List[str]:

        pass














