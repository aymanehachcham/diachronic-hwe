
import requests
from .schemas import NewsPaper, DetailedIssue, CompiledDoc
from itertools import islice
from typing import List
import os
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re


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
    def __init__(self,
                 root_dir: os.PathLike):
        if not os.path.exists(root_dir):
            raise ValueError(
                f'Error: {root_dir} does not exist'
            )
        # Extract all files from the root directory
        self.docs = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
        self.stopwords_list = set(stopwords.words('english'))
        self.compiled_docs = {}

    def __word_frequency(self,
                       text: str,
                       top_k: int) -> dict:
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        unique_words = [word for word in tokens if word.isalpha() and word not in self.stopwords_list]
        top_words = Counter(unique_words).most_common(top_k)

        return [w[0] for w in top_words]

    def __get_children_tags(self, root: ET.Element) -> List[str]:
        return [child.tag for child in root]

    def format_doc(self, file: os.PathLike) -> None:
        tree = ET.parse(file)
        root = tree.getroot()

        compiled_doc = {}
        # Take the 4 digits representing the year from the file name, use regex
        pattern = re.compile(r'\d{4}')
        try:
            year = pattern.search(file).group()
        except AttributeError:
            year = 'NotSpecified'

        compiled_doc['year'] = year

        for record in root.findall('record'):
            if 'title' in self.__get_children_tags(record) and \
                'fulltext' in self.__get_children_tags(record):
                obj = CompiledDoc(**{
                    'title': record.find('title').text,
                    'fulltext': record.find('fulltext').text
                })
                print(obj)














