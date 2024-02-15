import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from itertools import islice
from typing import Optional, Union

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from .schemas import CompiledDoc, DetailedIssue, NewsPaper

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
        self.id_url = f"{self.__url_unifier}lccn/{id_lcnn}.json"
        try:
            resp = requests.get(self.id_url)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise ValueError(f"Error: {err}\n" f"Please check the id_lcnn: {id_lcnn}") from err

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
                raise ValueError(f"Error: {err}\n" f"Please check the issue url: {issue.url}") from err
            yield DetailedIssue(**resp.json())

    def page_to_image(self, page_url: str) -> str:
        """
        Convert the page url to an image url
        :param page_url:
        :return: The image url
        """
        return page_url[:-4] + "jp2"

    def pages(self, limit: int = 2, save: bool = False) -> Optional[dict]:
        """
        Return a list of issues for the newspaper target
        :param limit: The number of issues to return
        :param save: Whether to save the images or not
        :return: A dictionary of issues and their pages
        """
        pages = {}
        for issue in islice(self.__extract_issue(), limit):
            pages[issue.date_issued] = list(map(lambda x: x[:-4] + "jp2", [page.url for page in issue.pages]))

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
                        raise ValueError(f"Error: {err}\n" f"Please check the page url: {page}") from err
                    with open(f'{self.target.name}/{date}-{page.split("/")[-1]}', "wb") as f:
                        f.write(resp.content)
            return None

        return pages


class NewsPaperExtractorXml:
    """
    This class is used to extract consistent articles from the New York Times xml data.
    """

    def __init__(self):
        raise NotImplementedError("Use the class method from_xml to initialize this class")

    @staticmethod
    def __format_text(text: str) -> str:
        """
        Format the text to remove unwanted characters
        :param text: The text to format
        :return: The formatted text
        """
        # Remove line breaks
        text = text.replace("\n", " ")

        # Remove non-alphanumeric characters
        text = re.sub(r"[^a-zA-Z0-9,. -]", "", text)
        return text

    @classmethod
    def from_xml(cls, xml_file_path: Union[str, os.PathLike]) -> None:
        """
        Extract the documents from the xml file and save them as json
        :param xml_file_path: the path to the xml file
        :return: None
        """
        if not os.path.exists(xml_file_path):
            raise ValueError(f"Error: {xml_file_path} does not exist")

        root = ET.parse(xml_file_path).getroot()
        docs = []
        elements = root.findall(".//*[title][fulltext]")
        for elem in tqdm(elements, desc=f"Extracting documents from: {xml_file_path}"):
            if elem.find("title").text is not None and elem.find("fulltext").text is not None:
                title = cls.__format_text(elem.find("title").text)
                fulltext = cls.__format_text(elem.find("fulltext").text)

                docs += [CompiledDoc(title=title, fulltext=fulltext).model_dump()]

        save_dir = os.getenv("COMPILED_DOCS_PATH")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        saved_file = str(xml_file_path).rsplit("/", maxsplit=1)[-1].split(".")[0] + ".json"
        with open(os.path.join(save_dir, saved_file), "w") as f:
            json.dump(docs, f, indent=4)
