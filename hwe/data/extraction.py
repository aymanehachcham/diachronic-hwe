
import requests
from .schemas import NewsPaper, DetailedIssue
from itertools import islice
class NewsPapersExtractor:

    __url_unifier = "https://chroniclingamerica.loc.gov/"
    def __init__(self, id_lcnn: str):
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

    def __extract_issue(self):
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

    def issues(self):
        for issue in islice(self.__extract_issue(), 4):
            



