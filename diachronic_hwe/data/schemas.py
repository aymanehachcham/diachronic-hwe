
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Issue(BaseModel):
    url: str = Field(..., pattern=r'^https://chroniclingamerica.loc.gov/lccn/.+\.json$')
    date_issued: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')

class TitleIssue(BaseModel):
    url: str = Field(..., pattern=r'^https://chroniclingamerica.loc.gov/lccn/.+\.json$')
    name: str

class BatchIssue(BaseModel):
    url: str = Field(..., pattern=r'^https://chroniclingamerica.loc.gov/batches/.+\.json$')
    name: str

class PageIssue(BaseModel):
    url: str = Field(..., pattern=r'^https://chroniclingamerica.loc.gov/lccn/.+\.json$')
    sequence: Optional[int]
class DetailedIssue(Issue):
    title: TitleIssue
    number: Optional[str] = Field(None, pattern=r'^\d*$')
    edition: Optional[int]
    pages: list[PageIssue]

class NewsPaper(BaseModel):
    place_of_publication: str
    lccn: str
    start_year: str = Field(..., pattern=r'^\d{4}$')
    end_year: str
    name: str
    publisher: str
    issues: list[Issue]

class CompiledDoc(BaseModel):
    title: str
    fulltext: str