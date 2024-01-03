

from hwe.data.extraction import NewsPaperExtractorXml
import json


if __name__ == '__main__':
   docs = NewsPaperExtractorXml.from_xml(
       file_path='sample_data/TheNewYorkTimes1980.xml',
   )