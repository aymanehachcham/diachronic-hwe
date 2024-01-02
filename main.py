

from hwe.data.extraction import NewsPaperExtractorXml
import json


if __name__ == '__main__':
   extractor = NewsPaperExtractorXml(
       root_dir='sample_data')

   doc = extractor.docs[0]
   extractor.format_doc(doc)