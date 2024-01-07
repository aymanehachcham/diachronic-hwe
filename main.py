

from hwe.data.extraction import NewsPaperExtractorXml
from hwe.data.processor import DocumentProcessor


if __name__ == '__main__':
   txt = DocumentProcessor(
       file_path='sample_data/TheNewYorkTimes1980.xml',
   ).most_frequent_w()
   print(txt)

       





