

from hwe.data.extraction import NewsPaperExtractorXml
from hwe.data.processor import DocumentProcessor


if __name__ == '__main__':
   txt = DocumentProcessor(
       file_path='post_process_data/compiled_docs/TheNewYorkTimes1980.json'
   ).most_frequent_w()
   print(txt)