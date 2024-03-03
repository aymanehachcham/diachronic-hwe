import os

from diachronic_hwe.hwe.embeddings import RetrieveEmbeddings
from diachronic_hwe.ai_finder.finder import GptSimilarityFinder
from diachronic_hwe.data.processor import DocumentProcessor
from diachronic_hwe.hwe.pre_train import HWEPreTrain
from diachronic_hwe.utils import cutDoc
from nltk.corpus import wordnet as wn
import logging
import lz4.frame
import concurrent.futures
from pathlib import Path
import srsly
import random
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Running the framework:
    sam = HWEPreTrain(
        words_folder_path=os.getenv("WORDS")
    )
    for year in ['1980', '1990', '2000', '2005', '2017']:
        sam.create_hwe_dataset(year)
