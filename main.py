import os

from diachronic_hwe.hwe.embeddings import RetrieveEmbeddings
from diachronic_hwe.ai_finder.finder import GptSimilarityFinder
from diachronic_hwe.data.processor import DocumentProcessor
from diachronic_hwe.hwe.pre_train import HWEPreTrain
from diachronic_hwe.hwe.train_hwe import TrainPoincareEmbeddings
from diachronic_hwe.hwe.infer_hwe import InferPoincareEmbeddings
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
    root_folder = os.path.join(os.getenv("HWE_FOLDER"), "packages")
    sim = InferPoincareEmbeddings(
        model_dir=os.path.join(root_folder, "1980_poincare_gensim")
    ).test_visualize()