import os

from diachronic_hwe.hwe.embeddings import RetrieveEmbeddings
from diachronic_hwe.ai_finder.finder import GptSimilarityFinder
from diachronic_hwe.data.processor import DocumentProcessor
from diachronic_hwe.hwe.pre_train import HWEPreTrain
from diachronic_hwe.hwe.train_hwe import PoincareEmbeddings
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
    # sam = HWEPreTrain(
    #     words_folder_path=os.getenv("WORDS")
    # )
    # for year in ['1980', '1990', '2000', '2005', '2017']:
    #     PoincareEmbeddings(
    #         data_path=os.path.join(os.getenv("TRAIN"), f"{year}_hwe.tsv")
    #     ).train()
    model_1980 = PoincareEmbeddings(
        data_path=os.path.join(os.getenv("TRAIN"), "1980_hwe.tsv")
    ).load()
    model_1990 = PoincareEmbeddings(
        data_path=os.path.join(os.getenv("TRAIN"), "1990_hwe.tsv")
    ).load()
    similar_1980 = model_1980.kv.most_similar('network.n.01', topn=20)
    similar_1990 = model_1990.kv.most_similar('network.n.01', topn=20)
    senses = srsly.read_json(os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json"))
    for idx, sim in enumerate(similar_1990):
        target = sim[0].split(".")[0]
        if target == "network":
            sense = senses[target].get("senses").get(sim[0])
            similar_1990[idx] = (sense, sim[1])

    print(similar_1990)

    # print(model_1980.kv.most_similar('network.n.01', topn=20))
    # print(model_1990.kv.most_similar('network.n.01', topn=20))
