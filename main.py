import os

from sympy import im

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
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import euclidean
import itertools
load_dotenv()

def plot_combined_hierarchy(periods_data, target_word):
    G = nx.DiGraph()
    pos = {}
    layer_y = 0
    max_width = max(len(senses) for senses in periods_data.values())

    # Position the target word at the top
    pos[target_word] = (max_width / 2, layer_y)
    G.add_node(target_word)
    layer_y -= 1

    # Assign positions for each sense in each period to create layers
    for period, senses in periods_data.items():
        layer_width = len(senses)
        start_x = (max_width - layer_width) / 2

        for i, (sense, rank) in enumerate(sorted(senses.items(), key=lambda x: x[1])):
            G.add_node(f"{sense}\n({period})")
            G.add_edge(target_word, f"{sense}\n({period})", weight=1 / rank)
            pos[f"{sense}\n({period})"] = (start_x + i, layer_y)
        layer_y -= 1

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, arrows=False,
            node_color='skyblue', font_size=10, font_weight='bold', alpha=0.8)
    plt.title(f"Evolution of Senses for '{target_word}' Across Periods")
    plt.axis('off')  # Turn off the axis for better presentation
    plt.show()


if __name__ == "__main__":
    # Running the framework:
    # Train Poincare embeddings dim 2:

    from diachronic_hwe.utils import joinTsv
    import json

    with open(os.path.join(os.getenv("TRAIN"), "network.json")) as f:
        schema = json.load(f)

    senses = list(schema["network"]["senses"].keys())
    subsenses = list(itertools.chain(*[list(sense["subsenses"].keys()) for sense in schema["network"]["senses"].values()]))

    print(senses)
    print('\n')
    print(subsenses)

    exit()

    periods = [1980,1990,2000,2010, 2017]
    file_name = "network_{}.tsv"

    paths = [os.path.join(os.getenv("TRAIN"), file_name.format(period)) for period in periods]

    data = joinTsv(paths, annotate = periods, )
    data.to_csv(os.path.join(os.getenv("TRAIN"), "network_all.tsv"), sep="\t", index=False)

    TrainPoincareEmbeddings(
        data_path=os.path.join(os.getenv("TRAIN"), "network_all.tsv")
    ).train_2d()

    # root_folder = os.path.join(os.getenv("HWE_FOLDER"), "packages")
    # embeddings = InferPoincareEmbeddings(
    #     model_dir=os.path.join(root_folder, "1980_poincare_gensim_2d")
    # )
    # vocab_embeddings = embeddings.model.kv.vocab


