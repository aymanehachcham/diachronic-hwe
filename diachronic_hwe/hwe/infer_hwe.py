import os

import numpy as np
import srsly
from typing import List
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
from .train_hwe import TrainPoincareEmbeddings
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
import itertools

load_dotenv()


class InferPoincareEmbeddings:
    """
    Class to infer Poincare embeddings from a given model.
    """
    _model_path: str
    _vectors_path: str
    _model: PoincareModel
    _kv: PoincareKeyedVectors

    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Folder {model_dir} does not exist.")
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f"{model_dir} is not a directory.")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".model"):
                    self._model_path = os.path.join(root, file)
                elif file.endswith(".w2vec"):
                    self._vectors_path = os.path.join(root, file)

        self.model = PoincareModel.load(self._model_path)

    def extracting_embeddings(self, word: str) -> List:
        """
        Extract the embeddings from the model.
        """
        if self.model.kv is None:
            print('No vector for:', word)
        return self.model.kv[word]

    def visualize_embeddings(self, word: str):
        """
        Visualize the embeddings of the given words in a 2d space.
        """
        words = [f'{word}.n.02', f'{word}.n.04', f'{word}.n.05']
        all_sims = [self.most_similar(w) for w in words]
        sims = list(itertools.chain(*all_sims))
        embeddings_w = [self.extracting_embeddings(word) for word in words]
        embeddings_sim = [self.extracting_embeddings(sim) for sim in sims]
        all_embeddings = np.array(embeddings_w + embeddings_sim)

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
        embeddings_2d_transformed = tsne.fit_transform(all_embeddings)
        cluster_assignments = {}
        for i, sim_word_embedding in enumerate(embeddings_2d_transformed[len(words):]):
            distances = [euclidean(sim_word_embedding, embeddings_2d_transformed[j]) for j in range(len(words))]
            nearest_word_idx = np.argmin(distances)
            if nearest_word_idx in cluster_assignments:
                cluster_assignments[nearest_word_idx].append(i + len(words))
            else:
                cluster_assignments[nearest_word_idx] = [i + len(words)]

        # Now, compute the embedding for "network"
        network_n01_embedding = self.extracting_embeddings("network.n.01")
        # Reduce its dimensionality with the same t-SNE model (or retrain including this point)
        network_n01_embedding_2d = tsne.fit_transform(np.append(all_embeddings, [network_n01_embedding], axis=0))[-1]

        # Find the nearest cluster/word to "network.n.01"
        distances = [euclidean(network_n01_embedding_2d, point) for point in embeddings_2d_transformed]
        nearest_idx = np.argmin(distances)
        print(nearest_idx)

        # Correct the plotting logic
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'orange', 'purple']

        for i, word in enumerate(words):
            x, y = embeddings_2d_transformed[i, 0], embeddings_2d_transformed[i, 1]
            plt.scatter(x, y, color=colors[i], s=100, label=f'Centroid: {word}')
            plt.annotate(word, (x, y))
            if i in cluster_assignments:
                for sim_idx in cluster_assignments[i]:
                    sim_x, sim_y = embeddings_2d_transformed[sim_idx, 0], embeddings_2d_transformed[sim_idx, 1]
                    plt.scatter(sim_x, sim_y, color=colors[i], s=20)

        # Then plot "network.n.01" separately, ensuring it doesn't rely on embeddings_2d_transformed
        # This assumes network_n01_embedding_2d is computed correctly, outside the loop
        plt.scatter(network_n01_embedding_2d[0], network_n01_embedding_2d[1], color='red', alpha=0.4, s=300, label="network")
        plt.annotate("network", (network_n01_embedding_2d[0], network_n01_embedding_2d[1]))

        year = self._model_path.split("/")[-2].split("_")[0]
        plt.title(f'Visualization with "network" year {year}')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

    def most_similar(self, word: str, topn: int = 40):
        """
        Get the most similar words to the given word.
        """
        # print(self.model.kv.descendants(f'{word}'))
        # print(f'Closest child to {word}', self.model.kv.closest_child(f'{word}'))
        # print('n.02', self.model.kv.difference_in_hierarchy(f'{word}', 'network.n.01'))
        # print('n.04', self.model.kv.difference_in_hierarchy(f'{word}', 'network.n.04'))
        # print('n.05', self.model.kv.difference_in_hierarchy(f'{word}', 'network.n.05'))
        # # print('n.05', self.model.kv.difference_in_hierarchy(f'{word}', 'network.n.05'))
        # print(self.model.kv.distances(f'{word}', ['network.n.01', 'network.n.04', 'network.n.05']))
        similar = self.model.kv.most_similar(f'{word}', topn=topn)
        senses = srsly.read_json(os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json"))
        for idx, sim in enumerate(similar):
            target = sim[0].split(".")[0]
            if target == word or target == 'network':
                sense = senses[target].get("senses").get(sim[0])
                similar[idx] = (sim[0], sim[1])

        # return only the words from similar:
        return [sim[1] for sim in similar]

    def rank_senses(self, word: str):
        """
        Rank the senses of the given word.
        """
        ranks = {}
        years = [1980, 1990, 2000, 2005, 2017]
        senses = srsly.read_json(os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json"))
        for idx in [2, 4, 5]:
            rank = self.model.kv.rank(f'{word}.n.0{idx}', f'{word}.n.01')
            ranks[f'{word}.n.0{idx}'] = rank
        print(ranks)

    def test_visualize(self):
        """
        Test the visualization of the embeddings.
        """
        trainable_model = TrainPoincareEmbeddings(
            os.path.join(os.getenv("TRAIN"), "1980_hwe.tsv")
        )
        trainable_model.model_cfg["size"] = 2
        trainable_model.model.train(**trainable_model.train_cfg)
