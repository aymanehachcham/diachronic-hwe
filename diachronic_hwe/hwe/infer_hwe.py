import os

import numpy as np
import srsly
from typing import List, Tuple
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
from .train_hwe import TrainPoincareEmbeddings
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
import itertools
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from queue import Queue


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

    def vocab_embeddings(self, vocab: List[str] = None) -> Tuple:
        """
        Extract all vocab embeddings
        """
        if vocab:
            return np.array([self.extracting_embeddings(k) for k in vocab]), vocab

        words = [k for k, v in self.model.kv.key_to_index.items()]
        return np.array([self.extracting_embeddings(k) for k in words]), words

    def get_vocab(self) -> List:
        """
        Get the vocabulary of the model.
        """
        return [k for k, v in self.model.kv.key_to_index.items()]

    def extracting_embeddings(self, word: str) -> List:
        """
        Extract the embeddings from the model.
        """
        if self.model.kv is None:
            print('No vector for:', word)
        return self.model.kv[word]

    def generate_synthetic_data(x, y, n_samples, scale_factor=0.05, noise_level=0.01):
        synthetic_data = []
        for _ in range(n_samples):
            # Randomly choose an index to scale and perturb
            idx = np.random.randint(0, len(x))
            new_x = x[idx] * (1 + np.random.uniform(-scale_factor, scale_factor))
            new_y = y[idx] * (1 + np.random.uniform(-scale_factor, scale_factor))

            # Add noise
            new_x += np.random.uniform(-noise_level, noise_level)
            new_y += np.random.uniform(-noise_level, noise_level)

            # Ensure the new points are within the original range
            new_x = np.clip(new_x, np.min(x), np.max(x))
            new_y = np.clip(new_y, np.min(y), np.max(y))

            synthetic_data.append([new_x, new_y])

        return np.array(synthetic_data)

    @staticmethod
    def hierarchy_clustering(vocab_embedding: np.array, k: int = 5) -> np.array:
        """
        Visualize the hierarchy of the embeddings.
        """
        x = vocab_embedding[:, 0]
        y = vocab_embedding[:, 1]
        points = np.vstack((x, y)).T

        # Use scikit-learn's NearestNeighbors to find k nearest neighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(points)
        distances, indices = nn.kneighbors(points)

        # Create a graph using networkx
        G = nx.Graph()
        for i in range(indices.shape[0]):
            for j in range(1, indices.shape[1]):  # start from 1 to skip self-loop
                G.add_edge(i, indices[i, j], weight=distances[i, j])

        # Perform Spectral Clustering
        clustering = SpectralClustering(n_clusters=k+1, assign_labels="discretize", random_state=0).fit(points)
        return clustering.labels_

    def overlapped_clusters(self, clusters_a: List[List[str]], clusters_b: List[List[str]]) -> List[List[str]]:
        """
        Find the best overlapping clusters from clusters_b for each cluster in clusters_a
        and return a new list of clusters from clusters_b that best match each cluster in clusters_a.
        """
        # Initialize a list to store the best match for each cluster in clusters_a
        best_matches = []

        for cluster_a in clusters_a:
            best_match_index = None
            best_match_count = -1

            for j, cluster_b in enumerate(clusters_b):
                overlap_count = len(set(cluster_a) & set(cluster_b))  # Count of common words

                if overlap_count > best_match_count:
                    best_match_count = overlap_count
                    best_match_index = j

            # Store the best matching cluster's index
            best_matches.append(best_match_index)

        # Form the new clusters list based on the best matches
        transformed_clusters = [clusters_b[idx] if idx is not None else [] for idx in best_matches]

        return transformed_clusters

    def visualize_hierarchy(self, vocab: List, vocab_embedding: np.array, k: int = 4):
        """
        Visualize the hierarchy of the embeddings.
        """
        clusters = self.hierarchy_clustering(vocab_embedding, k)
        points = np.vstack((vocab_embedding[:, 0], vocab_embedding[:, 1])).T
        # Define colors for each cluster
        colors = ['blue', 'magenta', 'cyan', 'green', 'orange']
        print(len(vocab), clusters.shape)

        fig = go.Figure()
        center = np.array([0, 0])
        # Plot each cluster
        for i in np.unique(clusters):
            cluster_points = points[clusters == i]
            cluster_vocab = [vocab[j] for j in np.where(clusters == i)[0]]  # Extract vocab for current cluster
            if len(cluster_points) == 0:
                continue

            # Compute distances from the center to all points in the cluster
            distances = np.linalg.norm(cluster_points - center, axis=1)
            sorted_indices = np.argsort(distances)

            # Determine point sizes based on distance, larger sizes for points closer to the center
            max_size, min_size = 30, 8
            sizes = max_size - (
                    (distances - distances.min()) / (distances.max() - distances.min()) * (max_size - min_size))

            # Use a queue to iteratively connect points starting from the one closest to the center
            q = Queue()
            visited = set()
            q.put(sorted_indices[0])

            while not q.empty():
                current = q.get()
                if current in visited:
                    continue

                visited.add(current)  # Mark current as visited
                current_point = cluster_points[current]

                # Find and plot edges to nearest unvisited neighbors
                unvisited_indices = [idx for idx in sorted_indices if idx not in visited]
                if unvisited_indices:
                    distances_to_current = cdist(cluster_points[unvisited_indices], np.array([current_point]))
                    nearest_to_current_idx = unvisited_indices[np.argmin(distances_to_current)]
                    nearest_point = cluster_points[nearest_to_current_idx]

                    # Plot edge
                    fig.add_trace(
                        go.Scatter(x=[current_point[0], nearest_point[0]], y=[current_point[1], nearest_point[1]],
                                   mode='lines', line=dict(color=colors[i % len(colors)], width=1)))
                    q.put(nearest_to_current_idx)

            # Plot the points in the cluster with variable sizes and hover text
            fig.add_trace(go.Scatter(x=cluster_points[:, 0], y=cluster_points[:, 1], mode='markers',
                                     marker=dict(size=sizes, color=colors[i % len(colors)],
                                                 line=dict(width=2, color='DarkSlateGrey')),
                                     text=cluster_vocab, hoverinfo='text', name=f'Cluster {i + 1}'))

        # Highlight the center with a large marker
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=10, color='black'), name='Center'))

        # Customize the figure layout
        # Add cross of origin and customize appearance
        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="Black", width=0.2))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="Black", width=0.2))
        # Add and customize grid appearance
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightPink')
        # Change background color and set axes' limits
        fig.update_layout(plot_bgcolor='white', xaxis=dict(range=[-1, 1], scaleanchor="y", scaleratio=1),
                          yaxis=dict(range=[-1, 1]),
                          title="Poincaré Embeddings Hierarchy",
                          showlegend=False)
        fig.show()
        # Show the plot
        # fig.write_image("./1990_solo.png", width=800, height=800)

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
        plt.scatter(network_n01_embedding_2d[0], network_n01_embedding_2d[1], color='red', alpha=0.4, s=300,
                    label="network")
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
        return [sim[0] for sim in similar]

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
