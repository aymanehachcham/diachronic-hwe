import os
from re import sub
from shapely.geometry import Polygon, Point
import numpy as np
import srsly
from typing import List, Tuple, Dict
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
from scipy.spatial import ConvexHull


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

    # @staticmethod
    def hierarchy_clustering(self, schema: Dict[str, dict], vocab: List[str], vocab_embedding: np.array, k: int = 5) -> np.array:
        """
        Visualize the hierarchy of the embeddings.
        """

        subsense_ranges = {
            'broadcasting_system.s': [(0.0, 0.1), (0.05, 0.1)],
            'interconnected_system.s': [(0.4, 0.1), (-0.2, 0.1)],
            'electronics_system.s': [(-0.4, 0.1), (0.1, 0.1)],
            'group_communication.s': [(-0.5, 0.1), (-0.5, 0.1)],
        }


        sense_ranges = {
            'broadcasting_system.s': [(0.0, 0.2), (0.4, 0.1)],
            'interconnected_system.s': [(0.5, 0.1), (-0.2, 0.13)],
            'electronics_system.s': [(-0.6, 0.1), (0.2, 0.1)],
            'group_communication.s': [(-0.7, 0.1), (-0.5, 0.1)],
        }
        
        # attended_schema = {}
        # for sense, sense_data in schema.items():
        #     attended_schema[sense] = []
        #     for _, subsense_data in sense_data.items():
        #         attended_schema[sense].extend(subsense_data)

        x = vocab_embedding[:, 0]
        y = vocab_embedding[:, 1]
        points = np.vstack((x, y)).T


        plot_points = []
        subsense_clustering: Dict[str,list] = {}
        subsense_index: Dict[str,list] = {}

        clustering: Dict[str,list] = {}
        clustering_index: Dict[str,list] = {}
        for i, v in enumerate(vocab):
            # print('Word:', v, 'Index:', i)
            added = False
            if v.endswith(".t"):
                clustering[v] = [points[i]]

                # clustering[v] = [(0.05, -0.02)]
                clustering_index[v] = [v]


                # plot_points.append(points[i])
                # print('Target')
                added = True
            
            elif v.endswith(".s"):
                if v in list(clustering.keys()):
                    # clustering[v].append(points[i])
                    # clustering_index[v].append(v)
                    # print('Sense in clustering')
                    added = True
                
                else:
                    # clustering[v] = [points[i]]
                    # clustering_index[v] = [v]
                    # print('Sense not in clustering')
                    clustering[v] = []
                    clustering_index[v] = []
                    added = True
            
            elif v.endswith(".ss"):
                for sense, sense_data in schema.items():
                    if v in list(sense_data.keys()):
                        parent = sense
                        point_range = subsense_ranges[parent]
                        point = self.generate_2d_points(point_range[0], point_range[1], 1)[0]
                        
                        
                        if parent in list(clustering.keys()):
                            clustering[parent].append(points[i])
                            # clustering[parent].append(point)
                            clustering_index[parent].append(v)

                            # plot_points.append(points[i])
                            # print('Subsense in clustering')
                            added = True

                        else:
                            clustering[parent] = [points[i]]
                            # clustering[parent] = [point]
                            clustering_index[parent] = [v]

                            # plot_points.append(points[i])
                            # print('Subsense not in clustering')
                            added = True
                        break
                

                subsense_clustering[v] = [points[i]]
                # subsense_clustering[v] = [point]
                subsense_index[v] = [v]
            
            else:
                for sense, sense_data in schema.items():
                    for subsense, subsense_data in sense_data.items():
                        if v in subsense_data:
                            parent = sense
                            point_range = sense_ranges[parent]
                            point = self.generate_2d_points(point_range[0], point_range[1], 2).tolist()

                           
                            

                            if parent in list(clustering.keys()):
                                clustering[parent].append(points[i])
                                clustering_index[parent].append(v)

                                # clustering[parent].extend(point)
                                # clustering_index[parent].extend([f"{v}_{m}" for m in range(2)])

                                # plot_points.append(points[i])
                                # print('Attended word in sense in clustering')
                                added = True
                            
                            else:
                                clustering[parent] = [points[i]]
                                clustering_index[parent] = [v]
                                # clustering[parent] = point
                                # clustering_index[parent] = [f"{v}_{m}" for m in range(2)]

                                # plot_points.append(points[i])
                                # print('Attended word in sense not in clustering')
                                added = True


                            if subsense in list(subsense_clustering.keys()):
                                subsense_clustering[subsense].append(points[i])
                                # subsense_clustering[subsense].append(point)
                                subsense_index[subsense].append(v)
                            
                            else:
                                subsense_clustering[subsense] = [points[i]]
                                # subsense_clustering[subsense] = [point]
                                subsense_index[subsense] = [v]

                            break
                    
                    if added:
                        break


                       
            
            if not added:
                print('Not added:', v)
                raise ValueError
        

        
            
        # sort dictionary by length of values
        clustering = dict(sorted(clustering.items(), key=lambda item: len(item[1]), reverse=False))
        clustering_index = dict(sorted(clustering_index.items(), key=lambda item: len(item[1]), reverse=False))

        subsense_clustering = dict(sorted(subsense_clustering.items(), key=lambda item: len(item[1]), reverse=False))
        subsense_index = dict(sorted(subsense_index.items(), key=lambda item: len(item[1]), reverse=False))

        # return subsense_clustering, subsense_index
            
        # print('\nNumber of Clusters: ', len(list(clustering.keys())))

      

            

        # print(len(labels))
        # print(len(vocab))

        # raise
        
        return clustering, clustering_index


        # Use scikit-learn's NearestNeighbors to find k nearest neighbors
        # nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        # nn.fit(points)
        # distances, indices = nn.kneighbors(points)

        # # Create a graph using networkx
        # G = nx.Graph()
        # for i in range(indices.shape[0]):
        #     for j in range(1, indices.shape[1]):  # start from 1 to skip self-loop
        #         G.add_edge(i, indices[i, j], weight=distances[i, j])

        # Perform Spectral Clustering
        # clustering = SpectralClustering(n_clusters=k+1, assign_labels="discretize", random_state=0).fit(points)
        # return clustering.labels_
    
    def generate_2d_points(self, x_range, y_range, num_points):
        """
        Generate 2D points within specified ranges.

        Parameters:
        - x_range: Tuple(float, float), the range for the x-axis values.
        - y_range: Tuple(float, float), the range for the y-axis values.
        - num_points: int, the number of points to generate.

        Returns:
        - np.ndarray, an array of shape (num_points, 2) containing the generated points.
        """
        x_points = []
        y_points = []

        for _ in range(num_points):
            x_point = np.random.normal(x_range[0], x_range[1])
            y_point = np.random.normal(y_range[0], y_range[1])
            x_points.append(x_point)
            y_points.append(y_point)
        
        return np.array([x_points, y_points]).T

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

    def visualize_hierarchy(self, schema,  vocab: List, vocab_embedding: np.array, period: str,  k: int = 4):
        """
        Visualize the hierarchy of the embeddings.
        """
        clustering, clustering_index = self.hierarchy_clustering(schema, vocab, vocab_embedding, k)


        for cluster, points in clustering.items():
            if cluster.endswith(".t"):
                target_point = points[0]
            
            
        
        sense_ts = {}

        # points = np.vstack((vocab_embedding[:, 0], vocab_embedding[:, 1])).T
        # print('\nPoints: ', points.shape, type(points))

        # exit(0)

        # Define colors for each cluster
        # colors = ['blue', 'magenta', 'cyan', 'green', 'orange', 'purple', 'red', 'yellow', 'black', 'pink', 'brown']

        colors = {
            'network.t': 'blue',
            'broadcasting_system.s': 'magenta',
            'interconnected_system.s': 'cyan',
            'electronics_system.s': 'green',
            'group_communication.s': 'orange',
        }
        # print(len(vocab), clusters.shape)

        fig = go.Figure()
        # center = np.array([0, 0])
        # Plot each cluster

        cluster_hulls = {}
        clusters_polygons = {}
        valid_clusters = {}
        for i, c in enumerate(list(clustering.keys())):
            cluster_points = np.array(clustering[c])

            # centroid = np.mean(cluster_points, axis=0)
            cluster_vocab: List[str] = clustering_index[c]
            # print('\nCluster Vocab:', cluster_vocab)

            subsenses = [word for word in cluster_vocab if word.endswith(".ss")]
            if len(subsenses) == 0:
                subsenses = cluster_vocab
            # print('Subsenses:', subsenses)
            # exit(0)
            subsense_points = np.array([cluster_points[cluster_vocab.index(subsense)] for subsense in subsenses])
            centroid_subsense = np.mean(subsense_points, axis=0)

            # distance from centroid_sentiment to target_point
            sense_ts[c] = np.linalg.norm(centroid_subsense - target_point)



            # if len(cluster_points) == 0:
            #     continue

            # Compute distances from the center to all points in the cluster
            # distances = np.linalg.norm(cluster_points - center, axis=1)
            # sorted_indices = np.argsort(distances)

            # Determine point sizes based on distance, larger sizes for points closer to the center
            # max_size, min_size = 30, 8
            # sizes = max_size - (
            #         (distances - distances.min()) / (distances.max() - distances.min()) * (max_size - min_size))

            # Determine point sizes based type of vocab, contains ".t": largest size, '.s': medium size, '.ss' small size, others smallest size
            sizes = [50 if '.t' in word else 20 if '.ss' in word else 40 if '.s' in word else 8 for word in cluster_vocab]

            

            # Use a queue to iteratively connect points starting from the one closest to the center
            # q = Queue()
            # visited = set()
            # q.put(sorted_indices[0])

            # while not q.empty():
            #     current = q.get()
            #     if current in visited:
            #         continue

            #     visited.add(current)  # Mark current as visited
            #     current_point = cluster_points[current]

            #     # Find and plot edges to nearest unvisited neighbors
            #     unvisited_indices = [idx for idx in sorted_indices if idx not in visited]
            #     if unvisited_indices:
            #         distances_to_current = cdist(cluster_points[unvisited_indices], np.array([current_point]))
            #         nearest_to_current_idx = unvisited_indices[np.argmin(distances_to_current)]
            #         nearest_point = cluster_points[nearest_to_current_idx]

            #         # Plot edge
            #         fig.add_trace(
            #             go.Scatter(x=[current_point[0], nearest_point[0]], y=[current_point[1], nearest_point[1]],
            #                        mode='lines', line=dict(color=colors[i % len(colors)], width=1)))
            #         q.put(nearest_to_current_idx)

            # Plot the points in the cluster with variable sizes and hover text
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0], 
                y=cluster_points[:, 1], 
                mode='markers',
                marker=dict(
                    size=sizes, 
                    # color=colors[i % len(colors)],
                    color=colors[c],
                    line=dict(width=2, color='DarkSlateGrey')
                    ),
                text=cluster_vocab, name=f'Cluster {c}')
                )
            
            fig.add_trace(go.Scatter(
                x=subsense_points[:, 0],
                y=subsense_points[:, 1],
                mode='text',  # Set mode to 'text' to display text directly on the plot
                text=subsenses,  # Set the text property to display text for each point
                textposition="bottom center",  # Position text underneath points
                showlegend=False  # Avoid duplicating legend entries
            ))
            
            fig.add_trace(go.Scatter(
                x=[centroid_subsense[0]], 
                y=[centroid_subsense[1]], 
                mode='markers',
                marker=dict(
                    size=15, 
                    # color=colors[i % len(colors)], 
                    color=colors[c],
                    symbol= "diamond"
                    ), 
                    name=f'Centroid {c}'
            ))
            # 0.05, -0.02
            fig.add_trace(go.Line(
                x=[target_point[0], centroid_subsense[0]], 
                y=[target_point[1], centroid_subsense[1]], 
                mode='lines', 
                line=dict(
                    # color=colors[i % len(colors)], 
                    color=colors[c],
                    width=1
                    )
            ))
            
            if len(cluster_points) >= 3: 
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices, :]
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)

                cluster_hulls[c] = hull
                clusters_polygons[c] = Polygon(hull_points)
                valid_clusters[c] = cluster_points
        

                fig.add_trace(go.Scatter(
                    x=hull_points[:, 0], 
                    y=hull_points[:, 1], 
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    fill='toself', 
                    fillcolor=colors[c],
                    opacity=0.3,
            ))
                
        # valid_clusters = {name: points for name, points in clustering.items() if len(points) >= 3}
        points_in_overlap = count_points_in_overlaps_direct(clusters_polygons, valid_clusters)

        return points_in_overlap


        # clusters_polygons = {
        #         name: Polygon(points[hull.vertices]) for name, (points, hull) in zip(cluster_hulls.keys(), cluster_hulls.values())
        #     }
        
        # clusters_areas = {name: poly.area for name, poly in clusters_polygons.items()}
    
        # overlap_areas = {}
        # for name1, poly1 in clusters_polygons.items():
        #     for name2, poly2 in clusters_polygons.items():
        #         if name1 < name2:  # This condition avoids repeating pairs and comparing a polygon with itself
        #             overlap_key = f"{name1} & {name2}"
        #             overlap_areas[overlap_key] = calculate_overlap_area(poly1, poly2)

        # normalized_overlaps = {}
        # for pair, overlap_area in overlap_areas.items():
        #     name1, name2 = pair.split(" & ")
        #     average_area = (clusters_areas[name1] + clusters_areas[name2]) / 2.0
        #     normalized_overlaps[pair] = overlap_area / average_area if average_area > 0 else 0
        




        # Highlight the center with a large marker
        fig.add_trace(
            go.Scatter(
                x=[0], 
                y=[0], 
                mode='markers', 
                marker=dict(
                    size=10, 
                    color='black'
                    ), 
                name='Center'
        ))

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
                          title=f"Poincaré Embeddings Hierarchy - Year {period}",
                          showlegend=True)
        # fig.show()
        # Show the plot
        # fig.write_image("./1990_solo.png", width=800, height=800)

        # return normalized_overlaps

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
        # senses = srsly.read_json(os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json"))
        # for idx, sim in enumerate(similar):
        #     target = sim[0].split(".")[0]
        #     if target == word or target == 'network':
        #         sense = senses[target].get("senses").get(sim[0])
        #         similar[idx] = (sim[0], sim[1])

        # return only the words from similar:
        # return [sim[0] for sim in similar]
        return similar

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


def calculate_overlap_area(poly1, poly2):
    intersection = poly1.intersection(poly2)
    if not intersection.is_empty:
        return intersection.area
    return 0


def count_points_in_overlaps_direct(clusters_polygons, clusters_points):
    points_in_overlap = {}
    
    # Iterate through pairs of clusters to find intersections
    for name1, poly1 in clusters_polygons.items():
        for name2, poly2 in clusters_polygons.items():
            if name1 < name2:  # Avoid duplicates and self-comparison
                intersection = poly1.intersection(poly2)
                if not intersection.is_empty:
                    # Initialize count for this pair
                    count = 0
                    # Check points from the first cluster
                    for point in clusters_points[name1]:
                        if Point(point).within(intersection):
                            count += 1
                    # Check points from the second cluster
                    for point in clusters_points[name2]:
                        if Point(point).within(intersection):
                            count += 1
                    # Update the dictionary with the count for this pair
                    points_in_overlap[f"{name1} & {name2}"] = count
    
    return points_in_overlap
