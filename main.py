import os
from re import sub

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


def get_Dataset(data, target, period):
    import pandas as pd
    senses: dict = data[target]["senses"]
    sense_list = list(senses.keys())

    pairs = []
    periods = [1980,1990,2000,2010,2017]
    p = periods.index(int(period))

    
    for sense in sense_list:
        sense_data: dict = senses[sense]
        subsenses: dict = sense_data["subsenses"]
        subsense_list = list(subsenses.keys())

        for subsense in subsense_list:
            subsense_data: dict = subsenses[subsense]
            children = subsense_data["children"][p]

            if children == 0:
                continue

            for child in children:
                s = sense + ".s"
                ss = subsense + ".ss"
                c = child 
                t = target + ".t"

                leafs = [
                    {"child": s, "parent": t},
                    {"child": ss, "parent": s},
                    {"child": c, "parent": ss}
                ]
                # leafs = [(s,t), (ss,s), (c,ss)]
                pairs.extend(leafs)
    
    df = pd.DataFrame(pairs, columns=["child", "parent"])
    df.drop_duplicates(inplace=True)
    return df




def plot_overlap_time_series(sense, other_senses, time_series_data, periods):
    """
    Plots the time series of overlap counts for a specific sense against other senses.

    Parameters:
    - sense: The sense of interest (str).
    - other_senses: A list of other senses to compare against (list of str).
    - time_series_data: The time series data containing overlap counts (list of dicts).
    """
    # Initialize a Plotly figure
    fig = go.Figure()

    # Iterate through other senses to plot their time series overlap with the sense of interest
    for other_sense in other_senses:
        counts_over_time = []  # Store the overlap counts for this sense pair over time

        # Extract counts for the current other_sense over each time slice
        for time_slice in time_series_data:
            # Construct the key used in the dictionary for this sense pair
            key = f"{other_sense} & {sense}"
            if key not in time_slice:
                # The keys are directional; try the reverse direction if the initial key is not found
                key = f"{sense} & {other_sense}"
            
            # Append the count to our list, default to 0 if the key is not found
            counts_over_time.append(time_slice.get(key, 0))

        # Add the time series to the figure
        fig.add_trace(go.Scatter(x=list(range(1, len(counts_over_time) + 1)),
                                 y=counts_over_time,
                                 mode='lines+markers',
                                 name=other_sense))

    # Update layout with titles and axis labels
    fig.update_layout(title=f'Time Series of Overlap Counts for {sense}',
                      xaxis=dict(title='Time Slice', tickmode='array', tickvals=list(range(1, len(periods) + 1)), ticktext=periods),
                      xaxis_title='Time Slice',
                      yaxis_title='Overlap Count',
                      legend_title='Compared with')

    # Show the figure
    fig.show()







if __name__ == "__main__":
    # Running the framework:

    target = "network"
    periods = ["1980", "1990", "2000", "2010", "2017"]

    # Schema:
    import json

    with open(f"input/{target}.json", "r") as f:
        schema = json.load(f)

    for p, period in enumerate(periods):
        df = get_Dataset(schema, target, period)
        df.to_csv(f"input/{target}_{period}.tsv", sep="\t", header=False, index=False)

    senses = list(schema[target]["senses"].keys())
    # subsenses = list(itertools.chain(*[list(sense["subsenses"].keys()) for sense in schema["network"]["senses"].values()]))
    subsenses = {sense: list(subsense["subsenses"].keys()) for sense, subsense in schema[target]["senses"].items() if "subsenses" in subsense}


    temporal_schema = {}
    for p, period in enumerate(periods):
        temporal_schema[period] = {}
        for sense in senses:
            temporal_schema[period][f"{sense}.s"] = {}
            for subsense in subsenses[sense]:
                temporal_schema[period][f"{sense}.s"][f"{subsense}.ss"] = schema[target]["senses"][sense]["subsenses"][subsense]["children"][p]
        


    # print('Senses:', senses)
    # print('Subsenses:', subsenses)
    # print('Temporal Schema:', temporal_schema["1980"])
    # exit()

  
    # Data processing:
    data_path = "input/network_{}.tsv"

    # import pandas as pd
    # for i, period in enumerate(periods):
    #     df = pd.read_csv(data_path.format(period), sep="\t", header=None, names=["children", "parents"])

    #     for _, row in df.iterrows():
    #         if row["children"] in senses:
    #             row["children"] += ".s"

    #         if row["children"] in subsenses:
    #             row["children"] += ".ss"
            
    #         if row["parents"] in senses:
    #             row["parents"] += ".s"
            
    #         if row["parents"] in subsenses:
    #             row["parents"] += ".ss"
            
    #         if row["parents"] == target:
    #             row["parents"] += ".t"
        
    #     df.to_csv(data_path.format(period), sep="\t", header=False, index=False)
    


    # Train Poincare embeddings dim 2
    
    # p = 0
    # TrainPoincareEmbeddings(
    #     data_path= data_path.format(periods[p]),
    #     ).train_2d()

    
    # Inference

    data = {}
    for p in range(len(periods)):
        # if p != 0:
        # TrainPoincareEmbeddings(
        #         data_path= data_path.format(periods[p]),
        #         ).train_2d()
            

        root_folder = os.path.join(os.getenv("HWE_FOLDER"), "packages")
        embeddings = InferPoincareEmbeddings(
            model_dir=os.path.join(root_folder, f"{periods[p]}_poincare_gensim_2d")
        )
        
        # vocab = embeddings.get_vocab()
        vocab_embeddings, vocab = embeddings.vocab_embeddings()
    
    



        overlap_areas = embeddings.visualize_hierarchy(
            temporal_schema[periods[p]], 
            vocab, vocab_embeddings, 
            period = periods[p]
        )
        data[periods[p]] = overlap_areas



        print(f'\n--- Most similar words to "network" in {periods[p]} ---\n')

        most_similar = embeddings.most_similar("network.t", topn=10)
        for word, score in most_similar:
            print(f"{word}: {score:.4f}")
        

    clean_overlap_areas = {f"{sense}.s": [] for sense in senses}
    for period, overlap_areas in data.items():
        for sense in senses:
            related = {k: v for k, v in overlap_areas.items() if sense in k}
            clean_overlap_areas[f"{sense}.s"].append(related)
            
        

    # exit(0)

    # clean_data = {s: [data[period][f"{s}.s"] for period in periods] for s in senses}
    
    # Save the data
    with open(f"output/{target}_ts.json", "w") as f:
        json.dump(clean_overlap_areas, f, indent=4)


    import plotly.graph_objects as go

    # Iterate through each sense to plot its overlaps with the others over time
    for sense, time_series_data in clean_overlap_areas.items():
        # Create a new figure for each sense
        fig = go.Figure()


        # Prepare data for plotting
        other_senses = [s for s in clean_overlap_areas.keys() if s != sense]
      
        
        plot_overlap_time_series(sense, other_senses, time_series_data, periods)
