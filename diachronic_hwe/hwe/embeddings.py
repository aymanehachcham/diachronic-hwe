import json
import os
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random

import srsly
import torch
import logging
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer
from .sense_embeddings import VectorEmbeddings
from ..utils import cutDoc
from dotenv import load_dotenv
from ..ai_finder.finder import GptSimilarityFinder

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.getLogger('nltk').setLevel(logging.ERROR)


class RetrieveEmbeddings:
    """
    Class that retrieves all the information concerning the embeddings for each time corpora.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.gpt = GptSimilarityFinder(os.getenv("GPT_GENERATE_EXAMPLES"))
        self.vector = VectorEmbeddings()

    def __filter_context_words(self, tokens: List[str]) -> List[str]:
        """
        Function to filter out unwanted tokens
        :param tokens: list
        """
        # Filter for nouns and non-stop words
        tokens = list(set(tokens))
        noun_tokens = [token for token in tokens if token not in self.stop_words]

        return noun_tokens

    # Function to map NLTK POS tags to WordNet POS tags
    @staticmethod
    def get_wordnet_pos(treebank_tag: str) -> Optional[str]:
        """
        Function to get the wordnet pos tag
        :param treebank_tag: str
        :return: str
            """
        if treebank_tag.startswith("J"):
            return wn.ADJ
        if treebank_tag.startswith("V"):
            return wn.VERB
        if treebank_tag.startswith("N"):
            return wn.NOUN
        if treebank_tag.startswith("R"):
            return wn.ADV
        return wn.NOUN

    @staticmethod
    def get_hyponyms(word: str) -> List[str]:
        """
        Function to get the hyponyms of a word
        :param word: str
        """
        synsets = wn.synsets(word)
        hyponyms = []
        for synset in synsets:
            for synset in synset.hyponyms():
                name = synset.name().split('.')[0]
                if '_' in name:
                    name = name.replace('_', ' ')
                hyponyms.append(name)

        random.shuffle(hyponyms)
        return hyponyms

    @staticmethod
    def get_sense_definitions(word_pos: Tuple) -> Dict:
        """
        Function to get the synsets of a word based on the pos tagging
        :param word_pos: Tuple
        """
        senses = {}
        synsets = wn.synsets(word_pos[0], pos=word_pos[1])
        for synset in synsets:
            senses[synset.name()] = synset.definition()
        return senses.copy()

    def pos_tag(self, context: str, target_word: str) -> Optional[Tuple]:
        """
        Function to get the part of speech tag of a word
        :param context: str
        :param target_word: str
        """
        std_context = self.vector.standardize_word_variations(context, target_word)
        for word, tag in pos_tag(word_tokenize(std_context)):
            if word == target_word:
                return word, self.get_wordnet_pos(tag)
        return None

    def get_most_similar(self, target_word: str, context: str) -> str:
        """
        Function that computes the context words from the target word,
        and evaluates the most similar hyponym given the context of the target word.
        :param target_word: str
        :param context: str
        """
        similarities = []
        scores = {}
        hyponyms = self.get_hyponyms(target_word)
        logging.info(f"Found Hyponyms of {target_word}: {hyponyms}")
        context_words = self.get_attended_words(context, target_word, filtering=True)
        logging.info(f"Context words: {context_words}")
        for hyponym in tqdm(hyponyms, desc="Computing similarities for target word..."):
            embedding_hyponym = self.vector.word_embedding(hyponym)
            similarities += [self.vector.get_similarity(
                embedding_hyponym,
                self.vector.infer_vector(doc=context, target_word=target_word)
            )]
            scores[hyponym] = max(similarities)

        leaf_nodes = []
        for word in context_words:
            leaf_nodes.append(self.get_sim_hyponym_context(word, context, hyponyms))

        return f"{target_word} -> {max(scores, key=scores.get)} ->  {leaf_nodes}"

    def get_sim_hyponym_context(
            self,
            context_word: str,
            context: str,
            hyponyms: List
    ) -> str:
        """
        Function to get the most similar hyponym given the context of the target word.
        :param context_word: str
        :param context: str
        :param hyponyms: List
        """
        similarities = []
        scores = {}
        embedding_context_word = self.vector.infer_vector(context, context_word)
        for hyponym in tqdm(hyponyms, desc="Computing similarities for context word..."):
            similarities += [self.vector.get_similarity(
                self.vector.word_embedding(hyponym),
                embedding_context_word
            )]
            scores[hyponym] = max(similarities)

        return f"{context_word} -> {max(scores, key=scores.get)}"

    def sense_identification(self, context: str, target_word: Tuple) -> Optional[str]:
        """
        Function to get the sense identification or most similar sense of the target word
        in a given context.
        :param target_word: str
        :param senses: Dict
        :param context: str
        """
        json_pattern = r"`json\n([\s\S]*?)\n`"
        senses = self.get_sense_definitions(target_word)
        response = self.gpt.get_sense_from_target_word(target_word=target_word[0], text=context, senses=senses)
        match = re.search(json_pattern, response)
        if match:
            json_response = json.loads(match.group(1))
            return json_response[0].get("name")
        return None

    def get_attended_words(
            self,
            context: str,
            target_word: str,
            focus_layers=None,
            filtering: bool = False
    ) -> List[str]:
        """
        Function to get the most attended words for each target token
        :param context: str
        :param target_word: str
        :param focus_layers: list
        :param filtering: bool
        """
        logging.info("Retrieving most attended words....")
        std_context = self.vector.standardize_word_variations(context, target_word)
        cut_doc = cutDoc(target_word, self.vector.tokenizer, std_context)
        inputs = self.vector.tokenizer(cut_doc, return_tensors="pt")
        outputs = self.vector.model(**inputs)
        attention = outputs.attentions  # Tuple of attention weights from each layer

        tokens = self.vector.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        target_index = tokens.index(target_word)  # Find the index of the target word

        if focus_layers is None:
            # If no specific layers are provided, focus on the last quarter of layers
            focus_layers = range(len(attention) * 3 // 4, len(attention))

        # Aggregate attention weights for the target word across specified layers
        target_attention = torch.stack([attention[layer][0][:, target_index, :].mean(0) for layer in focus_layers])

        # Average the attention across specified layers
        avg_attention = target_attention.mean(0)

        # Filter out [CLS], [SEP], punctuation, and subword continuation tokens
        filtered_tokens_indices = [
            i
            for i, token in enumerate(tokens)
            if token not in ["[CLS]", "[SEP]", ".", ","] and not token.startswith("##")
        ]
        avg_attention_filtered = avg_attention[filtered_tokens_indices]
        tokens_filtered = [tokens[i] for i in filtered_tokens_indices]

        # Sort tokens based on attention and exclude the target word itself
        sorted_indices = torch.argsort(avg_attention_filtered, descending=True)
        context_words = [
            (tokens_filtered[idx], avg_attention_filtered[idx].item())
            for idx in sorted_indices
            if tokens_filtered[idx] != target_word
        ]
        if filtering:
            lemmas = []
            words = filter(lambda x: x[0] not in self.stop_words, context_words)
            for word, _ in words:
                if self.pos_tag(context, word) is not None:
                    _, pos = self.pos_tag(context, word)
                    lemma = self.lemmatizer.lemmatize(word, pos=pos)
                    if lemma == target_word:
                        continue
                    lemmas.append(word)
            return lemmas[:3]

        return [word for word, _ in context_words][:3]

    @staticmethod
    def get_senses():
        """
        Creating a list of senses and sub_senses for each target word.
        :return:
        """
        targets = ['network', 'virus', 'web', 'wall', 'stream']
        network = [
            "social_network.n.01",
            "transportation_network.n.02",
            "utility_network.n.03",
            "computer_network.n.04",
            "television_network.n.05",
            "radio_network.n.06",
            "satellite_network.n.07",
            "digital_broadcasting_network.n.08",
            "fishing_net.n.09",
            "sports_net.n.10",
            "cargo_net.n.11",
            "safety_net.n.12",
            "road_network.n.13",
            "canal_network.n.14",
            "grid_network.n.15",
            "public_transportation_network.n.16",
            "circuit_board.n.17",
            "integrated_circuit_network.n.18",
            "power_grid.n.19",
            "sensor_network.n.20",
            "socialize.v.01",
            "collaborate.v.02",
            "share_information.v.03",
            "build_connections.v.04"
        ]
        virus = [
    "animal_virus.n.01",
    "plant_virus.n.02",
    "bacteriophage.n.03",
    "marine_virus.n.04",
    "ideological_corruption.n.05",
    "social_disruption.n.06",
    "economic_instability.n.07",
    "moral_decay.n.08",
    "ransomware.n.09",
    "worm.n.10",
    "trojan.n.11",
    "spyware.n.12"
]
        web = [
    "intricate_weaving.n.01",
    "interwoven_structure.n.02",
    "natural_network.n.03",
    "fabrication_network.n.04",
    "entanglement_trap.n.05",
    "ensnarement_device.n.06",
    "predatory_network.n.07",
    "deceptive_structure.n.08",
    "feather_barbs.n.09",
    "shaft_attachment.n.10",
    "bird_feather_structure.n.11",
    "plumage_component.n.12",
    "system_interconnection.n.13",
    "digital_network.n.14",
    "social_network.n.15",
    "transportation_network.n.16",
    "internet_collection.n.17",
    "multimedia_resources.n.18",
    "hypertext_system.n.19",
    "online_platform.n.20",
    "textile_creation.n.21",
    "weaving_process.n.22",
    "fabric_manufacture.n.23",
    "loom_work.n.24",
    "aquatic_adaptation.n.25",
    "bird_swimming_aid.n.26",
    "mammalian_membrane.n.27",
    "toe_connector.n.28",
    "weaving_activity.n.29",
    "web_construction.n.30",
    "fabric_weaving.n.31",
    "network_creation.n.32"
]
        wall = [
    "partition_wall.n.01",
    "enclosure_wall.n.02",
    "supporting_wall.n.03",
    "dividing_wall.n.04",
    "barrier_effect.n.05",
    "protection_structure.n.06",
    "separation_barrier.n.07",
    "boundary_wall.n.08",
    "anatomical_layer.n.09",
    "membrane_enclosure.n.10",
    "structural_enclosure.n.11",
    "lining_protection.n.12",
    "challenging_situation.n.13",
    "obstacle_situation.n.14",
    "restrictive_condition.n.15",
    "awkward_condition.n.16",
    "rock_face.n.17",
    "mountain_face.n.18",
    "cave_wall.n.19",
    "cliff_wall.n.20",
    "enclosing_layer.n.21",
    "space_enclosure.n.22",
    "material_layer.n.23",
    "insulation_wall.n.24",
    "garden_wall.n.25",
    "estate_fence.n.26",
    "masonry_barrier.n.27",
    "decorative_wall.n.28",
    "defensive_embankment.n.29",
    "fortification_wall.n.30",
    "protective_rampart.n.31",
    "embankment_structure.n.32",
    "fortify_structure.n.33",
    "enclose_space.n.34",
    "build_barrier.n.35",
    "construction_fortification.n.36"
]
        stream = [
    "mountain_stream.n.01",
    "underground_river.n.02",
    "river_flow.n.03",
    "brook_stream.n.04",
    "historical_progression.n.05",
    "idea_sequence.n.06",
    "event_current.n.07",
    "conceptual_flow.n.08",
    "continuous_progression.n.09",
    "flowing_movement.n.10",
    "progressive_sequence.n.11",
    "uninterrupted_transition.n.12",
    "moving_continuously.n.13",
    "flowing_resemble.n.14",
    "continuous_motion.n.15",
    "fluid_flow.n.16",
    "natural_current.n.17",
    "air_current.n.18",
    "water_current.n.19",
    "fluid_steady_flow.n.20",
    "wave_extension.n.21",
    "float_outward.n.22",
    "wind_movement.n.23",
    "banner_stream.n.24",
    "profuse_exude.n.25",
    "sweat_stream.n.26",
    "bleed_profusely.n.27",
    "secrete_abundantly.n.28",
    "crowd_movement.n.29",
    "migrate_in_numbers.n.30",
    "traffic_flow.n.31",
    "animal_migration.n.32",
    "heavy_rainfall.n.33",
    "downpour_weather.n.34",
    "torrential_rain.n.35",
    "cloudburst_event.n.36",
    "abundant_flow.n.37",
    "free_running.n.38",
    "waterfall_cascade.n.39",
    "river_discharge.n.40"
]
        word = {}
        for target in targets:
            senses = {}
            for synset in wn.synsets(target):
                senses[synset.name()] = synset.definition()

            word[target] = {}
            word[target]['senses'] = senses
            word[target]['sub_senses'] = eval(target)

        # save to a json file:
        file_path = os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json")
        srsly.write_json(file_path, word)

