import json
import os
import re
from typing import Dict, List, Optional, Tuple
import itertools
from numpy import mean
from tqdm import tqdm
import random

import nltk
import torch
import logging
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer
from .sense_embeddings import VectorEmbeddings

from ..ai_finder.finder import GptSimilarityFinder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.getLogger('nltk').setLevel(logging.ERROR)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')


class RetrieveEmbeddings:
    """
    Class that retrieves all the information concerning the embeddings for each time corpora.
    """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.gpt = GptSimilarityFinder(os.getenv("GPT_GENERATE_EXAMPLES"))
        # specify the model path:model_path="trained_bert_corpora_1980"
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
        return None

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
        for word, tag in pos_tag(word_tokenize(context)):
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
        inputs = self.tokenizer(std_context, return_tensors="pt")
        outputs = self.model(**inputs)
        attention = outputs.attentions  # Tuple of attention weights from each layer

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
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
            return lemmas

        return [word for word, _ in context_words][:3]

    def generate_hierarchical_data(
            self,
            context: str,
            target_word: str
    ):
        """
        Function to generate hierarchical data for the target word
        :param context: str
        :param target_word: str
        """
        train_data = {}
        context_words = self.get_attended_words(context, target_word)
        postag_words = list(self.pos_tagging(context_words))

        target = list(self.pos_tagging([target_word]))[0]
        sense_target = self.sense_identification(context, target)
        if sense_target is not None:
            print(f"Sense of the target word: {target_word} -> {sense_target}")
            sense_child_word = self.sense_identification(context, postag_words[0])
            train_data["parent"] = sense_target

        print(train_data)
