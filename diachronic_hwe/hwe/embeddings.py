import json
import re
from typing import Dict, List, Optional, Tuple

import nltk
import torch
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer

from ..ai_finder.finder import GptSimilarityFinder

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")


class RetrieveEmbeddings:
    """
    Class that retrieves all the information concerning the embeddings for each time corpora.
    """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.gpt = GptSimilarityFinder()

    def __filter_context_words(self, tokens: List[str]) -> List[str]:
        """
        Function to filter out unwanted tokens
        :param tokens: list
        """
        # Filter for nouns and non-stop words
        tokens = list(set(tokens))
        noun_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

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
    def get_sense_definitions(word_pos: Tuple) -> Dict:
        """
        Function to get the synsets of a word based on the pos tagging
        :param word_pos: Tuple
        """
        senses = {}
        synsets = wn.synsets(word_pos[0], pos=word_pos[1])
        for synset in synsets:
            senses[synset] = synset.definition()
        return senses.copy()

    def pos_tagging(self, words: List[str]) -> List[str]:
        """
        Function to get the part of speech tag of a word
        :param words: List[str]
        """
        for word, tag in pos_tag(words):
            if self.get_wordnet_pos(tag) is not None:
                yield word, self.get_wordnet_pos(tag)
            else:
                yield word, wn.NOUN

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

    def get_attended_words(self, context: str, target_word: str, focus_layers=None) -> List[str]:
        """
        Function to get the most attended words for each target token
        :param context: str
        :param target_word: str
        :param focus_layers: list
        """
        inputs = self.tokenizer(context, return_tensors="pt")
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

        return self.__filter_context_words([word for word, _ in context_words][:5])
