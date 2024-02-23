from typing import List
import torch
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, PreTrainedTokenizer
from transformers import logging as lg
from pydantic import BaseModel
import numpy as np
from numpy.linalg import norm


class Sense(BaseModel):
    id: str
    definition: str
    embedding: List[float]


class SenseIdentifier(BaseModel):
    word: str
    senses: List[Sense]


class VectorEmbeddings:
    """
    This class is used to infer the vector embeddings of a word from a sentence.

    Methods
    -------
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        _bert_case_preparation()
            This method is used to prepare the BERT model for the inference.
    """
    _model_path: str
    _tokens: List
    _model: BertPreTrainedModel
    _bert_tokenizer: PreTrainedTokenizer
    vocab: bool
    _tokens_tensor: torch.Tensor
    _segments_tensors: torch.Tensor

    def __init__(
            self,
            model_path: str = None,
    ):
        self.model_path = model_path
        self._tokens = []
        self._tokens_tensor = torch.tensor([])
        self._segments_tensors = torch.tensor([])

        lg.set_verbosity_error()
        self._bert_case_preparation()

    @property
    def tokens(self):
        return self._tokens

    def _bert_case_preparation(self) -> None:
        """
        This method is used to prepare the BERT model for the inference.
        """
        if self.model_path is not None:
            self._bert_tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self._model = BertModel.from_pretrained(
                self.model_path,
                output_hidden_states=True,
            )
            self._model.eval()
            self.vocab = True
            return

        self._bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True,
        )
        self._model.eval()
        self.vocab = True

    @staticmethod
    def get_similarity(vect_a: np.array, vect_b: np.array):
        """
        Returns the cosine similarity between two vectors
        Args:
            vect_a: np.array
            vect_b: np.array

        Returns:
            float
        """
        return (vect_a @ vect_b) / (norm(vect_a) * norm(vect_b))

    def word_embedding(self, target_word: str) -> torch.Tensor:
        """
        This method is used to get the word embeddings of the target word.
        :param target_word: str
        :return: torch.Tensor
        """
        tokens = self._bert_tokenizer.tokenize(target_word, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = self._model(**tokens)
            hidden_states = outputs.hidden_states  # Get all hidden states

        # Extract embeddings for the word, ignoring [CLS] and [SEP] tokens
        word_embeddings = hidden_states[-1][0, 1:-1, :].mean(dim=0)
        return word_embeddings

    def infer_vector(self, doc: str, target_word: str) -> torch.Tensor:
        """
        Main function, used to infer the vector embeddings of a word from a sentence.
        :param doc: str, The sentence from which to infer the vector embeddings
        :param target_word: str, The word to infer the vector embeddings from
        :return: torch.Tensor, tensor of the vector embeddings
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self._model} has not been initialized'
            )
        marked_text = "[CLS] " + doc + " [SEP]"
        tokens = self._bert_tokenizer.tokenize(marked_text)
        try:
            main_token_id = tokens.index(target_word.lower())
            idx = self._bert_tokenizer.convert_tokens_to_ids(tokens)
            segment_id = [1] * len(tokens)

            self._tokens_tensor = torch.tensor([idx])
            self._segments_tensors = torch.tensor([segment_id])

            with torch.no_grad():
                outputs = self._model(self._tokens_tensor, self._segments_tensors)
                hidden_states = outputs[2]

            return hidden_states[-2][0][main_token_id]

        except ValueError:
            raise ValueError(
                f'The word: "{target_word}" does not exist in the list of tokens: {tokens} from {doc}'
            )