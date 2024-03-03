from typing import List
import torch
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, PreTrainedModel, PreTrainedTokenizer
from transformers import logging as lg
from pydantic import BaseModel
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from numpy.linalg import norm
from ..utils import cutDoc


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
    _model: PreTrainedModel
    _tokenizer: PreTrainedTokenizer
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

    def _roberta_preparation(self) -> None:
        """
        This method is used to prepare the Roberta model for the inference.
        """
        if self.model_path is not None:
            self._tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
            self._model = RobertaModel.from_pretrained(
                self.model_path,
                output_hidden_states=True,
                ouput_attentions=True
            )
            self._model.eval()
            self.vocab = True
            return

        self._tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self._model = RobertaModel.from_pretrained(
            'roberta-base',
            output_hidden_states=True,
        )
        self._model.eval()
        self.vocab = True

    def _bert_case_preparation(self) -> None:
        """
        This method is used to prepare the BERT model for the inference.
        """
        if self.model_path is not None:
            self._tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self._model = BertModel.from_pretrained(
                self.model_path,
                output_hidden_states=True,
                output_attentions=True
            )
            self._model.eval()
            self.vocab = True
            return

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states=True,
            output_attentions=True
        )
        self._model.eval()
        self.vocab = True

    @staticmethod
    def standardize_word_variations(paragraph: str, target_word: str) -> str:
        """
        This method is used to standardize the word variations in a paragraph.
        :param paragraph: str
        :param target_word: str
        :return: str
        """
        # Initialize the stemmer
        stemmer = PorterStemmer()
        target_root = stemmer.stem(target_word)

        # Tokenize the paragraph into words
        words = word_tokenize(paragraph)

        # Process each word in the paragraph
        standardized_paragraph = ' '.join(
            [stemmer.stem(word) if stemmer.stem(word) == target_root else word for word in words])

        # Replace the stemmed version of the target word with the target word itself
        standardized_paragraph = standardized_paragraph.replace(target_root, target_word)
        return standardized_paragraph

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
        tokens = self._tokenizer(target_word, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            # Make sure to include output_hidden_states=True to get all hidden states
            outputs = self._model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Get all hidden states

        # Extract embeddings for the word, ignoring [CLS] and [SEP] tokens
        word_embeddings = hidden_states[-1][0, 1:-1, :].mean(dim=0)
        return word_embeddings

    def roberta_embedding(self, target_word: str) -> torch.Tensor:
        """
        Get roberta embeddings for a word
        :param target_word: str
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self._model.__class__.__name__} has not been initialized'
            )

        target_word = ' ' + target_word.strip()
        doc = f" {target_word} "
        return self.roberta_infer_vector(doc, target_word)

    def roberta_infer_vector(self, doc: str, target_word: str) -> torch.Tensor:
        """
        This method is used to infer the vector embeddings of a word from a document
        Args:
            doc (str): Document to process
            target_word (str): Main work to extract the vector embeddings for
        Returns:
            embeddings (torch.Tensor): Tensor of stacked embeddings of shape (num_embeddings, embedding_size) where num_embeddings is the number of times the main_word appears in the doc
        """
        if not self.vocab:
            raise ValueError(
                f'The Embedding model {self._model.__class__.__name__} has not been initialized'
            )
        std_doc = self.standardize_word_variations(doc, target_word)
        cut_doc = cutDoc(target_word, self._tokenizer, std_doc)
        input_ids = self._tokenizer(cut_doc, return_tensors="pt").input_ids
        target_word = ' ' + target_word.strip()
        token = self._tokenizer.encode(target_word, add_special_tokens=False)[0]
        word_token_index = torch.where(input_ids == token)[1]

        try:
            with torch.no_grad():
                embeddings = self._model(input_ids).last_hidden_state

            emb = [embeddings[0, idx] for idx in word_token_index]
            return torch.stack(emb).mean(dim=0)

        except IndexError:
            raise ValueError(f'The word: "{target_word}" does not exist in the list of tokens')

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
        std_doc = self.standardize_word_variations(doc, target_word)
        cut_doc = cutDoc(target_word, self._tokenizer, std_doc)
        marked_text = "[CLS] " + cut_doc + " [SEP]"
        tokens = self._tokenizer.tokenize(marked_text)

        try:
            main_token_id = tokens.index(target_word.lower())
            idx = self._tokenizer.convert_tokens_to_ids(tokens)
            segment_id = [1] * len(tokens)

            self._tokens_tensor = torch.tensor([idx])
            self._segments_tensors = torch.tensor([segment_id])

            with torch.no_grad():
                outputs = self._model(self._tokens_tensor, self._segments_tensors)
                hidden_states = outputs[2]

            return hidden_states[-2][0][main_token_id]

        except ValueError:
            raise ValueError(
                f'The word: "{target_word}" does not exist in the list of tokens: {tokens} from {std_doc}'
            )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model
