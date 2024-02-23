from typing import List
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class NewsExamplesDataset(Dataset):
    """
    Optimized dataset class for news examples, designed for efficiency and performance.

    Attributes
    ----------
    examples : List[str]
        List of news article excerpts or sentences.
    words_to_mask : List[str]
        List of words that should be masked in the dataset.
    chunk_size : int
        Maximum length of the tokenized input.

    Methods
    -------
    __init__(self, examples, words_to_mask, chunk_size=512)
        Initializes the dataset with examples, words to mask, and tokenization parameters.
    __len__(self)
        Returns the number of examples in the dataset.
    __getitem__(self, idx)
        Returns the tokenized and masked input at the given index.
    _mask_inputs(self)
        Applies masking to the specified words in all examples.
    """

    def __init__(self, examples: List[str], words_to_mask: List[str], chunk_size=512):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.examples = examples
        self.chunk_size = chunk_size
        self.words_to_mask = words_to_mask

        # Tokenize all examples
        self.encodings = self.tokenizer(examples, return_tensors='pt', max_length=self.chunk_size, truncation=True,
                                        padding="max_length")
        # Prepare labels for masked language modeling
        self.encodings['labels'] = self.encodings['input_ids'].detach().clone()
        # Mask specific words in all examples
        self._mask_inputs()

    def _mask_inputs(self):
        """
        Masks the specified words in all examples.
        """
        mask_token_id = self.tokenizer.mask_token_id
        words_ids_to_mask = self.tokenizer.convert_tokens_to_ids(self.words_to_mask)

        # Iterate through each word to mask in each encoding and replace with mask token ID
        for word_id in words_ids_to_mask:
            self.encodings['input_ids'][self.encodings['input_ids'] == word_id] = mask_token_id
            # Ensure labels for non-masked tokens are set to -100 so they are not included in the loss
            self.encodings['labels'][self.encodings['input_ids'] != mask_token_id] = -100

    def __getitem__(self, idx):
        """
        Returns the data needed for training from the dataset at the specified index.
        """
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items() if
                key in ['input_ids', 'attention_mask', 'labels']}
        return item

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return len(self.examples)
