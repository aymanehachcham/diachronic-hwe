import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BertTrainer:
    """
    A class to train a BERT model and handle its tokenizer for extracting embeddings,
    with support for training on various devices, including Apple's M1/M2 silicon.

    Attributes
    ----------
    train_dataset : torch.utils.data.Dataset
        The dataset to train on.
    tokenizer : transformers.BertTokenizer
        The tokenizer for processing input data.
    device : str
        The device to train on ('cuda', 'mps', or 'cpu').
    epochs : int
        The number of epochs to train for.

    Methods
    -------
    train()
        Trains the BERT model and saves the tokenizer alongside the model.
    """

    def __init__(self, train_dataset: Dataset, tokenizer: BertTokenizer = None, epochs=1):
        """
        Initializes the BertTrainer with the dataset, tokenizer, device, and epochs.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            The dataset for training.
        tokenizer : transformers.BertTokenizer
            The tokenizer for processing input data.
        epochs : int, optional
            The number of epochs to train for (default is 1).
        """
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.epochs = epochs
        self.device = self._select_device()
        logging.info(f"Training on {self.device}")

        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()

        self.train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    @staticmethod
    def _select_device():
        """
        Automatically selects the most appropriate device for training.

        Returns
        -------
        device : str
            The device identifier.
        """
        if torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return 'cuda'  # NVIDIA GPU
        else:
            return 'cpu'

    def train(self):
        """
        Trains the BERT model on the specified dataset and device.
        """
        for epoch in range(self.epochs):
            progress_bar = tqdm(self.train_dataloader, leave=True, desc=f'Epoch {epoch}')
            for batch in progress_bar:
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                progress_bar.set_postfix(loss=loss.item())

        # Save the model and tokenizer to the same directory
        model_save_path = "bert_model_trained"
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
