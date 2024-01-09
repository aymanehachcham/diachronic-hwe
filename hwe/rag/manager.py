import os
from typing import List, Union

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma


class RAGManager:
    """
    Class that handles and manages the RAG retrieval process
    """

    def __init__(self, file_path: Union[str, os.PathLike]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.endswith("txt"):
            raise ValueError(f"File {file_path} is not a txt file.")
        self.text_loader = TextLoader(file_path)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    def __load_chunks(self) -> List[Document]:
        """
        Load the text file and split it into chunks
        :return: List of documents
        """
        return self.splitter.split_documents(self.text_loader.load())

    def __v_store(self) -> Chroma:
        """
        Create a vector store from the chunks
        :return: A vector store
        """
        vector_store = Chroma.from_documents(
            documents=self.__load_chunks(),
            embedding=SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2",
            ),
        )

        return vector_store

    def retrieve_docs(self, query: str) -> List[str]:
        """
        Retrieve documents from the vector store
        :param query: The query string
        :return: List of documents
        """
        vector_store = self.__v_store()
        return vector_store.similarity_search(query)
