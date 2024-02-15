import logging
import os
from typing import List, Union

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class RAGManager:
    """
    Class that handles and manages the RAG retrieval process
    """

    _splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    _client = chromadb.Client(Settings(anonymized_telemetry=False))

    def __init__(
        self,
        docs: Union[str, TextLoader],
    ):
        if isinstance(docs, str):
            self.documents = [Document(page_content=doc) for doc in self._splitter.split_text(docs)]
        elif isinstance(docs, TextLoader):
            self.documents = self._splitter.split_documents(docs.load())
        else:
            raise TypeError(f"Argument documents must be of type str or TextLoader, not {type(docs)}")

    @classmethod
    def from_file(cls, file_path: Union[str, os.PathLike]):
        """
        Create a RAGManager from a txt file
        :param file_path: the path to the txt file
        :return: None
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.endswith("txt"):
            raise ValueError(f"File {file_path} is not a txt file.")
        return cls(docs=TextLoader(file_path=file_path))

    def __v_store(self, top_k: int = 10) -> Chroma:
        """
        Create a vector store from the chunks
        :return: A vector store
        """
        logging.info("Creating vector store...")
        vector_store = Chroma.from_documents(
            client=self._client,
            documents=self.documents,
            embedding=OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            ),
        )
        logging.info("Finished creating vector store.")
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    def retrieve_docs(self, query: str) -> List[Document]:
        """
        Retrieve documents from the vector store
        :param query: The query string
        :return: List of documents
        """
        logging.info("Retrieving documents...")
        vector_store = self.__v_store()
        return vector_store.invoke(query)
