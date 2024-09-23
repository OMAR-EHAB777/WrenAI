from abc import ABCMeta, abstractmethod

from haystack.document_stores.types import DocumentStore

class LLMProvider(metaclass=ABCMeta):
    """
    Abstract class that defines how to interact with Large Language Models (LLMs).
    Subclasses should implement methods to retrieve the generator model and its parameters.
    """
    @abstractmethod
    def get_generator(self, *args, **kwargs):
        """
        Retrieves the generator model for text generation (LLM).
        Subclasses should implement this method.
        """
        ...

    def get_model(self):
        """
        Returns the loaded generation model.
        """
        return self._generation_model

    def get_model_kwargs(self):
        """
        Returns the keyword arguments (settings) used to configure the model.
        """
        return self._model_kwargs


class EmbedderProvider(metaclass=ABCMeta):
    """
    Abstract class for providing text and document embedding models.
    Subclasses should implement methods for retrieving text and document embedders.
    """
    @abstractmethod
    def get_text_embedder(self, *args, **kwargs):
        """
        Returns the model used for embedding text.
        """
        ...

    @abstractmethod
    def get_document_embedder(self, *args, **kwargs):
        """
        Returns the model used for embedding documents.
        """
        ...

    def get_model(self):
        """
        Returns the loaded embedding model.
        """
        return self._embedding_model

    def get_dimensions(self):
        """
        Returns the number of dimensions of the embedding model's output.
        """
        return self._embedding_model_dim


class DocumentStoreProvider(metaclass=ABCMeta):
    """
    Abstract class for providing document stores.
    Subclasses should implement methods to retrieve the document store and retriever models.
    """
    @abstractmethod
    def get_store(self, *args, **kwargs) -> DocumentStore:
        """
        Returns the document store, which can be used for saving and retrieving documents.
        """
        ...

    @abstractmethod
    def get_retriever(self, *args, **kwargs):
        """
        Returns the retriever model for fetching documents from the document store.
        """
        ...
