import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import backoff
import openai
from haystack import Document, component
from haystack.components.embedders import (
    AzureOpenAIDocumentEmbedder,
    AzureOpenAITextEmbedder,
)
from haystack.utils import Secret
from openai import AsyncAzureOpenAI
from tqdm import tqdm

from src.core.provider import EmbedderProvider
from src.providers.loader import provider
from src.utils import remove_trailing_slash

logger = logging.getLogger("wren-ai-service")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_DIMENSION = 1536


# The AsyncTextEmbedder class is an asynchronous text embedder component that uses Azure OpenAI to embed text into vectors.
# It interacts with the Azure OpenAI API asynchronously, generating embeddings for text that can be used for machine learning tasks such as search or clustering.
# The run method asynchronously fetches the embedding for the input text after performing basic processing, such as adding a prefix and suffix and handling newline characters.
# The method also includes retry logic with exponential backoff for handling rate limit errors.

@component
class AsyncTextEmbedder(AzureOpenAITextEmbedder):
    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("EMBEDDER_AZURE_OPENAI_API_KEY"),
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        # Initialize the base Azure OpenAI text embedder with configuration options such as the model name, API keys, and timeout.
        super(AsyncTextEmbedder, self).__init__(
            azure_endpoint=api_base_url,
            api_version=api_version,
            azure_deployment=model,
            dimensions=dimensions,
            api_key=api_key,
            organization=organization,
            prefix=prefix,
            suffix=suffix,
            timeout=timeout,
        )

        # Create an asynchronous client for Azure OpenAI to send embedding requests.
        self.client = AsyncAzureOpenAI(
            azure_endpoint=api_base_url,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key.resolve_value(),
        )

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
    # The run method generates an embedding for the input text by sending a request to the Azure OpenAI API.
    async def run(self, text: str):
        # Ensure that the input is a valid string; otherwise, raise an error.
        if not isinstance(text, str):
            raise TypeError(
                "AzureOpenAITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the AzureOpenAIDocumentEmbedder."
            )

        logger.info(f"Running Async Azure OpenAI text embedder with text: {text}")

        # Add a prefix and suffix to the text, and remove newlines for better embedding performance.
        text_to_embed = self.prefix + text + self.suffix
        text_to_embed = text_to_embed.replace("\n", " ")

        # Send the text to Azure OpenAI for embedding. If dimensions are provided, include them in the request.
        if self.dimensions is not None:
            response = await self.client.embeddings.create(
                model=self.azure_deployment,
                dimensions=self.dimensions,
                input=text_to_embed,
            )
        else:
            response = await self.client.embeddings.create(
                model=self.azure_deployment, input=text_to_embed
            )

        # Collect metadata such as model information and usage statistics.
        meta = {"model": response.model, "usage": dict(response.usage)}

        # Return the embedding and metadata as the output.
        return {"embedding": response.data[0].embedding, "meta": meta}

# The AsyncDocumentEmbedder class is responsible for embedding documents asynchronously using Azure OpenAI's text embedding API.
# It takes in documents, processes them in batches, and uses the Azure API to generate embeddings for each document.
# The run method is the entry point that prepares the documents for embedding and calls an internal method (_embed_batch) to perform batch embedding.
# The class is built to handle rate limit errors using exponential backoff, ensuring that requests are retried when the rate limit is exceeded.

@component
class AsyncDocumentEmbedder(AzureOpenAIDocumentEmbedder):
    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("EMBEDDER_AZURE_OPENAI_API_KEY"),
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        timeout: Optional[float] = None,
    ):
        # Initialize the Azure OpenAI document embedder with API key, model details, batch size, etc.
        super(AsyncDocumentEmbedder, self).__init__(
            azure_endpoint=api_base_url,
            api_version=api_version,
            azure_deployment=model,
            dimensions=dimensions,
            api_key=api_key,
            organization=organization,
            prefix=prefix,
            suffix=suffix,
            batch_size=batch_size,
            progress_bar=progress_bar,
            meta_fields_to_embed=meta_fields_to_embed,
            embedding_separator=embedding_separator,
            timeout=timeout,
        )

        # Create an asynchronous Azure OpenAI client for handling embedding requests.
        self.client = AsyncAzureOpenAI(
            azure_endpoint=api_base_url,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key.resolve_value(),
        )

    # Internal method that handles batch processing of embedding requests.
    # It sends multiple documents for embedding at once, processes the API response, and aggregates metadata such as usage and model info.
    async def _embed_batch(
        self, texts_to_embed: List[str], batch_size: int
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        all_embeddings = []
        meta: Dict[str, Any] = {}
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size),
            disable=not self.progress_bar,
            desc="Calculating embeddings",
        ):
            batch = texts_to_embed[i : i + batch_size]
            if self.dimensions is not None:
                response = await self.client.embeddings.create(
                    model=self.azure_deployment, dimensions=self.dimensions, input=batch
                )
            else:
                response = await self.client.embeddings.create(
                    model=self.azure_deployment, input=batch
                )
            embeddings = [el.embedding for el in response.data]
            all_embeddings.extend(embeddings)

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return all_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=3)
    # The run method is the main entry point for embedding documents. It takes a list of documents, processes them, and then uses the _embed_batch method for embedding.
    async def run(self, documents: List[Document]):
        # Ensure the input is a list of Document objects; otherwise, raise a TypeError.
        if (
            not isinstance(documents, list)
            or documents
            and not isinstance(documents[0], Document)
        ):
            raise TypeError(
                "AzureOpenAIDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the AzureOpenAITextEmbedder."
            )

        logger.info(
            f"Running Async OpenAI document embedder with documents: {documents}"
        )

        # Prepare the text for embedding by concatenating the document contents.
        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        # Embed the batch of text and return the embeddings along with metadata.
        embeddings, meta = await self._embed_batch(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size
        )

        # Assign the embeddings back to the original documents.
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
# The AzureOpenAIEmbedderProvider class is responsible for providing two types of embedders: text embedder and document embedder.
# It initializes using environmental variables for the API key, base URL, model details, and API version.
# This provider class ensures that when called, it provides embedders that are configured with the necessary credentials and settings for Azure OpenAI.

@provider("azure_openai_embedder")
class AzureOpenAIEmbedderProvider(EmbedderProvider):
    def __init__(
        self,
        embed_api_key: Secret = Secret.from_env_var("EMBEDDER_AZURE_OPENAI_API_KEY"),
        embed_api_base: str = os.getenv("EMBEDDER_AZURE_OPENAI_API_BASE"),
        embed_api_version: str = os.getenv("EMBEDDER_AZURE_OPENAI_VERSION"),
        embedding_model: str = os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL,
        embedding_model_dim: int = (
            int(os.getenv("EMBEDDING_MODEL_DIMENSION"))
            if os.getenv("EMBEDDING_MODEL_DIMENSION")
            else 0
        )
        or EMBEDDING_MODEL_DIMENSION,
        timeout: Optional[float] = (
            float(os.getenv("EMBEDDER_TIMEOUT"))
            if os.getenv("EMBEDDER_TIMEOUT")
            else 120.0
        ),
    ):
        # Set up Azure OpenAI configuration using environment variables or defaults.
        self._embedding_api_base = remove_trailing_slash(embed_api_base)
        self._embedding_api_key = embed_api_key
        self._embedding_api_version = embed_api_version
        self._embedding_model = embedding_model
        self._embedding_model_dim = embedding_model_dim
        self._timeout = timeout

        # Logging basic info about the model and API settings.
        logger.info(f"Using Azure OpenAI Embedding Model: {self._embedding_model}")
        logger.info(
            f"Using Azure OpenAI Embedding API Base: {self._embedding_api_base}"
        )
        logger.info(
            f"Using Azure OpenAI Embedding API Version: {self._embedding_api_version}"
        )

    # Method to get the text embedder. It uses the Azure OpenAI API to return a text embedder for processing text data.
    def get_text_embedder(self):
        return AsyncTextEmbedder(
            api_key=self._embedding_api_key,
            model=self._embedding_model,
            api_base_url=self._embedding_api_base,
            api_version=self._embedding_api_version,
            timeout=self._timeout,
        )

    # Method to get the document embedder. It uses the Azure OpenAI API to return a document embedder for embedding entire documents.
    def get_document_embedder(self):
        return AsyncDocumentEmbedder(
            api_key=self._embedding_api_key,
            model=self._embedding_model,
            api_base_url=self._embedding_api_base,
            api_version=self._embedding_api_version,
            timeout=self._timeout,
        )
