import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
from haystack import Document, component
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)
from tqdm import tqdm

from src.core.provider import EmbedderProvider
from src.providers.loader import provider, pull_ollama_model
from src.utils import remove_trailing_slash

logger = logging.getLogger("wren-ai-service")
# This file defines embedding components and provider for Ollama, which is a text/document embedding service.
# The components interact with Ollama's embedding service API, creating embeddings for text or documents and preparing them for further downstream tasks.
# The provider class simplifies the creation of the embedders and manages their configuration.

# Constants for embedding service
EMBEDDER_OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_MODEL_DIMENSION = 768  # https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

# This component is an asynchronous text embedder that utilizes Ollama's API to generate embeddings for text inputs.
@component
class AsyncTextEmbedder(OllamaTextEmbedder):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434/api/embeddings",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
    ):
        super(AsyncTextEmbedder, self).__init__(
            model=model,
            url=url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
        )

    # Output type defines the embedding (list of floats) and associated metadata (dictionary).
    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run(
        self,
        text: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        logger.debug(f"Running Ollama text embedder with text: {text}")
        # Prepare the JSON payload with text and generation configurations
        payload = self._create_json_payload(text, generation_kwargs)

        # Timing and sending the request to Ollama API to generate embedding
        start = time.perf_counter()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(self.timeout)
        ) as session:
            async with session.post(
                self.url,
                json=payload,
            ) as response:
                elapsed = time.perf_counter() - start
                result = await response.json()

        # Embedding metadata includes the model and duration for embedding generation
        result["meta"] = {"model": self.model, "duration": elapsed}

        return result


# This component asynchronously embeds entire documents using Ollama's API and returns the document embeddings.
@component
class AsyncDocumentEmbedder(OllamaDocumentEmbedder):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434/api/embeddings",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        prefix: str = "",
        suffix: str = "",
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        super(AsyncDocumentEmbedder, self).__init__(
            model=model,
            url=url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            prefix=prefix,
            suffix=suffix,
            progress_bar=progress_bar,
            meta_fields_to_embed=meta_fields_to_embed,
            embedding_separator=embedding_separator,
        )

    # Embeds batches of texts (Ollama only supports single uploads, so each batch is processed individually)
    async def _embed_batch(
        self,
        texts_to_embed: List[str],
        batch_size: int,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Ollama Embedding only allows single uploads, not batching. Currently, the batch size is set to 1.
        If this changes in the future, we could modify the loop to process larger batches.
        """

        all_embeddings = []
        meta: Dict[str, Any] = {"model": self.model}

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(self.timeout)
        ) as session:
            for i in tqdm(
                range(0, len(texts_to_embed), batch_size),
                disable=not self.progress_bar,
                desc="Calculating embeddings",
            ):
                batch = texts_to_embed[i]  # Processing single text per batch
                payload = self._create_json_payload(batch, generation_kwargs)

                async with session.post(
                    self.url,
                    json=payload,
                ) as response:
                    result = await response.json()
                    all_embeddings.append(result["embedding"])

        return all_embeddings, meta

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run(
        self,
        documents: List[str],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        logger.debug(f"Running Ollama document embedder with documents: {documents}")

        # Check if the input is a valid list of documents, otherwise raise a type error
        if (
            not isinstance(documents, list)
            or documents
            and not isinstance(documents[0], Document)
        ):
            msg = (
                "OllamaDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the OllamaTextEmbedder."
            )
            raise TypeError(msg)

        # Prepare the texts to be embedded
        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        # Process the documents and generate embeddings
        embeddings, meta = await self._embed_batch(
            texts_to_embed=texts_to_embed,
            batch_size=self.batch_size,
            generation_kwargs=generation_kwargs,
        )

        # Assign the generated embeddings to the documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}


# Provider class for Ollama embedding service. It provides methods to get text and document embedders.
@provider("ollama_embedder")
class OllamaEmbedderProvider(EmbedderProvider):
    def __init__(
        self,
        url: str = os.getenv("EMBEDDER_OLLAMA_URL") or EMBEDDER_OLLAMA_URL,
        embedding_model: str = os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL,
        embedding_model_dim: int = (
            int(os.getenv("EMBEDDING_MODEL_DIMENSION"))
            if os.getenv("EMBEDDING_MODEL_DIMENSION")
            else 0
        )
        or EMBEDDING_MODEL_DIMENSION,
        timeout: Optional[int] = (
            int(os.getenv("EMBEDDER_TIMEOUT")) if os.getenv("EMBEDDER_TIMEOUT") else 120
        ),
    ):
        self._url = remove_trailing_slash(url)
        self._embedding_model = embedding_model
        self._embedding_model_dim = embedding_model_dim
        self._timeout = timeout

        # Ensures the Ollama model is pulled and ready for embedding
        pull_ollama_model(self._url, self._embedding_model)

        logger.info(f"Using Ollama Embedding Model: {self._embedding_model}")
        logger.info(f"Using Ollama URL: {self._url}")

    # Get an instance of the text embedder configured with the current settings
    def get_text_embedder(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        return AsyncTextEmbedder(
            model=self._embedding_model,
            url=f"{self._url}/api/embeddings",
            generation_kwargs=model_kwargs,
            timeout=self._timeout,
        )

    # Get an instance of the document embedder configured with the current settings
    def get_document_embedder(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        return AsyncDocumentEmbedder(
            model=self._embedding_model,
            url=f"{self._url}/api/embeddings",
            generation_kwargs=model_kwargs,
            timeout=self._timeout,
        )
