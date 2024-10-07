import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import qdrant_client
from haystack import Document, component
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import (
    QdrantDocumentStore,
    document_store,
)
from haystack_integrations.document_stores.qdrant.converters import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    convert_id,
    convert_qdrant_point_to_haystack_document,
)
from haystack_integrations.document_stores.qdrant.filters import (
    convert_filters_to_qdrant,
)
from qdrant_client.http import models as rest
from tqdm import tqdm

from src.core.provider import DocumentStoreProvider
from src.providers.loader import get_default_embedding_model_dim, provider

logger = logging.getLogger("wren-ai-service")


# Converts a list of Haystack documents into Qdrant points for indexing.
# The method handles the conversion of embeddings (both sparse and dense) into Qdrant format.
# If sparse embeddings are used, it creates a sparse vector object; otherwise, only dense embeddings are considered.
# The points are structured in a format expected by Qdrant's API and appended to a list to be returned.
def convert_haystack_documents_to_qdrant_points(
    documents: List[Document],
    *,
    use_sparse_embeddings: bool,
) -> List[rest.PointStruct]:
    points = []
    for document in documents:
        payload = document.to_dict(flatten=True)
        if use_sparse_embeddings:
            vector = {}

            dense_vector = payload.pop("embedding", None)
            if dense_vector is not None:
                vector[DENSE_VECTORS_NAME] = dense_vector

            sparse_vector = payload.pop("sparse_embedding", None)
            if sparse_vector is not None:
                sparse_vector_instance = rest.SparseVector(**sparse_vector)
                vector[SPARSE_VECTORS_NAME] = sparse_vector_instance

        else:
            vector = payload.pop("embedding") or {}
        _id = convert_id(payload.get("id"))

        point = rest.PointStruct(
            payload=payload,
            vector=vector,
            id=_id,
        )
        points.append(point)
    return points


# AsyncQdrantDocumentStore is an extension of QdrantDocumentStore but operates asynchronously.
# It is designed for high-performance embeddings and indexing with the option to use sparse embeddings.
# This class manages the interaction with the Qdrant API for indexing and querying, leveraging various configurations for better performance.
class AsyncQdrantDocumentStore(QdrantDocumentStore):
    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[Secret] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
        index: str = "Document",
        embedding_dim: int = 768,
        on_disk: bool = False,
        use_sparse_embeddings: bool = False,
        sparse_idf: bool = False,
        similarity: str = "cosine",
        return_embedding: bool = False,
        progress_bar: bool = True,
        recreate_index: bool = False,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[dict] = None,
        optimizers_config: Optional[dict] = None,
        wal_config: Optional[dict] = None,
        quantization_config: Optional[dict] = None,
        init_from: Optional[dict] = None,
        wait_result_from_api: bool = True,
        metadata: Optional[dict] = None,
        write_batch_size: int = 100,
        scroll_size: int = 10_000,
        payload_fields_to_index: Optional[List[dict]] = None,
    ):
        # Initialize the superclass (QdrantDocumentStore) with the provided configurations.
        super(AsyncQdrantDocumentStore, self).__init__(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            force_disable_check_same_thread=force_disable_check_same_thread,
            index=index,
            embedding_dim=embedding_dim,
            on_disk=on_disk,
            use_sparse_embeddings=use_sparse_embeddings,
            sparse_idf=sparse_idf,
            similarity=similarity,
            return_embedding=return_embedding,
            progress_bar=progress_bar,
            recreate_index=recreate_index,
            shard_number=shard_number,
            replication_factor=replication_factor,
            write_consistency_factor=write_consistency_factor,
            on_disk_payload=on_disk_payload,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            wal_config=wal_config,
            quantization_config=quantization_config,
            init_from=init_from,
            wait_result_from_api=wait_result_from_api,
            metadata=metadata,
            write_batch_size=write_batch_size,
            scroll_size=scroll_size,
            payload_fields_to_index=payload_fields_to_index,
        )

        # Initialize an asynchronous Qdrant client for better scalability and non-blocking operations.
        self.async_client = qdrant_client.AsyncQdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key.resolve_value() if api_key else None,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            force_disable_check_same_thread=force_disable_check_same_thread,
            metadata=metadata or {},
        )

        # Improve the performance of indexing by creating an index on the 'id' field.
        # This setup optimizes searches and queries by ensuring 'id' is indexed.
        self.client.create_payload_index(
            collection_name=index, field_name="id", field_schema="keyword"
        )

    # Perform a query to find documents using a provided embedding. The result is retrieved asynchronously.
    # The query can filter documents and return a specified number (top_k) of most relevant results.
    # If scale_score is True, the score will be adjusted based on the similarity measure (e.g., cosine).
    async def _query_by_embedding(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: bool = False,
    ) -> List[Document]:
        qdrant_filters = convert_filters_to_qdrant(filters)

        points = await self.async_client.search(
            collection_name=self.index,
            query_vector=rest.NamedVector(
                name=DENSE_VECTORS_NAME if self.use_sparse_embeddings else "",
                vector=query_embedding,
            ),
            search_params=(
                rest.SearchParams(
                    quantization=rest.QuantizationSearchParams(
                        rescore=True,
                        oversampling=3.0,
                    ),
                )
                if len(query_embedding)
                >= 1024  # Reference: Use binary quantization when embeddings are large enough.
                else None
            ),
            query_filter=qdrant_filters,
            limit=top_k,
            with_vectors=return_embedding,
        )
        # Convert Qdrant points to Haystack documents and scale scores if necessary.
        results = [
            convert_qdrant_point_to_haystack_document(
                point, use_sparse_embeddings=self.use_sparse_embeddings
            )
            for point in points
        ]
        if scale_score:
            for document in results:
                score = document.score
                if self.similarity == "cosine":
                    score = (score + 1) / 2
                else:
                    score = float(1 / (1 + np.exp(-score / 100)))
                document.score = score
        return results


  # Asynchronously deletes documents from the Qdrant collection based on the provided filters.
# If no filters are provided, it deletes all documents. It converts the filters to the format expected by Qdrant.
# If a non-existing document is referenced, a warning is logged.
async def delete_documents(self, filters: Optional[Dict[str, Any]] = None):
    if not filters:
        qdrant_filters = rest.Filter()
    else:
        qdrant_filters = convert_filters_to_qdrant(filters)

    try:
        await self.async_client.delete(
            collection_name=self.index,
            points_selector=qdrant_filters,
            wait=self.wait_result_from_api,
        )
    except KeyError:
        logger.warning(
            "Called QdrantDocumentStore.delete_documents() on a non-existing ID",
        )

# Asynchronously counts the number of documents in the Qdrant collection based on the provided filters.
# If no filters are provided, it counts all documents. Converts filters into a Qdrant-compatible format.
async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
    if not filters:
        qdrant_filters = rest.Filter()
    else:
        qdrant_filters = convert_filters_to_qdrant(filters)

    return (
        await self.async_client.count(
            collection_name=self.index, count_filter=qdrant_filters
        )
    ).count

# Asynchronously writes a batch of documents into the Qdrant collection.
# It first ensures that all documents are valid, handles duplicate documents based on the provided policy, 
# and splits them into batches for better performance. The documents are converted to Qdrant-compatible points 
# and inserted into the collection.
async def write_documents(
    self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
):
    for doc in documents:
        if not isinstance(doc, Document):
            msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
            raise ValueError(msg)

    # Set up the collection in Qdrant if it doesn't already exist
    self._set_up_collection(
        self.index,
        self.embedding_dim,
        False,
        self.similarity,
        self.use_sparse_embeddings,
        self.sparse_idf,
        self.on_disk,
        self.payload_fields_to_index,
    )

    if len(documents) == 0:
        logger.warning(
            "Calling QdrantDocumentStore.write_documents() with empty list"
        )
        return

    # Handle duplicate documents according to the specified policy
    document_objects = self._handle_duplicate_documents(
        documents=documents,
        index=self.index,
        policy=policy,
    )

    # Batch the documents for more efficient writing
    batched_documents = document_store.get_batches_from_generator(
        document_objects, self.write_batch_size
    )

    # Insert documents into the Qdrant collection in batches
    with tqdm(
        total=len(document_objects), disable=not self.progress_bar
    ) as progress_bar:
        for document_batch in batched_documents:
            batch = convert_haystack_documents_to_qdrant_points(
                document_batch,
                use_sparse_embeddings=self.use_sparse_embeddings,
            )

            await self.async_client.upsert(
                collection_name=self.index,
                points=batch,
                wait=self.wait_result_from_api,
            )

            progress_bar.update(self.write_batch_size)

    return len(document_objects)



# The AsyncQdrantEmbeddingRetriever class is an asynchronous retriever that extends the QdrantEmbeddingRetriever.
# It is used to fetch documents from an asynchronous Qdrant document store based on embedding similarity. 
# The 'run' method takes in a query embedding and optional filters, top_k (number of results to return), and other configurations.
# It retrieves relevant documents using the QdrantDocumentStore's query method.
class AsyncQdrantEmbeddingRetriever(QdrantEmbeddingRetriever):
    def __init__(
        self,
        document_store: AsyncQdrantDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: bool = False,
    ):
        super(AsyncQdrantEmbeddingRetriever, self).__init__(
            document_store=document_store,
            filters=filters,
            top_k=top_k,
            scale_score=scale_score,
            return_embedding=return_embedding,
        )

    @component.output_types(documents=List[Document])
    async def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        # Perform an asynchronous query in the document store based on the query embedding and provided configurations.
        docs = await self._document_store._query_by_embedding(
            query_embedding=query_embedding,
            filters=filters or self._filters,
            top_k=top_k or self._top_k,
            scale_score=scale_score or self._scale_score,
            return_embedding=return_embedding or self._return_embedding,
        )

        return {"documents": docs}

# The QdrantProvider class is responsible for initializing and providing instances of the AsyncQdrantDocumentStore
# and AsyncQdrantEmbeddingRetriever. It manages the configuration of Qdrant's document store and retriever settings, 
# such as location, API key, and embedding model dimensions.
@provider("qdrant")
class QdrantProvider(DocumentStoreProvider):
    def __init__(
        self,
        location: str = os.getenv("QDRANT_HOST", "qdrant"),
        api_key: Optional[Secret] = Secret.from_env_var("QDRANT_API_KEY")
        if os.getenv("QDRANT_API_KEY")
        else None,
        timeout: Optional[int] = (
            int(os.getenv("QDRANT_TIMEOUT")) if os.getenv("QDRANT_TIMEOUT") else 120
        ),
    ):
        self._location = location
        self._api_key = api_key
        self._timeout = timeout

    # Returns an instance of AsyncQdrantDocumentStore, configured with the embedding model dimension and dataset name.
    # It can recreate the index if specified and supports optimizations like binary quantization for large embeddings.
    def get_store(
        self,
        embedding_model_dim: int = (
            int(os.getenv("EMBEDDING_MODEL_DIMENSION"))
            if os.getenv("EMBEDDING_MODEL_DIMENSION")
            else 0
        )
        or get_default_embedding_model_dim(
            os.getenv("EMBEDDER_PROVIDER", "openai_embedder")
        ),
        dataset_name: Optional[str] = None,
        recreate_index: bool = False,
    ):
        logger.info(
            f"Using Qdrant Document Store with Embedding Model Dimension: {embedding_model_dim}"
        )

        return AsyncQdrantDocumentStore(
            location=self._location,
            api_key=self._api_key,
            embedding_dim=embedding_model_dim,
            index=dataset_name or "Document",
            recreate_index=recreate_index,
            on_disk=True,
            timeout=self._timeout,
            quantization_config=(
                rest.BinaryQuantization(
                    binary=rest.BinaryQuantizationConfig(
                        always_ram=True,
                    )
                )
                if embedding_model_dim >= 1024
                else None
            ),
            hnsw_config=rest.HnswConfigDiff(
                payload_m=16,
                m=0,
            ),
        )

    # Returns an instance of AsyncQdrantEmbeddingRetriever, configured to retrieve top_k documents from the document store.
    def get_retriever(
        self,
        document_store: AsyncQdrantDocumentStore,
        top_k: int = 10,
    ):
        return AsyncQdrantEmbeddingRetriever(
            document_store=document_store,
            top_k=top_k,
        )
