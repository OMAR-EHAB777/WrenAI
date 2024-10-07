import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack import Document, component
from haystack.document_stores.types import DocumentStore
from langfuse.decorators import observe

from src.core.pipeline import BasicPipeline
from src.core.provider import DocumentStoreProvider, EmbedderProvider
from src.utils import (
    async_timer,
    timer,
)

logger = logging.getLogger("wren-ai-service")


@component
class ScoreFilter:
    """
    Filters the retrieved documents based on a minimum score threshold and sorts them by score in descending order.
    
    - `documents`: The list of documents to be filtered.
    - `score`: The minimum score threshold to filter the documents (default is 0.9).
    - Returns: A list of documents that meet or exceed the score threshold, sorted in descending order.
    """
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], score: float = 0.9):
        return {
            "documents": sorted(
                filter(lambda document: document.score >= score, documents),
                key=lambda document: document.score,
                reverse=True,
            )
        }

@component
class OutputFormatter:
    """
    Formats the filtered documents into a more user-friendly output, with relevant details such as question, summary, statement, and viewId.
    
    - `documents`: The list of filtered documents.
    - Returns: A dictionary containing formatted documents with keys: 'question', 'summary', 'statement', and 'viewId'.
    """
    @component.output_types(documents=List[Optional[Dict]])
    def run(self, documents: List[Document]):
        list = []
        logger.debug(f"historical_question_output_formatter: {documents}")
        for doc in documents:
            formatted = {
                "question": doc.content,
                "summary": doc.meta.get("summary"),
                "statement": doc.meta.get("statement"),
                "viewId": doc.meta.get("viewId"),
            }
            list.append(formatted)
        return {"documents": list}

## Start of Pipeline

@async_timer
@observe(capture_input=False)
async def count_documents(store: DocumentStore, id: Optional[str] = None) -> int:
    """
    Counts the number of documents available in the document store, with an optional project_id filter.

    - `store`: The document store where documents are stored.
    - `id`: Optional project_id to filter documents by a specific project.
    - Returns: The count of documents in the store that meet the specified filters.
    """
    filters = (
        {
            "operator": "AND",
            "conditions": [
                {"field": "project_id", "operator": "==", "value": id},
            ],
        }
        if id
        else None
    )
    document_count = await store.count_documents(filters=filters)
    return document_count

@async_timer
@observe(capture_input=False, capture_output=False)
async def embedding(count_documents: int, query: str, embedder: Any) -> dict:
    """
    Generates embeddings for the user query if there are documents available in the document store.
    
    - `count_documents`: The count of available documents.
    - `query`: The query for which embeddings need to be generated.
    - `embedder`: The embedding model used to generate embeddings.
    - Returns: The embedding of the query, or an empty dictionary if no documents are available.
    """
    if count_documents:
        logger.debug(f"query: {query}")
        return await embedder.run(query)
    return {}

@async_timer
@observe(capture_input=False)
async def retrieval(embedding: dict, id: str, retriever: Any) -> dict:
    """
    Retrieves documents from the document store based on the query embedding and project_id filter.
    
    - `embedding`: The embedding generated for the query.
    - `id`: The project_id used to filter the retrieval of documents.
    - `retriever`: The component responsible for retrieving documents from the store.
    - Returns: A dictionary containing the retrieved documents.
    """
    if embedding:
        filters = (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "project_id", "operator": "==", "value": id},
                ],
            }
            if id
            else None
        )
        res = await retriever.run(
            query_embedding=embedding.get("embedding"),
            filters=filters,
        )
        return dict(documents=res.get("documents"))
    return {}

@timer
@observe(capture_input=False)
def filtered_documents(retrieval: dict, score_filter: ScoreFilter) -> dict:
    """
    Filters the retrieved documents based on their score using the ScoreFilter component.
    
    - `retrieval`: The retrieved documents.
    - `score_filter`: The ScoreFilter component used to filter documents by score.
    - Returns: A dictionary containing the filtered documents.
    """
    if retrieval:
        logger.debug(
            f"retrieval: {orjson.dumps(retrieval, option=orjson.OPT_INDENT_2).decode()}"
        )
        return score_filter.run(documents=retrieval.get("documents"))
    return {}

@timer
@observe(capture_input=False)
def formatted_output(
    filtered_documents: dict, output_formatter: OutputFormatter
) -> dict:
    """
    Formats the filtered documents into a user-friendly output format using the OutputFormatter component.
    
    - `filtered_documents`: The filtered documents.
    - `output_formatter`: The OutputFormatter component used to format the documents.
    - Returns: A dictionary containing the formatted documents.
    """
    if filtered_documents:
        logger.debug(
            f"filtered_documents: {orjson.dumps(filtered_documents, option=orjson.OPT_INDENT_2).decode()}"
        )
        return output_formatter.run(documents=filtered_documents.get("documents"))
    return {"documents": []}

## End of Pipeline



class HistoricalQuestion(BasicPipeline):
    """
    This class defines a pipeline to retrieve historical questions based on query embeddings. It uses various components for embedding, retrieval, score filtering, and output formatting.

    - `embedder_provider`: The provider for text embedding.
    - `store_provider`: The provider for document storage.
    - Inherits from `BasicPipeline` and uses an asynchronous execution model.
    """
    def __init__(
        self,
        embedder_provider: EmbedderProvider,
        store_provider: DocumentStoreProvider,
    ) -> None:
        store = store_provider.get_store(dataset_name="view_questions")
        self._components = {
            "store": store,
            "embedder": embedder_provider.get_text_embedder(),
            "retriever": store_provider.get_retriever(document_store=store),
            "score_filter": ScoreFilter(),
            # The TODO indicates future improvement by adding a filter to remove low-scoring documents if ScoreFilter isn't sufficient.
            "output_formatter": OutputFormatter(),
        }

        # Initializes the pipeline with an asynchronous driver and sets the result builder to a dictionary format.
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(self, query: str, id: Optional[str] = None) -> None:
        """
        Visualizes the pipeline execution flow for the given query.

        - `query`: The user query to retrieve historical questions.
        - `id`: Optional project ID used for filtering the results.
        - The output is saved as a .dot file in the 'outputs/pipelines/retrieval' directory.
        """
        destination = "outputs/pipelines/retrieval"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["formatted_output"],
            output_file_path=f"{destination}/historical_question.dot",
            inputs={
                "query": query,
                "id": id or "",
                **self._components,
            },
            show_legend=True,
            orient="LR",
        )

    @async_timer
    @observe(name="Historical Question")
    async def run(self, query: str, id: Optional[str] = None):
        """
        Executes the historical question retrieval pipeline asynchronously.

        - `query`: The user query for retrieving historical questions.
        - `id`: Optional project ID used for filtering.
        - Runs the pipeline components from embedding, retrieval, filtering, to output formatting.
        """
        logger.info("HistoricalQuestion pipeline is running...")
        return await self._pipe.execute(
            ["formatted_output"],
            inputs={
                "query": query,
                "id": id or "",
                **self._components,
            },
        )


if __name__ == "__main__":
    """
    The main execution block initializes necessary providers, sets up the pipeline, and validates its execution.
    
    - Loads environment variables.
    - Initializes Langfuse for logging and tracking.
    - Initializes the embedder and document store providers.
    - Visualizes and validates the pipeline for a sample query.
    """
    from langfuse.decorators import langfuse_context

    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    load_env_vars()
    init_langfuse()

    _, embedder_provider, document_store_provider, _ = init_providers(engine_config=EngineConfig())

    pipeline = HistoricalQuestion(embedder_provider=embedder_provider, store_provider=document_store_provider)

    pipeline.visualize("this is a query")
    async_validate(lambda: pipeline.run("this is a query"))

    langfuse_context.flush()
