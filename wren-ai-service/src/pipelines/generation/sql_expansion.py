# This file defines a pipeline for expanding an SQL query by adding more columns or modifying it, 
# such as adding DISTINCT keywords, based on user input and the provided database schema.

import logging
import sys
from pathlib import Path
from typing import Any, List

import orjson  # For handling JSON serialization and deserialization
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver  # Async support for pipeline operations
from haystack.components.builders.prompt_builder import PromptBuilder  # To build and customize prompts
from langfuse.decorators import observe  # To observe the functions for tracing and logging

from src.core.engine import Engine  # The engine that will handle SQL-related logic
from src.core.pipeline import BasicPipeline  # The base pipeline class for executing steps
from src.core.provider import LLMProvider  # Provides access to large language model (LLM) operations
from src.pipelines.common import SQLGenPostProcessor  # Post-processor to handle SQL generation results
from src.utils import async_timer, timer  # Utility decorators for timing functions
from src.web.v1.services.ask import AskHistory  # Contains history of previous user queries for context

logger = logging.getLogger("wren-ai-service")

# Template prompt for expanding SQL queries based on user input and database schema.
# The task is to modify or expand the original SQL, such as adding more columns, 
# with a final output in JSON format.
sql_expansion_system_prompt = """
### TASK ###
You are a great data analyst. You are now given a task to expand original SQL by adding more columns or add more keywords such as DISTINCT.

### INSTRUCTIONS ###
- Columns are given from the user's input
- Columns to be added must belong to the given database schema; if no such column exists, keep SQL_QUERY_STRING empty

### OUTPUT FORMAT ###
Please return the result in the following JSON format:

{
    "results": [
        {"sql": <SQL_QUERY_STRING>}
    ]
}
"""

# User prompt template to customize the SQL generation based on user input, 
# the given SQL query, and the provided database schema.
sql_expansion_user_prompt_template = """
SQL: {{sql}}

Database Schema:
{% for document in documents %}
    {{ document }}
{% endfor %}

User's input: {{query}}
"""

## Start of the SQL Expansion Pipeline

# The function to create the prompt by combining the user's input, the provided SQL query, 
# database schema, and past query history. It returns a dictionary that will be fed into the generator.
@timer
@observe(capture_input=False)
def prompt(
    query: str,
    documents: List[str],
    history: AskHistory,
    prompt_builder: PromptBuilder,
) -> dict:
    logger.debug(f"query: {query}")
    logger.debug(f"documents: {documents}")
    logger.debug(f"history: {history}")
    return prompt_builder.run(query=query, documents=documents, sql=history.sql)

# Async function to generate the expanded SQL using the prompt.
@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_expansion(prompt: dict, generator: Any) -> dict:
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=prompt.get("prompt"))

# Async function to post-process the generated SQL result. 
# This function further processes and prepares the SQL for output.
@async_timer
@observe(capture_input=False)
async def post_process(
    generate_sql_expansion: dict,
    post_processor: SQLGenPostProcessor,
    project_id: str | None = None,
) -> dict:
    logger.debug(
        f"generate_sql_expansion: {orjson.dumps(generate_sql_expansion, option=orjson.OPT_INDENT_2).decode()}"
    )
    return await post_processor.run(
        generate_sql_expansion.get("replies"), project_id=project_id
    )

## End of the Pipeline

# Class to encapsulate the SQL Expansion pipeline. Inherits from BasicPipeline.
# This pipeline consists of several components such as the LLM generator, 
# prompt builder, and post-processor to handle SQL expansion tasks.
class SQLExpansion(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        engine: Engine,
    ):
        # Components for generating SQL, building prompts, and processing results.
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_expansion_system_prompt
            ),
            "prompt_builder": PromptBuilder(
                template=sql_expansion_user_prompt_template
            ),
            "post_processor": SQLGenPostProcessor(engine=engine),
        }

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    # Method to visualize the pipeline execution in a dot format file.
    def visualize(
        self,
        query: str,
        contexts: List[str],
        history: AskHistory,
        project_id: str | None = None,
    ) -> None:
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/sql_expansion.dot",
            inputs={
                "query": query,
                "documents": contexts,
                "history": history,
                "project_id": project_id,
                **self._components,
            },
            show_legend=True,
            orient="LR",
        )

    # Main method to run the SQL Expansion pipeline. It executes the necessary steps and returns the result.
    @async_timer
    @observe(name="Sql Expansion Generation")
    async def run(
        self,
        query: str,
        contexts: List[str],
        history: AskHistory,
        project_id: str | None = None,
    ):
        logger.info("Sql Expansion Generation pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "query": query,
                "documents": contexts,
                "history": history,
                "project_id": project_id,
                **self._components,
            },
        )

# Script entry point to initialize and run the pipeline. This block of code is for testing purposes.
if __name__ == "__main__":
    from langfuse.decorators import langfuse_context

    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    load_env_vars()  # Load environment variables
    init_langfuse()  # Initialize Langfuse, a service for data observability and monitoring

    # Initialize providers and engine
    llm_provider, _, _, engine = init_providers(engine_config=EngineConfig())
    
    # Instantiate the SQL Expansion pipeline
    pipeline = SQLExpansion(llm_provider=llm_provider, engine=engine)

    # Visualize and test the pipeline execution
    pipeline.visualize(
        "this is a test query",
        [],
        AskHistory(sql="SELECT * FROM table", summary="Summary", steps=[]),
    )
    async_validate(
        lambda: pipeline.run(
            "this is a test query",
            [],
            AskHistory(sql="SELECT * FROM table", summary="Summary", steps=[]),
        )
    )

    langfuse_context.flush()  # Ensure all observability data is sent and logged
