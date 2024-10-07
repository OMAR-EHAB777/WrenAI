# This script defines a pipeline that helps in correcting syntactically incorrect Trino SQL queries.
# The goal is to take an incorrect SQL query, analyze it using the database schema and error messages,
# and generate a corrected version of the SQL query along with the original summary.

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import orjson  # Library for fast JSON serialization
from hamilton import base  # Hamilton declarative pipelines
from hamilton.experimental.h_async import AsyncDriver  # Async support for Hamilton
from haystack import Document  # Document class from Haystack
from haystack.components.builders.prompt_builder import PromptBuilder  # Prompt building utilities
from langfuse.decorators import observe  # Observability and tracing for logging

from src.core.engine import Engine  # SQL execution engine
from src.core.pipeline import BasicPipeline  # Base class for creating pipelines
from src.core.provider import LLMProvider  # Large Language Model provider
from src.pipelines.common import (  # Common utilities and system prompts
    TEXT_TO_SQL_RULES,
    SQLGenPostProcessor,
    sql_generation_system_prompt,
)
from src.utils import async_timer, timer  # Timing utilities for logging execution time

# Set up a logger for the module
logger = logging.getLogger("wren-ai-service")

# SQL Correction Prompt Template
# This is the system prompt that will guide the Large Language Model (LLM) to correct 
# invalid SQL queries using the provided error messages and database schema.
sql_correction_user_prompt_template = """
You are a Trino SQL expert with exceptional logical thinking skills and debugging skills.

### TASK ###
Now you are given a list of syntactically incorrect Trino SQL queries and related error messages.
With given database schema, please think step by step to correct these wrong Trino SQL quries.
...
{% for document in documents %}
    {{ document.content }}
{% endfor %}

### FINAL ANSWER FORMAT ###
The final answer must be a list of corrected SQL quries and its original corresponding summary in JSON format

{
    "results": [
        {"sql": <CORRECTED_SQL_QUERY_STRING_1>, "summary": <ORIGINAL_SUMMARY_STRING_1>},
        {"sql": <CORRECTED_SQL_QUERY_STRING_2>, "summary": <ORIGINAL_SUMMARY_STRING_2>}
    ]
}

{{ alert }}

### QUESTION ###
{% for invalid_generation_result in invalid_generation_results %}
    sql: {{ invalid_generation_result.sql }}
    summary: {{ invalid_generation_result.summary }}
    error: {{ invalid_generation_result.error }}
{% endfor %}

Let's think step by step.
"""

## Start of Pipeline

# Prompt function: Builds the input prompt for the LLM based on the documents and invalid SQL queries
@timer
@observe(capture_input=False)
def prompt(
    documents: List[Document],
    invalid_generation_results: List[Dict],
    alert: str,
    prompt_builder: PromptBuilder,
) -> dict:
    """
    Build a prompt for SQL correction based on documents (database schema) and invalid SQL queries.
    
    Args:
        documents (List[Document]): Database schema documents.
        invalid_generation_results (List[Dict]): List of invalid SQL queries and their error messages.
        alert (str): Alert or warning to guide the correction.
        prompt_builder (PromptBuilder): Builder for generating the LLM prompt.
        
    Returns:
        dict: A dictionary containing the constructed prompt.
    """
    logger.debug(
        f"documents: {orjson.dumps(documents, option=orjson.OPT_INDENT_2).decode()}"
    )
    logger.debug(
        f"invalid_generation_results: {orjson.dumps(invalid_generation_results, option=orjson.OPT_INDENT_2).decode()}"
    )
    return prompt_builder.run(
        documents=documents,
        invalid_generation_results=invalid_generation_results,
        alert=alert,
    )

# SQL Correction Generation: Generates the corrected SQL query based on the prompt
@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_correction(prompt: dict, generator: Any) -> dict:
    """
    Generate the corrected SQL query using the prompt.
    
    Args:
        prompt (dict): The input prompt for the LLM to generate SQL corrections.
        generator (Any): The LLM used to generate the corrected SQL queries.
        
    Returns:
        dict: The corrected SQL query.
    """
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=prompt.get("prompt"))


# Post-processing function: Processes and formats the generated SQL corrections
@async_timer
@observe(capture_input=False)
async def post_process(
    generate_sql_correction: dict,
    post_processor: SQLGenPostProcessor,
    project_id: str | None = None,
) -> dict:
    """
    Post-process the generated SQL corrections.
    
    Args:
        generate_sql_correction (dict): The generated SQL corrections.
        post_processor (SQLGenPostProcessor): A post-processor to further refine the generated SQL.
        project_id (str, optional): Optional project ID for tracking purposes.
        
    Returns:
        dict: The post-processed SQL corrections.
    """
    logger.debug(
        f"generate_sql_correction: {orjson.dumps(generate_sql_correction, option=orjson.OPT_INDENT_2).decode()}"
    )
    return await post_processor.run(
        generate_sql_correction.get("replies"), project_id=project_id
    )

## End of Pipeline

# SQLCorrection Class: This class represents the full pipeline for SQL query correction
class SQLCorrection(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        engine: Engine,
    ):
        # Initialize the components for SQL Correction (LLM generator, prompt builder, and post processor)
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_generation_system_prompt
            ),
            "prompt_builder": PromptBuilder(
                template=sql_correction_user_prompt_template
            ),
            "post_processor": SQLGenPostProcessor(engine=engine),
        }

        self._configs = {
            "alert": TEXT_TO_SQL_RULES,
        }

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    # Visualization method: Generates a visualization of the pipeline execution
    def visualize(
        self,
        contexts: List[Document],
        invalid_generation_results: List[Dict[str, str]],
        project_id: str | None = None,
    ) -> None:
        """
        Visualize the SQL Correction pipeline.
        
        Args:
            contexts (List[Document]): Database schema documents.
            invalid_generation_results (List[Dict[str, str]]): Invalid SQL queries and error messages.
            project_id (str, optional): Optional project ID for tracking.
        """
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/sql_correction.dot",
            inputs={
                "invalid_generation_results": invalid_generation_results,
                "documents": contexts,
                "project_id": project_id,
                **self._components,
                **self._configs,
            },
            show_legend=True,
            orient="LR",
        )

    # Run method: Executes the SQL correction pipeline asynchronously
    @async_timer
    @observe(name="SQL Correction")
    async def run(
        self,
        contexts: List[Document],
        invalid_generation_results: List[Dict[str, str]],
        project_id: str | None = None,
    ):
        """
        Run the SQL Correction pipeline.
        
        Args:
            contexts (List[Document]): Database schema documents.
            invalid_generation_results (List[Dict[str, str]]): Invalid SQL queries and error messages.
            project_id (str, optional): Optional project ID for tracking.
            
        Returns:
            dict: The result of the SQL correction.
        """
        logger.info("SQLCorrection pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "invalid_generation_results": invalid_generation_results,
                "documents": contexts,
                "project_id": project_id,
                **self._components,
                **self._configs,
            },
        )


# Script entry point to run and visualize the SQL Correction pipeline
if __name__ == "__main__":
    from langfuse.decorators import langfuse_context

    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    load_env_vars()  # Load environment variables
    init_langfuse()  # Initialize observability for Langfuse

    # Initialize the LLM provider and SQL execution engine
    llm_provider, _, _, engine = init_providers(engine_config=EngineConfig())
    pipeline = SQLCorrection(
        llm_provider=llm_provider,
        engine=engine,
    )

    # Visualize the pipeline
    pipeline.visualize([], [])
    
    # Run the pipeline asynchronously for validation
    async_validate(lambda: pipeline.run([], []))

    # Flush the Langfuse context to capture logs and tracing data
    langfuse_context.flush()
