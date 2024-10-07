import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.engine import Engine
from src.core.pipeline import BasicPipeline, async_validate
from src.core.provider import LLMProvider
from src.utils import async_timer, timer

# Set up logging for debugging purposes
logger = logging.getLogger("wren-ai-service")

# System prompt template guiding the model to answer questions based on SQL data
sql_to_answer_system_prompt = """
### TASK

You are a data analyst that is great at answering user's questions based on the data, sql, and sql summary. Please answer the user's question in concise and clear manner.
...


### INSTRUCTIONS

1. Read the user's question and understand the user's intention.
2. Read the sql summary and understand the data.
3. Read the sql and understand the data.
4. Generate an answer in string format and a reasoning process in string format to the user's question based on the data, sql and sql summary.

### OUTPUT FORMAT

Return the output in the following JSON format:

{
    "reasoning": "<STRING>",
    "answer": "<STRING>",
}
"""
sql_to_answer_user_prompt_template = """
### Input
User's question: {{ query }}
SQL: {{ sql }}
SQL summary: {{ sql_summary }}
Data: {{ sql_data }}

Please think step by step and answer the user's question.
"""

# DataFetcher class that fetches data from the SQL engine
@component
class DataFetcher:
    def __init__(self, engine: Engine):
        """
        Initializes the DataFetcher with an SQL engine.

        Args:
            engine (Engine): The engine used to execute SQL queries.
        """
        self._engine = engine

    @component.output_types(results=Optional[Dict[str, Any]])
    async def run(self, sql: str, project_id: str | None = None):
        """
        Executes the SQL query using the engine.

        Args:
            sql (str): The SQL query to be executed.
            project_id (str | None): Optional project ID for the query.

        Returns:
            dict: The SQL query results as a dictionary.
        """
        async with aiohttp.ClientSession() as session:
            _, data, _ = await self._engine.execute_sql(sql, session, project_id=project_id, dry_run=False)
            return {"results": data}


# Post-processor that takes generated answers and extracts reasoning and answer from them
@component
class SQLAnswerGenerationPostProcessor:
    @component.output_types(results=Dict[str, Any])
    def run(self, replies: str):
        """
        Processes the SQL query generation results to extract the answer and reasoning.

        Args:
            replies (str): JSON string containing the generated replies from the model.

        Returns:
            dict: A dictionary containing the final answer, reasoning, or any error encountered.
        """
        try:
            data = orjson.loads(replies[0])
            return {
                "results": {
                    "answer": data["answer"],
                    "reasoning": data["reasoning"],
                    "error": "",
                }
            }
        except Exception as e:
            logger.exception(f"Error in SQLAnswerGenerationPostProcessor: {e}")
            return {
                "results": {
                    "answer": "",
                    "reasoning": "",
                    "error": str(e),
                }
            }


## Pipeline Execution Functions

# Asynchronously executes the SQL query and fetches the data
@async_timer
@observe(capture_input=False)
async def execute_sql(sql: str, data_fetcher: DataFetcher, project_id: str | None = None) -> dict:
    """
    Executes the SQL query and fetches the resulting data.

    Args:
        sql (str): The SQL query.
        data_fetcher (DataFetcher): The component responsible for fetching data.
        project_id (str | None): Optional project ID for the query.

    Returns:
        dict: The data fetched from the SQL query.
    """
    logger.debug(f"Executing SQL: {sql}")
    return await data_fetcher.run(sql=sql, project_id=project_id)


# Builds the prompt for SQL generation using the query, sql, and summary
@timer
@observe(capture_input=False)
def prompt(query: str, sql: str, sql_summary: str, execute_sql: dict, prompt_builder: PromptBuilder) -> dict:
    """
    Constructs a prompt for SQL answer generation based on the user's question and SQL query.

    Args:
        query (str): User's question.
        sql (str): SQL query to be used for the answer.
        sql_summary (str): Summary of the SQL query.
        execute_sql (dict): The results of the executed SQL query.
        prompt_builder (PromptBuilder): The component responsible for building the prompt.

    Returns:
        dict: The generated prompt ready for SQL answer generation.
    """
    logger.debug(f"query: {query}")
    logger.debug(f"sql: {sql}")
    logger.debug(f"sql_summary: {sql_summary}")
    logger.debug(f"sql data: {execute_sql}")

    return prompt_builder.run(query=query, sql=sql, sql_summary=sql_summary, sql_data=execute_sql["results"])


# Asynchronously generates the answer based on the prompt
@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_answer(prompt: dict, generator: Any) -> dict:
    """
    Asynchronously generates the answer based on the given prompt.

    Args:
        prompt (dict): The prompt containing the question and SQL query.
        generator (Any): The model component responsible for generating answers.

    Returns:
        dict: The generated answer and reasoning in JSON format.
    """
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=prompt.get("prompt"))


# Post-processes the generated answer to finalize it
@timer
@observe(capture_input=False)
def post_process(generate_answer: dict, post_processor: SQLAnswerGenerationPostProcessor) -> dict:
    """
    Post-processes the generated answer to extract the final result.

    Args:
        generate_answer (dict): The generated answer and reasoning.
        post_processor (SQLAnswerGenerationPostProcessor): Component that post-processes the results.

    Returns:
        dict: The final processed answer and reasoning.
    """
    logger.debug(f"generate_answer: {orjson.dumps(generate_answer, option=orjson.OPT_INDENT_2).decode()}")
    return post_processor.run(generate_answer.get("replies"))


## End of Pipeline


# SQLAnswer Pipeline Class
class SQLAnswer(BasicPipeline):
    def __init__(self, llm_provider: LLMProvider, engine: Engine):
        """
        Initializes the SQL Answer generation pipeline.

        Args:
            llm_provider (LLMProvider): The provider for the language model used for generation.
            engine (Engine): The engine used to execute SQL queries.
        """
        # Initialize the components for the SQL answer generation
        self._components = {
            "data_fetcher": DataFetcher(engine=engine),
            "prompt_builder": PromptBuilder(template=sql_to_answer_user_prompt_template),
            "generator": llm_provider.get_generator(system_prompt=sql_to_answer_system_prompt),
            "post_processor": SQLAnswerGenerationPostProcessor(),
        }

        super().__init__(AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult()))

    def visualize(self, query: str, sql: str, sql_summary: str, project_id: str | None = None) -> None:
        """
        Visualizes the pipeline execution for SQL answer generation.

        Args:
            query (str): The user's question.
            sql (str): The SQL query.
            sql_summary (str): Summary of the SQL query.
            project_id (str | None): Optional project ID for tracking.

        Returns:
            None
        """
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/sql_answer.dot",
            inputs={"query": query, "sql": sql, "sql_summary": sql_summary, "project_id": project_id, **self._components},
            show_legend=True,
            orient="LR",
        )

    # Asynchronously run the pipeline
    @async_timer
    @observe(name="SQL Answer Generation")
    async def run(self, query: str, sql: str, sql_summary: str, project_id: str | None = None) -> dict:
        """
        Runs the SQL Answer generation pipeline.

        Args:
            query (str): The user's question.
            sql (str): The SQL query.
            sql_summary (str): Summary of the SQL query.
            project_id (str | None): Optional project ID for tracking.

        Returns:
            dict: The final answer and reasoning generated by the model.
        """
        logger.info("Sql_Answer Generation pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={"query": query, "sql": sql, "sql_summary": sql_summary, "project_id": project_id, **self._components},
        )


# Main execution block for testing and debugging
if __name__ == "__main__":
    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    load_env_vars()
    init_langfuse()

    # Initialize the language model provider and engine
    llm_provider, _, _, engine = init_providers(EngineConfig())
    pipeline = SQLAnswer(llm_provider=llm_provider, engine=engine)

  
    # Visualize the pipeline for SQL Answer Generation
    pipeline.visualize("query", "SELECT * FROM table_name", "sql summary")

    # Run the SQL Answer generation pipeline
    async_validate(lambda: pipeline.run("query", "SELECT * FROM table_name", "sql summary"))

    # Flush the Langfuse context to ensure logging and tracking data is submitted
    langfuse_context.flush()
