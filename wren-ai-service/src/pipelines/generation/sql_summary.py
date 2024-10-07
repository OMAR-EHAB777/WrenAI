import logging
import sys
from pathlib import Path
from typing import Any, List

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.utils import async_timer, timer

logger = logging.getLogger("wren-ai-service")


sql_summary_system_prompt = """
### TASK ###
You are a great data analyst. You are now given a task to summarize a list SQL queries in a human-readable format where each summary should be within 10-20 words using the same language as the user's question.
You will be given a list of SQL queries and a user's question.

### INSTRUCTIONS ###
- SQL query summary must be in the same language as the user's question.
- SQL query summary must be within 10-20 words.
- SQL query summary must be human-readable and easy to understand.
- SQL query summary must be concise and to the point.

### OUTPUT FORMAT ###
Please return the result in the following JSON format:

{
    "sql_summary_results": [
        {
            "summary": <SQL_QUERY_SUMMARY_USING_SAME_LANGUAGE_USER_QUESTION_USING_1>
        },
        {
            "summary": <SQL_QUERY_SUMMARY_USING_SAME_LANGUAGE_USER_QUESTION_USING_2>
        },
        {
            "summary": <SQL_QUERY_SUMMARY_USING_SAME_LANGUAGE_USER_QUESTION_USING_3>
        },
        ...
    ]
}
"""

sql_summary_user_prompt_template = """
User's Question: {{query}}
SQLs: {{sqls}}

Please think step by step.
"""


@component
class SQLSummaryPostProcessor:
    @component.output_types(
        sql_summary_results=List[str],  # The output is a list of strings containing SQL summaries.
    )
    def run(self, sqls: List[str], replies: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Processes the SQL generation results, extracting and formatting the SQL queries and their summaries.

        Args:
        - sqls (List[str]): List of generated SQL queries.
        - replies (List[str]): The corresponding replies from the LLM with the summaries.

        Returns:
        - dict: A dictionary containing the SQL queries and their associated summaries.
        """
        try:
            return {
                "sql_summary_results": [
                    {"sql": sql["sql"], "summary": summary["summary"]}
                    for (sql, summary) in zip(
                        sqls, orjson.loads(replies[0])["sql_summary_results"]
                    )
                ],
            }
        except Exception as e:
            logger.exception(f"Error in SQLSummaryPostProcessor: {e}")

            return {
                "sql_summary_results": [],  # Return an empty list in case of an error.
            }


## Start of Pipeline

@timer
@observe(capture_input=False)
def prompt(
    query: str,
    sqls: List[str],
    prompt_builder: PromptBuilder,
) -> dict:
    """
    Builds a prompt for generating summaries of the given SQL queries.

    Args:
    - query (str): The user's input query.
    - sqls (List[str]): List of generated SQL queries.
    - prompt_builder (PromptBuilder): The prompt builder component that constructs the prompt.

    Returns:
    - dict: A structured prompt to be sent to the LLM for SQL summaries.
    """
    logger.debug(f"query: {query}")
    logger.debug(f"sqls: {sqls}")

    return prompt_builder.run(
        query=query,
        sqls=sqls,
    )


@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_summary(prompt: dict, generator: Any) -> dict:
    """
    Sends the SQL summary prompt to the LLM generator asynchronously and retrieves the generated summaries.

    Args:
    - prompt (dict): The structured prompt for SQL summaries.
    - generator (Any): The LLM generator responsible for producing the summaries.

    Returns:
    - dict: The generated SQL summaries.
    """
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=prompt.get("prompt"))


@timer
def post_process(
    generate_sql_summary: dict,
    sqls: List[str],
    post_processor: SQLSummaryPostProcessor,
) -> dict:
    """
    Post-processes the generated SQL summaries to clean them up and format them for output.

    Args:
    - generate_sql_summary (dict): The generated SQL summaries from the LLM.
    - sqls (List[str]): The list of SQL queries that correspond to the summaries.
    - post_processor (SQLSummaryPostProcessor): Component responsible for the final processing of the SQL summaries.

    Returns:
    - dict: The post-processed SQL summaries.
    """
    logger.debug(
        f"generate_sql_summary: {orjson.dumps(generate_sql_summary, option=orjson.OPT_INDENT_2).decode()}"
    )
    return post_processor.run(sqls, generate_sql_summary.get("replies"))


## End of Pipeline


class SQLSummary(BasicPipeline):
    """
    SQLSummary defines a pipeline for generating summaries for SQL queries. It uses an LLM to generate 
    summaries and processes the results.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ):
        # Components involved in the pipeline: LLM generator, prompt builder, and post-processor.
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_summary_system_prompt  # Uses the system prompt for SQL summary generation.
            ),
            "prompt_builder": PromptBuilder(template=sql_summary_user_prompt_template),  # Builds the prompt for the LLM.
            "post_processor": SQLSummaryPostProcessor(),  # Post-processes the generated summaries.
        }

        # Initialize the asynchronous driver and results builder.
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(
        self,
        query: str,
        sqls: List[str],
    ) -> None:
        """
        Visualizes the SQL summary pipeline execution by generating a .dot file to represent 
        the flow and execution of the pipeline.

        Args:
        - query (str): The user's input query.
        - sqls (List[str]): List of SQL queries to summarize.
        """
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        # Visualize the post-processing execution of the pipeline.
        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/sql_summary.dot",
            inputs={
                "query": query,
                "sqls": sqls,
                **self._components,  # Use all the pipeline components in the visualization.
            },
            show_legend=True,
            orient="LR",  # Left-to-right orientation for the visualization.
        )

    @async_timer
    @observe(name="SQL Summary")
    async def run(
        self,
        query: str,
        sqls: List[str],
    ):
        """
        Executes the SQL summary pipeline asynchronously, generating and post-processing the SQL summaries.

        Args:
        - query (str): The user's input query.
        - sqls (List[str]): List of SQL queries to summarize.

        Returns:
        - dict: The processed SQL summaries.
        """
        logger.info("SQL Summary pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "query": query,
                "sqls": sqls,
                **self._components,  # Provide all the components to execute the pipeline.
            },
        )


if __name__ == "__main__":
    """
    Main block for testing the SQLSummary pipeline. Initializes the components 
    and runs the pipeline with sample input.
    """
    from langfuse.decorators import langfuse_context

    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    # Load environment variables.
    load_env_vars()
    init_langfuse()

    # Initialize the providers and components.
    llm_provider, _, _, _ = init_providers(engine_config=EngineConfig())

    # Instantiate the SQLSummary pipeline.
    pipeline = SQLSummary(
        llm_provider=llm_provider,
    )

    # Visualize the pipeline execution with sample input.
    pipeline.visualize("", [])

    # Run the SQL summary generation with sample input.
    async_validate(lambda: pipeline.run("", []))

    # Flush the Langfuse context to ensure logging and tracking are complete.
    langfuse_context.flush()

