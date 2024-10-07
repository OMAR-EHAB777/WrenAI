import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.engine import Engine
from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.pipelines.common import (
    TEXT_TO_SQL_RULES,
    SQLGenPostProcessor,
    sql_generation_system_prompt,
)
from src.utils import async_timer, timer

logger = logging.getLogger("wren-ai-service")


sql_generation_user_prompt_template = """
### TASK ###
Given a user query that is ambiguous in nature, your task is to interpret the query in various plausible ways and
generate three SQL statements that could potentially answer each interpreted version of the queries.
Provide three different interpretations and corresponding SQL queries that reflect these interpretations.
Ensure that your SQL queries are diverse, covering a range of possible meanings behind the ambiguous query.

### EXAMPLES ###
Consider the structure of a generic database which includes common tables like users, orders, products, and transactions.
Here are the ambiguous user queries:

1. "Find the records of recent high-value transactions."
2. "Show me popular items that are not selling well."
3. "Retrieve user feedback on products from last month."

For each query, start by explaining the different ways the query can be interpreted. Then, provide SQL queries corresponding to each interpretation.
Your SQL statements should include SELECT statements, appropriate WHERE clauses to filter the results, and JOINs if necessary to combine information from different tables.
Remember to include ordering and limit clauses where relevant to address the 'recent', 'high-value', 'popular', and 'last month' aspects of the queries.

Example for the first query:

Interpretation 1: Recent high-value transactions are defined as transactions that occurred in the last 30 days with a value greater than $10,000.
SQL Query 1: SELECT * FROM "transactions" WHERE "transaction_date" >= NOW() - INTERVAL '30 days' AND "value" > 10000 ORDER BY "transaction_date" DESC;
SUMMARY 1: Recent high-value transactions.

Interpretation 2: High-value transactions are those in the top "10%" of all transactions in terms of value, and 'recent' is defined as the last 3 months.
SQL Query 2: WITH "ranked_transactions" AS (SELECT *, NTILE(10) OVER (ORDER BY "value" DESC) AS "percentile_rank" FROM "transactions" WHERE "transaction_date" >= NOW() - INTERVAL '3 months') SELECT * FROM "ranked_transactions" WHERE "percentile_rank" = 1 ORDER BY "transaction_date" DESC;
SUMMARY 2: Top 10% transactions last 3 months.

Interpretation 3: 'Recent' refers to the last week, and 'high-value' transactions are those above the average transaction value of the past week.
SQL Query 3: SELECT * FROM "transactions" WHERE "transaction_date" >= NOW() - INTERVAL '7 days' AND "value" > (SELECT AVG("value") FROM "transactions" WHERE "transaction_date" >= NOW() - INTERVAL '7 days') ORDER BY "transaction_date" DESC;
SUMMARY 3: Above-average transactions last week.

Proceed in a similar manner for the other queries.

### DATABASE SCHEMA ###
{% for document in documents %}
    {{ document }}
{% endfor %}

### EXCLUDED STATEMETS ###
Ensure that the following excluded statements are not used in the generated queries to maintain variety and avoid repetition.
{% for doc in exclude %}
    {{ doc.statement }}
{% endfor %}

{{ alert }}

### FINAL ANSWER FORMAT ###
The final answer must be the JSON format like following:

{
    "results": [
        {"sql": <SQL_QUERY_STRING_1>},
        {"sql": <SQL_QUERY_STRING_2>},
        {"sql": <SQL_QUERY_STRING_3>}
    ]
}

### QUESTION ###
{{ query }}

Let's think step by step.
"""


## Start of Pipeline

@timer  # Timing the execution of this function
@observe(capture_input=False)  # Observer to track inputs (without capturing them)
def prompt(
    query: str, 
    documents: List[str], 
    exclude: List[Dict], 
    alert: str, 
    prompt_builder: PromptBuilder
) -> dict:
    """
    Generates the prompt to be sent to the LLM for SQL generation.

    Args:
    - query (str): The natural language input/query from the user.
    - documents (List[str]): Contextual documents or database schema related to the query.
    - exclude (List[Dict]): SQL queries or elements to exclude during generation.
    - alert (str): Alerts or specific rules (e.g., for SQL syntax) to be followed.
    - prompt_builder (PromptBuilder): The component responsible for creating the prompt format.

    Returns:
    - dict: The generated prompt that will be sent to the LLM.
    """
    logger.debug(f"query: {query}")
    logger.debug(f"documents: {documents}")
    logger.debug(f"exclude: {orjson.dumps(exclude, option=orjson.OPT_INDENT_2).decode()}")
    
    # Build and return the prompt for LLM
    return prompt_builder.run(query=query, documents=documents, exclude=exclude, alert=alert)


@async_timer  # Timing for asynchronous operations
@observe(as_type="generation", capture_input=False)  # Observer for LLM generation
async def generate_sql(prompt: dict, generator: Any) -> dict:
    """
    Sends the generated prompt to the LLM to generate the SQL query.

    Args:
    - prompt (dict): The formatted prompt from the 'prompt' function.
    - generator (Any): The LLM generator responsible for creating the SQL query.

    Returns:
    - dict: The generated SQL query from the LLM.
    """
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    
    # Send the prompt to the LLM generator and return the generated SQL
    return await generator.run(prompt=prompt.get("prompt"))


@async_timer
@observe(capture_input=False)
async def post_process(
    generate_sql: dict,
    post_processor: SQLGenPostProcessor,
    project_id: str | None = None
) -> dict:
    """
    Post-processes the SQL generation output, applying any necessary cleanup or transformations.

    Args:
    - generate_sql (dict): The raw SQL generated by the LLM.
    - post_processor (SQLGenPostProcessor): The component to clean or modify the generated SQL.
    - project_id (str | None): Optional project identifier.

    Returns:
    - dict: The cleaned and processed SQL output.
    """
    logger.debug(f"generate_sql: {orjson.dumps(generate_sql, option=orjson.OPT_INDENT_2).decode()}")
    
    # Run the post-processor and return the final SQL output
    return await post_processor.run(generate_sql.get("replies"), project_id=project_id)


## End of Pipeline


class SQLGeneration(BasicPipeline):
    """
    The SQLGeneration class defines a pipeline to generate SQL queries from natural language inputs
    using an LLM provider, prompt builder, and post-processor.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,  # Provider of the large language model (LLM)
        engine: Engine  # Engine to execute the generated SQL
    ):
        # Define the components: LLM generator, prompt builder, and post-processor.
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_generation_system_prompt  # LLM system prompt for SQL generation
            ),
            "prompt_builder": PromptBuilder(  # Builds the user prompt
                template=sql_generation_user_prompt_template  # Predefined template
            ),
            "post_processor": SQLGenPostProcessor(engine=engine),  # Cleans up the SQL output
        }

        # Configuration options such as rules for SQL generation.
        self._configs = {
            "alert": TEXT_TO_SQL_RULES,
        }

        # Initialize the parent BasicPipeline class with asynchronous driver support.
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(
        self,
        query: str,
        contexts: List[str],
        exclude: List[Dict],
        project_id: str | None = None
    ) -> None:
        """
        Visualizes the pipeline's execution and generates a .dot file for analysis.

        Args:
        - query (str): The user's natural language query.
        - contexts (List[str]): Additional contexts such as database schema.
        - exclude (List[Dict]): SQL queries or clauses to exclude.
        - project_id (str | None): Optional project identifier.
        """
        destination = "outputs/pipelines/generation"
        
        # Create the output directory if it doesn't exist.
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        # Visualize the execution of the pipeline.
        self._pipe.visualize_execution(
            ["post_process"],  # The step to visualize.
            output_file_path=f"{destination}/sql_generation.dot",  # Output path.
            inputs={
                "query": query,
                "documents": contexts,
                "exclude": exclude,
                "project_id": project_id,
                **self._components,  # Pipeline components used.
                **self._configs,  # Configuration settings.
            },
            show_legend=True,  # Include a legend in the output.
            orient="LR",  # Layout direction (left-to-right).
        )

    @async_timer
    @observe(name="SQL Generation")
    async def run(
        self,
        query: str,
        contexts: List[str],
        exclude: List[Dict],
        project_id: str | None = None
    ):
        """
        Executes the SQL generation pipeline asynchronously.

        Args:
        - query (str): User's natural language query.
        - contexts (List[str]): Contexts such as database schema.
        - exclude (List[Dict]): SQL queries or clauses to exclude.
        - project_id (str | None): Optional project identifier.

        Returns:
        - dict: The final processed SQL query.
        """
        logger.info("SQL Generation pipeline is running...")

        # Run the pipeline and return the result.
        return await self._pipe.execute(
            ["post_process"],  # Step to execute.
            inputs={
                "query": query,
                "documents": contexts,
                "exclude": exclude,
                "project_id": project_id,
                **self._components,  # Components used during execution.
                **self._configs,  # Configurations used.
            },
        )


if __name__ == "__main__":
    """
    Main block for testing the SQLGeneration pipeline.
    It initializes the environment, providers, and runs the pipeline with a test query.
    """
    from langfuse.decorators import langfuse_context
    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    # Load environment variables and initialize Langfuse for tracking.
    load_env_vars()
    init_langfuse()

    # Initialize the LLM provider and other components.
    llm_provider, _, _, engine = init_providers(engine_config=EngineConfig())
    
    # Instantiate the SQLGeneration pipeline.
    pipeline = SQLGeneration(
        llm_provider=llm_provider,
        engine=engine,
    )

    # Visualize the execution of the pipeline with a test query.
    pipeline.visualize("this is a test query", [], [])

    # Run the pipeline asynchronously with a test query.
    async_validate(lambda: pipeline.run("this is a test query", [], []))

    # Flush the Langfuse context to complete tracking and logging.
    langfuse_context.flush()
