import logging
import sys
from pathlib import Path
from typing import Any, List

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

# Import core components such as engine, pipeline, and provider
from src.core.engine import Engine
from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider

# Import common functions and constants used in SQL generation
from src.pipelines.common import (
    TEXT_TO_SQL_RULES,
    SQLGenPostProcessor,
    sql_generation_system_prompt,
)
from src.utils import async_timer, timer
from src.web.v1.services.ask import AskHistory

# Set up logging for debugging
logger = logging.getLogger("wren-ai-service")

# Template for generating SQL queries from follow-up questions, using previous query context
text_to_sql_with_followup_user_prompt_template = """
### TASK ###
Given the following user's follow-up question and previous SQL query and summary,
generate at most 3 SQL queries in order to interpret the user's question in various plausible ways.

### DATABASE SCHEMA ###
{% for document in documents %}
    {{ document }}
{% endfor %}

### EXAMPLES ###

Example 1
[INPUT]
Previous SQL Summary: A query to find the number of employees in each department.
Previous SQL Query: SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department;
User's Question: How do I modify this to only show departments with more than 10 employees?

[OUTPUT]
{
    "results": [
        {
            "sql": "SELECT department, COUNT() as employee_count FROM employees GROUP BY department HAVING COUNT() > 10"
        },
        {
            "sql": "SELECT department FROM employees GROUP BY department HAVING COUNT() > 10"
        },
        {
            "sql": "SELECT department, COUNT() as employee_count FROM employees WHERE department IN (SELECT department FROM employees GROUP BY department HAVING COUNT(*) > 10)"
        }
    ]
}

Example 2
[INPUT]
Previous SQL Summary: A query to retrieve the total sales per product.
Previous SQL Query: SELECT product_id, SUM(sales) as total_sales FROM sales GROUP BY product_id;
User's Question: Can you adjust this to include the product name as well?

[OUTPUT]
{
    "results": [
        {
            "sql": "SELECT products.name, SUM(sales.sales) as total_sales FROM sales JOIN products ON sales.product_id = products.id GROUP BY products.name"
        },
        {
            "sql": "SELECT p.name, s.total_sales FROM (SELECT product_id, SUM(sales) as total_sales FROM sales GROUP BY product_id) s JOIN products p ON s.product_id = p.id"
        },
        {
            "sql": "SELECT p.name, IFNULL(SUM(s.sales), 0) as total_sales FROM products p LEFT JOIN sales s ON p.id = s.product_id GROUP BY p.name"
        }
    ]
}

Example 3
[INPUT]
Previous SQL Summary: Query to find the highest salary in each department.
Previous SQL Query: SELECT department_id, MAX(salary) as highest_salary FROM employees GROUP BY department_id;
User's Question: What if I want to see the employee names with the highest salary in each department?

[OUTPUT]
{
    "results": [
        {
            "sql": "SELECT department_id, employee_name, salary FROM employees WHERE (department_id, salary) IN (SELECT department_id, MAX(salary) FROM employees GROUP BY department_id)"
        },
        {
            "sql": "SELECT e.department_id, e.employee_name, e.salary FROM employees e INNER JOIN (SELECT department_id, MAX(salary) as max_salary FROM employees GROUP BY department_id) d ON e.department_id = d.department_id AND e.salary = d.max_salary"
        },
        {
            "sql": "WITH MaxSalaries AS (SELECT department_id, MAX(salary) as max_salary FROM employees GROUP BY department_id) SELECT e.department_id, e.employee_name, e.salary FROM employees e JOIN MaxSalaries m ON e.department_id = m.department_id AND e.salary = m.max_salary"
        }
    ]
}

### FINAL ANSWER FORMAT ###
The final answer must be the JSON format like following:

{
    "results": [
        {"sql": <SQL_QUERY_STRING_1>},
        {"sql": <SQL_QUERY_STRING_2>},
        {"sql": <SQL_QUERY_STRING_3>}
    ]
}

{{ alert }}

### QUESTION ###
Previous SQL Summary: {{ history.summary }}
Previous SQL Query: {{ history.sql }}
User's Follow-up Question: {{ query }}

Let's think step by step.
"""


## Start of Pipeline


## Pipeline starts here

# This function generates a prompt for SQL generation using the given inputs
@timer
@observe(capture_input=False)  # Observes the prompt generation without capturing input data
def prompt(
    query: str,
    documents: List[str],
    history: AskHistory,
    alert: str,
    prompt_builder: PromptBuilder,
) -> dict:
    """
    Builds the final prompt for the follow-up SQL generation task.
    
    Args:
        query (str): User's follow-up question.
        documents (List[str]): Database schema or documents.
        history (AskHistory): Previous SQL query and its summary.
        alert (str): Text-to-SQL rules or alerts to include.
        prompt_builder (PromptBuilder): Component to build the prompt.
    
    Returns:
        dict: The generated prompt.
    """
    logger.debug(f"query: {query}")
    logger.debug(f"documents: {documents}")
    logger.debug(f"history: {history}")
    return prompt_builder.run(
        query=query, documents=documents, history=history, alert=alert
    )

# This function asynchronously generates SQL queries based on the provided prompt
@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_in_followup(prompt: dict, generator: Any) -> dict:
    """
    Asynchronously generates SQL queries based on the prompt.
    
    Args:
        prompt (dict): The prompt generated for SQL generation.
        generator (Any): The model component responsible for generating SQL.
    
    Returns:
        dict: The generated SQL queries in a dictionary.
    """
    logger.debug(f"prompt: {orjson.dumps(prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=prompt.get("prompt"))

# This function post-processes the generated SQL queries to refine or validate them
@async_timer
@observe(capture_input=False)
async def post_process(
    generate_sql_in_followup: dict,
    post_processor: SQLGenPostProcessor,
    project_id: str | None = None,
) -> dict:
    """
    Post-processes the generated SQL queries for refinement or validation.
    
    Args:
        generate_sql_in_followup (dict): The SQL queries generated.
        post_processor (SQLGenPostProcessor): Component responsible for post-processing.
        project_id (str | None): Optional project ID for tracking.
    
    Returns:
        dict: The post-processed SQL queries.
    """
    logger.debug(
        f"generate_sql_in_followup: {orjson.dumps(generate_sql_in_followup, option=orjson.OPT_INDENT_2).decode()}"
    )
    return await post_processor.run(
        generate_sql_in_followup.get("replies"), project_id=project_id
    )

## End of pipeline

# This class implements the follow-up SQL generation pipeline
class FollowUpSQLGeneration(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        engine: Engine,
    ):
        """
        Initializes the Follow-Up SQL Generation pipeline with necessary components.
        
        Args:
            llm_provider (LLMProvider): Provider for the language model used in generation.
            engine (Engine): Engine for executing or validating SQL queries.
        """
        # Set up the components required for SQL generation and post-processing
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_generation_system_prompt
            ),
            "prompt_builder": PromptBuilder(
                template=text_to_sql_with_followup_user_prompt_template
            ),
            "post_processor": SQLGenPostProcessor(engine=engine),
        }

        # Configuration for the pipeline
        self._configs = {
            "alert": TEXT_TO_SQL_RULES,
        }

        # Initialize the base pipeline class with an async driver
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    # This method visualizes the pipeline execution process
    def visualize(
        self,
        query: str,
        contexts: List[str],
        history: AskHistory,
        project_id: str | None = None,
    ) -> None:
        """
        Visualizes the execution of the pipeline using a DOT file format.
        
        Args:
            query (str): The follow-up query from the user.
            contexts (List[str]): Database schema or relevant documents.
            history (AskHistory): Previous SQL query history.
            project_id (str | None): Optional project ID for tracking.
        """
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/followup_sql_generation.dot",
            inputs={
                "query": query,
                "documents": contexts,
                "history": history,
                "project_id": project_id,
                **self._components,
                **self._configs,
            },
            show_legend=True,
            orient="LR",
        )

    # This method runs the entire follow-up SQL generation pipeline asynchronously
    @async_timer
    @observe(name="Follow-Up SQL Generation")
    async def run(
        self,
        query: str,
        contexts: List[str],
        history: AskHistory,
        project_id: str | None = None,
    ):
        """
        Executes the follow-up SQL generation pipeline.
        
        Args:
            query (str): The follow-up query from the user.
            contexts (List[str]): Database schema or relevant documents.
            history (AskHistory): Previous SQL query history.
            project_id (str | None): Optional project ID for tracking.
        
        Returns:
            dict: The final SQL queries generated by the pipeline.
        """
        logger.info("Follow-Up SQL Generation pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "query": query,
                "documents": contexts,
                "history": history,
                "project_id": project_id,
                **self._components,
                **self._configs,
            },
        )

# Main execution block for testing and debugging
if __name__ == "__main__":
    from langfuse.decorators import langfuse_context
    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    # Initialize environment variables, logging, and providers
    load_env_vars()
    init_langfuse()

    # Initialize the language model provider and engine
    llm_provider, _, _, engine = init_providers(engine_config=EngineConfig())
    pipeline = FollowUpSQLGeneration(llm_provider=llm_provider, engine=engine)

    # Visualize the pipeline for testing
    pipeline.visualize(
        "this is a test query",
        [],
        AskHistory(sql="SELECT * FROM table", summary="Summary", steps=[]),
    )
    
    # Run the pipeline with a test follow-up query
    async_validate(
        lambda: pipeline.run(
            "this is a test query",
            [],
            AskHistory(sql="SELECT * FROM table", summary="Summary", steps=[]),
        )
    )

    # Flush the context for observability
    langfuse_context.flush()
