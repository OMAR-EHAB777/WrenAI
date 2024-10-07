import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.engine import Engine
from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.pipelines.common import SQLBreakdownGenPostProcessor
from src.utils import async_timer, timer
from src.web.v1.services.sql_regeneration import (
    SQLExplanationWithUserCorrections,
)

logger = logging.getLogger("wren-ai-service")


sql_regeneration_system_prompt = """
### Instructions ###

- Given a list of user corrections, regenerate the corresponding SQL query.
- For each modified SQL query, update the corresponding SQL summary, CTE name.
- If subsequent steps are dependent on the corrected step, make sure to update the SQL query, SQL summary and CTE name in subsequent steps if needed.
- Regenerate the description after correcting all of the steps.

### INPUT STRUCTURE ###

{
    "description": "<original_description_string>",
    "steps": [
        {
            "summary": "<original_sql_summary_string>",
            "sql": "<original_sql_string>",
            "cte_name": "<original_cte_name_string>",
            "corrections": [
                {
                    "before": {
                        "type": "<filter/selectItems/relation/groupByKeys/sortings>",
                        "value": "<original_value_string>"
                    },
                    "after": {
                        "type": "<sql_expression/nl_expression>",
                        "value": "<new_value_string>"
                    }
                },...
            ]
        },...
    ]
}

### OUTPUT STRUCTURE ###

Generate modified results according to the following in JSON format:

{
    "description": "<modified_description_string>",
    "steps": [
        {
            "summary": "<modified_sql_summary_string>",
            "sql": "<modified_sql_string>",
            "cte_name": "<modified_cte_name_string>",
        },...
    ]
}
"""

sql_regeneration_user_prompt_template = """
inputs: {{ results }}

Let's think step by step.
"""

@component
class SQLRegenerationPreprocesser:
    @component.output_types(
        results=Dict[str, Any],  # The output will be a dictionary with string keys and any type of values.
    )
    def run(
        self,
        description: str,
        steps: List[SQLExplanationWithUserCorrections],  # List of steps with user corrections for SQL regeneration.
    ) -> Dict[str, Any]:
        """
        Prepares the input for the SQL regeneration process by formatting the 
        description and steps provided by the user into a dictionary.

        Args:
        - description (str): The description of the SQL task.
        - steps (List[SQLExplanationWithUserCorrections]): List of steps containing user corrections for SQL.

        Returns:
        - dict: A dictionary containing the description and steps, which is later used by the prompt.
        """
        return {
            "results": {
                "description": description,
                "steps": steps,
            }
        }


## Start of Pipeline

@timer
@observe(capture_input=False)
def preprocess(
    description: str,
    steps: List[SQLExplanationWithUserCorrections],
    preprocesser: SQLRegenerationPreprocesser,
) -> dict[str, Any]:
    """
    This function triggers the preprocessing step, where the user-provided description and 
    steps are processed into the expected format for further handling.

    Args:
    - description (str): The description of the task.
    - steps (List[SQLExplanationWithUserCorrections]): List of steps with user corrections.

    Returns:
    - dict: The preprocessed data that is passed forward in the pipeline.
    """
    logger.debug(f"steps: {steps}")
    logger.debug(f"description: {description}")
    return preprocesser.run(description=description, steps=steps)


@timer
@observe(capture_input=False)
def sql_regeneration_prompt(
    preprocess: Dict[str, Any],
    prompt_builder: PromptBuilder,
) -> dict:
    """
    Builds the SQL regeneration prompt that will be sent to the LLM for SQL query generation.

    Args:
    - preprocess (Dict[str, Any]): The preprocessed data containing the description and steps.
    - prompt_builder (PromptBuilder): The component responsible for creating the structured prompt.

    Returns:
    - dict: The structured prompt ready to be sent to the LLM for SQL regeneration.
    """
    logger.debug(f"preprocess: {preprocess}")
    return prompt_builder.run(results=preprocess["results"])


@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_regeneration(
    sql_regeneration_prompt: dict,
    generator: Any,
) -> dict:
    """
    Sends the SQL regeneration prompt to the LLM generator asynchronously and retrieves the generated SQL query.

    Args:
    - sql_regeneration_prompt (dict): The structured prompt for SQL regeneration.
    - generator (Any): The LLM generator responsible for producing the SQL query.

    Returns:
    - dict: The generated SQL query.
    """
    logger.debug(f"sql_regeneration_prompt: {orjson.dumps(sql_regeneration_prompt, option=orjson.OPT_INDENT_2).decode()}")
    return await generator.run(prompt=sql_regeneration_prompt.get("prompt"))


@async_timer
@observe(capture_input=False)
async def sql_regeneration_post_process(
    generate_sql_regeneration: dict,
    post_processor: SQLBreakdownGenPostProcessor,
    project_id: str | None = None,
) -> dict:
    """
    Post-processes the SQL query generated by the LLM to ensure it adheres to the necessary syntax 
    and formatting rules.

    Args:
    - generate_sql_regeneration (dict): The generated SQL from the LLM.
    - post_processor (SQLBreakdownGenPostProcessor): The post-processor responsible for cleaning up the SQL.
    - project_id (str | None): Optional project ID for tracking.

    Returns:
    - dict: The cleaned and processed SQL output.
    """
    logger.debug(f"generate_sql_regeneration: {orjson.dumps(generate_sql_regeneration, option=orjson.OPT_INDENT_2).decode()}")
    return await post_processor.run(
        replies=generate_sql_regeneration.get("replies"),
        project_id=project_id,
    )


## End of Pipeline


class SQLRegeneration(BasicPipeline):
    """
    SQLRegeneration is the main class defining the pipeline for regenerating SQL queries based on 
    user corrections and LLM output. It handles preprocessing, prompt generation, SQL generation, 
    and post-processing.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,  # Provider of the large language model (LLM)
        engine: Engine,  # SQL engine to execute the queries
    ):
        # Components used in the pipeline (LLM generator, prompt builder, post-processor)
        self._components = {
            "preprocesser": SQLRegenerationPreprocesser(),  # Preprocesses inputs for SQL regeneration
            "prompt_builder": PromptBuilder(
                template=sql_regeneration_user_prompt_template  # Uses predefined template for prompt generation
            ),
            "generator": llm_provider.get_generator(
                system_prompt=sql_regeneration_system_prompt  # SQL regeneration system prompt
            ),
            "post_processor": SQLBreakdownGenPostProcessor(engine=engine),  # Post-processor for SQL
        }

        # Call parent constructor to set up asynchronous processing
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(
        self,
        description: str,
        steps: List[SQLExplanationWithUserCorrections],
        project_id: str | None = None,
    ) -> None:
        """
        Visualizes the execution of the pipeline and outputs a .dot file for analysis.

        Args:
        - description (str): Description of the SQL task.
        - steps (List[SQLExplanationWithUserCorrections]): Steps for SQL corrections.
        - project_id (str | None): Optional project identifier.
        """
        destination = "outputs/pipelines/generation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["sql_regeneration_post_process"],  # Step to visualize
            output_file_path=f"{destination}/sql_regeneration.dot",  # Output path for visualization
            inputs={
                "description": description,
                "steps": steps,
                "project_id": project_id,
                **self._components,
            },
            show_legend=True,
            orient="LR",  # Left-to-right orientation for the diagram
        )

    @async_timer
    @observe(name="SQL-Regeneration Generation")
    async def run(
        self,
        description: str,
        steps: List[SQLExplanationWithUserCorrections],
        project_id: str | None = None,
    ):
        """
        Executes the SQL regeneration pipeline asynchronously and returns the regenerated SQL.

        Args:
        - description (str): Task description.
        - steps (List[SQLExplanationWithUserCorrections]): List of corrections for SQL regeneration.
        - project_id (str | None): Optional project identifier.

        Returns:
        - dict: The regenerated SQL query.
        """
        logger.info("SQL Regeneration Generation pipeline is running...")
        return await self._pipe.execute(
            ["sql_regeneration_post_process"],  # Step to execute
            inputs={
                "description": description,
                "steps": steps,
                "project_id": project_id,
                **self._components,
            },
        )


if __name__ == "__main__":
    """
    Main block for testing the SQLRegeneration pipeline. 
    Initializes environment and components, and runs the pipeline with sample input.
    """
    from langfuse.decorators import langfuse_context
    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    # Load environment variables
    load_env_vars()
    init_langfuse()

    # Initialize providers and components
    llm_provider, _, _, engine = init_providers(EngineConfig())
    
    # Instantiate the SQLRegeneration pipeline
    pipeline = SQLRegeneration(
        llm_provider=llm_provider,
        engine=engine,
    )

    # Visualize the pipeline execution with sample input
    pipeline.visualize("This is a description", [])

    # Run the SQL regeneration process with sample input
    async_validate(lambda: pipeline.run("This is a description", []))

    # Flush the Langfuse context to ensure logging and tracking are complete
    langfuse_context.flush()
