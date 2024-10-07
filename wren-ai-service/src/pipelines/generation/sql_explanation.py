import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.utils import async_timer, timer
from src.web.v1.services.sql_explanation import StepWithAnalysisResult

logger = logging.getLogger("wren-ai-service")


sql_explanation_system_prompt = """
### INSTRUCTIONS ###

Given the question, sql query, sql analysis result to the sql query, sql query summary for reference,
please explain sql analysis result within 20 words in layman term based on sql query:
1. how does the expression work
2. why this expression is given based on the question
3. why can it answer user's question
The sql analysis will be one of the types: selectItems, relation, filter, groupByKeys, sortings

### ALERT ###

1. There must be only one type of sql analysis result in the input(sql analysis result) and output(sql explanation)
2. The number of the sql explanation must be the same as the number of the <expression_string> in the input

### INPUT STRUCTURE ###

{
  "selectItems": {
    "withFunctionCallOrMathematicalOperation": [
      {
        "alias": "<alias_string>",
        "expression": "<expression_string>"
      }
    ],
    "withoutFunctionCallOrMathematicalOperation": [
      {
        "alias": "<alias_string>",
        "expression": "<expression_string>"
      }
    ]
  }
} | {
  "relation": [
    {
      "type": "INNER_JOIN" | "LEFT_JOIN" | "RIGHT_JOIN" | "FULL_JOIN" | "CROSS_JOIN" | "IMPLICIT_JOIN"
      "criteria": <criteria_string>,
      "exprSources": [
        {
          "expression": <expression_string>,
          "sourceDataset": <sourceDataset_string>
        }...
      ]
    } | {
      "type": "TABLE",
      "alias": "<alias_string>",
      "tableName": "<expression_string>"
    }
  ]
} | {
  "filter": <expression_string>
} | {
  "groupByKeys": [<expression_string>, ...]
} | {
  "sortings": [<expression_string>, ...]
}


### OUTPUT STRUCTURE ###

Please generate the output with the following JSON format depending on the type of the sql analysis result:

{
  "results": {
    "selectItems": {
      "withFunctionCallOrMathematicalOperation": [
        <explanation1_string>,
        <explanation2_string>,
      ],
      "withoutFunctionCallOrMathematicalOperation": [
        <explanation1_string>,
        <explanation2_string>,
      ]
    }
  }
} | {
  "results": {
    "groupByKeys|sortings|relation|filter": [
      <explanation1_string>,
      <explanation2_string>,
      ...
    ]
  }
}
"""

sql_explanation_user_prompt_template = """
Question: {{ question }}
SQL query: {{ sql }}
SQL query summary: {{ sql_summary }}
SQL query analysis: {{ sql_analysis_result }}

Let's think step by step.
"""


def _compose_sql_expression_of_filter_type(
    filter_analysis: Dict, top: bool = True
) -> Dict:
    """
    Composes and processes SQL filter-type expressions (such as WHERE conditions).
    Handles both logical expressions (e.g., AND, OR) and regular expressions (EXPR).
    
    Args:
    - filter_analysis (Dict): Dictionary representing the filter SQL expression. 
                              Contains details like the type of filter (EXPR, AND, OR) and its components.
    - top (bool): If True, treats the expression as the top-level filter; 
                 if False, it processes nested expressions (e.g., part of an AND/OR).

    Returns:
    - Dict: Dictionary with two keys:
        - "values": The composed SQL filter expression (as a string).
        - "id": The unique identifier for the filter expression (if present).
    """
    
    # Handle single expression types (e.g., a basic condition like age > 18)
    if filter_analysis["type"] == "EXPR":
        if top:
            # If it's the top-level expression, wrap it in a dictionary with "values" and "id"
            return {
                "values": filter_analysis["node"],  # The actual SQL condition
                "id": filter_analysis.get("id", ""),  # The unique identifier for this expression (if available)
            }
        else:
            # If not the top-level expression, return the expression node directly
            return filter_analysis["node"]

    # Handle logical operators like AND/OR by recursively processing left and right expressions
    elif filter_analysis["type"] in ("AND", "OR"):
        # Recursively compose the left-hand side of the logical expression
        left_expr = _compose_sql_expression_of_filter_type(
            filter_analysis["left"], top=False
        )
        # Recursively compose the right-hand side of the logical expression
        right_expr = _compose_sql_expression_of_filter_type(
            filter_analysis["right"], top=False
        )
        # Combine the left and right expressions with the logical operator (AND/OR)
        return {
            "values": f"{left_expr} {filter_analysis['type']} {right_expr}",
            "id": filter_analysis.get("id", ""),  # Attach the unique identifier if available
        }

    # Return an empty dictionary with default values if no matching expression type is found
    return {"values": "", "id": ""}


def _compose_sql_expression_of_groupby_type(
    groupby_keys: List[List[dict]],
) -> List[str]:
    """
    Composes SQL expressions for GROUP BY keys.
    Iterates over a list of group-by key expressions and formats them as SQL statements.

    Args:
    - groupby_keys (List[List[dict]]): A nested list where each inner list contains dictionaries 
                                       representing the GROUP BY expressions.

    Returns:
    - List[str]: A list of dictionaries, each containing:
        - "values": The actual GROUP BY SQL expression.
        - "id": The unique identifier for each GROUP BY expression (if available).
    """
    
    # Iterate over the nested list structure to extract and compose SQL GROUP BY expressions
    return [
        {
            "values": groupby_key["expression"],  # The GROUP BY expression (e.g., column names or expressions).
            "id": groupby_key.get("id", ""),  # The unique identifier for this GROUP BY key (if available).
        }
        # Loops over each group of groupby keys and each key within that group
        for groupby_key_list in groupby_keys
        for groupby_key in groupby_key_list
    ]


def _compose_sql_expression_of_relation_type(relation: Dict) -> List[str]:
    """
    Composes SQL expressions for relation-type components such as tables and joins. 
    It processes simple table relations as well as complex join operations (INNER, LEFT, RIGHT, etc.).
    The function avoids processing subqueries.

    Args:
    - relation (Dict): A dictionary representing the relation in the SQL query, such as a table or a join.

    Returns:
    - List[str]: A list of SQL expressions for the relations (tables or joins) involved.
    """

    def _is_subquery_or_has_subquery_child(relation):
        """
        Determines whether the current relation is a subquery or has a child relation that is a subquery.
        This function is used to avoid processing subqueries when composing the relations.

        Args:
        - relation (Dict): A dictionary representing the current relation in the SQL query.

        Returns:
        - bool: True if the relation is a subquery or involves a subquery; False otherwise.
        """
        if relation["type"] == "SUBQUERY":
            return True
        if relation["type"].endswith("_JOIN"):
            # Check if either side of the JOIN is a subquery.
            if (
                relation["left"]["type"] == "SUBQUERY"
                or relation["right"]["type"] == "SUBQUERY"
            ):
                return True
        return False

    def _collect_relations(relation, result, top_level: bool = True):
        """
        Recursively collects relations (e.g., tables or joins) from the SQL query.
        This function skips any subqueries and focuses on collecting table relations or joins.

        Args:
        - relation (Dict): The current relation (table or join) being processed.
        - result (List[str]): The list of SQL expressions being composed.
        - top_level (bool): If True, only collects top-level relations; otherwise, processes nested joins.
        """
        if _is_subquery_or_has_subquery_child(relation):
            return  # Skip subqueries and their children.

        # Process simple table relations at the top level.
        if relation["type"] == "TABLE" and top_level:
            result.append(
                {
                    "values": {
                        "type": relation["type"],  # The type of relation (TABLE).
                        "tableName": relation["tableName"],  # The table name in the relation.
                    },
                    "id": relation.get("id", ""),  # The unique identifier for the relation (if available).
                }
            )
        # Process JOIN operations (INNER, LEFT, RIGHT, etc.).
        elif relation["type"].endswith("_JOIN"):
            result.append(
                {
                    "values": {
                        "type": relation["type"],  # The type of JOIN (e.g., INNER_JOIN).
                        "criteria": relation["criteria"],  # The JOIN criteria (ON conditions).
                        "exprSources": [
                            {
                                "expression": expr_source["expression"],  # The JOIN expression (e.g., column comparison).
                                "sourceDataset": expr_source["sourceDataset"],  # The source dataset (table) for the JOIN.
                            }
                            for expr_source in relation["exprSources"]
                        ],
                    },
                    "id": relation.get("id", ""),  # The unique identifier for the JOIN (if available).
                }
            )
            # Recursively collect relations on both sides of the JOIN.
            _collect_relations(relation["left"], result, top_level=False)
            _collect_relations(relation["right"], result, top_level=False)

    results = []
    # Print the relation (for debugging or logging purposes).
    print(f"relation: {relation}")
    
    # Start collecting relations by calling _collect_relations.
    _collect_relations(relation, results)
    
    return results  # Return the composed list of relations.

def _compose_sql_expression_of_select_type(select_items: List[Dict]) -> Dict:
    """
    Composes SQL expressions for SELECT items in a query.
    Differentiates between expressions that involve function calls or mathematical operations 
    and those that do not, categorizing them accordingly.

    Args:
    - select_items (List[Dict]): A list of dictionaries, each representing a SELECT item 
                                 in a SQL query. Each item contains properties such as 
                                 whether it involves a function call or mathematical operation.

    Returns:
    - Dict: A dictionary with two keys:
        - "withFunctionCallOrMathematicalOperation": A list of SQL expressions (with aliases) 
                                                     that involve functions or mathematical operations.
        - "withoutFunctionCallOrMathematicalOperation": A list of SQL expressions (with aliases) 
                                                        that do not involve any functions or mathematical operations.
    """
    # Initialize result dictionary to categorize select items based on whether they include
    # function calls or mathematical operations.
    result = {
        "withFunctionCallOrMathematicalOperation": [],  # Stores expressions with functions or math operations.
        "withoutFunctionCallOrMathematicalOperation": [],  # Stores expressions without such operations.
    }

    # Iterate through each select item in the provided list.
    for select_item in select_items:
        # Check if the SELECT item involves either a function call or a mathematical operation.
        if (
            select_item["properties"]["includeFunctionCall"] == "true"
            or select_item["properties"]["includeMathematicalOperation"] == "true"
        ):
            # Add the SELECT item to the "withFunctionCallOrMathematicalOperation" list.
            result["withFunctionCallOrMathematicalOperation"].append(
                {
                    "values": {
                        "alias": select_item["alias"],  # The alias used in the SQL query for this item.
                        "expression": select_item["expression"],  # The actual SQL expression.
                    },
                    "id": select_item.get("id", ""),  # The unique identifier for this item (if available).
                }
            )
        else:
            # Add the SELECT item to the "withoutFunctionCallOrMathematicalOperation" list.
            result["withoutFunctionCallOrMathematicalOperation"].append(
                {
                    "values": {
                        "alias": select_item["alias"],  # The alias used in the SQL query for this item.
                        "expression": select_item["expression"],  # The actual SQL expression.
                    },
                    "id": select_item.get("id", ""),  # The unique identifier for this item (if available).
                }
            )

    # Return the result dictionary categorizing the SELECT items based on their properties.
    return result



def _compose_sql_expression_of_sortings_type(sortings: List[Dict]) -> List[str]:
    """
    Composes SQL expressions for sorting (ORDER BY) based on the provided sorting details.

    Args:
    - sortings (List[Dict]): A list of dictionaries representing sorting expressions. 
                             Each dictionary contains the expression to sort by and the ordering direction (ASC/DESC).

    Returns:
    - List[str]: A list of dictionaries, each containing:
        - "values": A string representing the sorting expression (e.g., "column_name ASC").
        - "id": The unique identifier for the sorting expression (if available).
    """
    return [
        {
            # Compose the sorting expression as 'expression ordering' (e.g., 'price DESC').
            "values": f'{sorting["expression"]} {sorting["ordering"]}',
            # Attach the unique identifier for the sorting expression, if provided.
            "id": sorting.get("id", ""),
        }
        for sorting in sortings  # Iterate over the list of sorting expressions.
    ]
def _extract_to_str(data):
    """
    Extracts a string from the provided data. If the data is a list, 
    it returns the first element in the list. If the data is a string, 
    it returns the string itself.

    Args:
    - data: Can be either a list of strings or a string.

    Returns:
    - str: The extracted string. If the input is a list, it returns the first element;
           if it's a string, it returns the string. If neither, it returns an empty string.
    """
    if isinstance(data, list) and data:
        # If the data is a non-empty list, return the first element in the list.
        return data[0]
    elif isinstance(data, str):
        # If the data is already a string, return the string as it is.
        return data

    # Return an empty string if the data is neither a non-empty list nor a string.
    return ""



@component
class SQLAnalysisPreprocessor:
    """
    A class responsible for preprocessing the results of SQL analysis. 
    This involves decomposing various parts of the SQL query, such as filters, group by keys, relations, 
    select items, and sortings, into structured data that can be used for further processing.
    """

    @component.output_types(
        preprocessed_sql_analysis_results=List[Dict],  # Defines the output type as a list of dictionaries.
    )
    def run(
        self,
        sql_analysis_results: List[Dict],  # Input: List of dictionaries representing analyzed SQL components.
    ) -> Dict[str, List[Dict]]:
        """
        Processes and decomposes SQL analysis results, extracting components like filters, group-by keys, 
        relations, select items, and sorting criteria.

        Args:
        - sql_analysis_results (List[Dict]): A list of dictionaries representing the analysis of a SQL query. 
                                             Each dictionary contains details about different SQL components 
                                             (e.g., filters, group-by keys, etc.).

        Returns:
        - Dict[str, List[Dict]]: A dictionary with a single key, 'preprocessed_sql_analysis_results', which 
                                 contains the decomposed SQL components as a list of dictionaries.
        """
        preprocessed_sql_analysis_results = []  # Initialize an empty list to hold the processed results.

        # Iterate over each SQL analysis result.
        for sql_analysis_result in sql_analysis_results:
            # Skip processing for subqueries or common table expressions (CTEs).
            if not sql_analysis_result.get("isSubqueryOrCte", False):
                preprocessed_sql_analysis_result = {}  # Initialize an empty dictionary for the current result.

                # Process and add the 'filter' component, if it exists.
                if "filter" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "filter"
                    ] = _compose_sql_expression_of_filter_type(
                        sql_analysis_result["filter"]  # Decompose filter expressions using the helper function.
                    )
                else:
                    preprocessed_sql_analysis_result["filter"] = {}  # Empty filter if not present.

                # Process and add the 'groupByKeys' component, if it exists.
                if "groupByKeys" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "groupByKeys"
                    ] = _compose_sql_expression_of_groupby_type(
                        sql_analysis_result["groupByKeys"]  # Decompose group by keys using the helper function.
                    )
                else:
                    preprocessed_sql_analysis_result["groupByKeys"] = []  # Empty group by keys if not present.

                # Process and add the 'relation' component, if it exists.
                if "relation" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "relation"
                    ] = _compose_sql_expression_of_relation_type(
                        sql_analysis_result["relation"]  # Decompose table relations and joins using the helper function.
                    )
                else:
                    preprocessed_sql_analysis_result["relation"] = []  # Empty relations if not present.

                # Process and add the 'selectItems' component, if it exists.
                if "selectItems" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "selectItems"
                    ] = _compose_sql_expression_of_select_type(
                        sql_analysis_result["selectItems"]  # Decompose select items (columns, functions).
                    )
                else:
                    # If no select items are present, initialize empty categories for function calls and math operations.
                    preprocessed_sql_analysis_result["selectItems"] = {
                        "withFunctionCallOrMathematicalOperation": [],
                        "withoutFunctionCallOrMathematicalOperation": [],
                    }

                # Process and add the 'sortings' component, if it exists.
                if "sortings" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "sortings"
                    ] = _compose_sql_expression_of_sortings_type(
                        sql_analysis_result["sortings"]  # Decompose sorting criteria using the helper function.
                    )
                else:
                    preprocessed_sql_analysis_result["sortings"] = []  # Empty sorting criteria if not present.

                # Append the processed SQL components for the current analysis result to the results list.
                preprocessed_sql_analysis_results.append(
                    preprocessed_sql_analysis_result
                )

        # Return a dictionary containing the list of preprocessed SQL analysis results.
        return {"preprocessed_sql_analysis_results": preprocessed_sql_analysis_results}


@component
class SQLExplanationGenerationPostProcessor:
    """
    This class handles the post-processing of SQL explanation results.
    It takes generated SQL explanations and maps them back to their original 
    SQL components, such as filters, group by keys, relations, select items, and sortings, 
    creating a structured output with explanations for each component.
    """

    @component.output_types(
        results=Optional[List[Dict[str, Any]]],  # Defines the output type as an optional list of dictionaries.
    )
    def run(
        self, generates: List[List[str]], preprocessed_sql_analysis_results: List[dict]
    ) -> Dict[str, Any]:
        """
        Processes the generated SQL explanations and aligns them with the preprocessed SQL analysis components.

        Args:
        - generates (List[List[str]]): A list of generated SQL explanations from the model.
        - preprocessed_sql_analysis_results (List[dict]): Preprocessed SQL analysis results, which include components 
                                                          like filters, relations, select items, etc.

        Returns:
        - Dict[str, Any]: A dictionary containing structured explanations for each SQL component, such as filters, 
                          group by keys, relations, select items, and sorting.
        """
        results = []  # Initialize an empty list to hold the final results.

        try:
            # Only process if there are preprocessed SQL analysis results.
            if preprocessed_sql_analysis_results:
                preprocessed_sql_analysis_results = preprocessed_sql_analysis_results[0]  # Access the first result.

                # Iterate over each generated explanation.
                for generate in generates:
                    # Extract the SQL explanation results from the generated response.
                    sql_explanation_results = orjson.loads(generate["replies"][0])[
                        "results"
                    ]

                    logger.debug(
                        f"sql_explanation_results: {orjson.dumps(sql_explanation_results, option=orjson.OPT_INDENT_2).decode()}"
                    )

                    # If the 'filter' component exists in both the preprocessed results and the generated explanation.
                    if preprocessed_sql_analysis_results.get(
                        "filter", {}
                    ) and sql_explanation_results.get("filter", {}):
                        # Append the 'filter' explanation to the results list.
                        results.append(
                            {
                                "type": "filter",  # Specifies the type as 'filter'.
                                "payload": {
                                    "id": preprocessed_sql_analysis_results["filter"][
                                        "id"
                                    ],  # The ID of the filter component.
                                    "expression": preprocessed_sql_analysis_results[
                                        "filter"
                                    ]["values"],  # The filter expression.
                                    "explanation": _extract_to_str(
                                        sql_explanation_results["filter"]
                                    ),  # The explanation generated for the filter.
                                },
                            }
                        )
                    # Process 'groupByKeys' component if present.
                    elif preprocessed_sql_analysis_results.get(
                        "groupByKeys", []
                    ) and sql_explanation_results.get("groupByKeys", []):
                        for (
                            groupby_key,
                            sql_explanation,
                        ) in zip(
                            preprocessed_sql_analysis_results["groupByKeys"],
                            sql_explanation_results["groupByKeys"],
                        ):
                            # Append the group by key explanations to the results.
                            results.append(
                                {
                                    "type": "groupByKeys",  # Specifies the type as 'groupByKeys'.
                                    "payload": {
                                        "id": groupby_key["id"],  # The ID of the group by key.
                                        "expression": groupby_key["values"],  # The group by key expression.
                                        "explanation": _extract_to_str(sql_explanation),  # The explanation generated for the group by key.
                                    },
                                }
                            )
                    # Process 'relation' component if present.
                    elif preprocessed_sql_analysis_results.get(
                        "relation", []
                    ) and sql_explanation_results.get("relation", []):
                        for (
                            relation,
                            sql_explanation,
                        ) in zip(
                            preprocessed_sql_analysis_results["relation"],
                            sql_explanation_results["relation"],
                        ):
                            # Append the relation explanations to the results.
                            results.append(
                                {
                                    "type": "relation",  # Specifies the type as 'relation'.
                                    "payload": {
                                        "id": relation["id"],  # The ID of the relation component.
                                        **relation["values"],  # The relation expression values.
                                        "explanation": _extract_to_str(sql_explanation),  # The explanation generated for the relation.
                                    },
                                }
                            )
                    # Process 'selectItems' component if present.
                    elif preprocessed_sql_analysis_results.get(
                        "selectItems", {}
                    ) and sql_explanation_results.get("selectItems", {}):
                        # Handle select items with function calls or mathematical operations.
                        sql_analysis_result_for_select_items = [
                            {
                                "type": "selectItems",  # Specifies the type as 'selectItems'.
                                "payload": {
                                    "id": select_item["id"],  # The ID of the select item.
                                    **select_item["values"],  # The select item expression values.
                                    "isFunctionCallOrMathematicalOperation": True,  # Indicates the presence of a function call or mathematical operation.
                                    "explanation": _extract_to_str(sql_explanation),  # The explanation generated for the select item.
                                },
                            }
                            for select_item, sql_explanation in zip(
                                preprocessed_sql_analysis_results["selectItems"][
                                    "withFunctionCallOrMathematicalOperation"
                                ],
                                sql_explanation_results["selectItems"][
                                    "withFunctionCallOrMathematicalOperation"
                                ],
                            )
                        ] + [
                            {
                                "type": "selectItems",  # Specifies the type as 'selectItems'.
                                "payload": {
                                    "id": select_item["id"],  # The ID of the select item.
                                    **select_item["values"],  # The select item expression values.
                                    "isFunctionCallOrMathematicalOperation": False,  # Indicates no function call or mathematical operation.
                                    "explanation": _extract_to_str(sql_explanation),  # The explanation generated for the select item.
                                },
                            }
                            for select_item, sql_explanation in zip(
                                preprocessed_sql_analysis_results["selectItems"][
                                    "withoutFunctionCallOrMathematicalOperation"
                                ],
                                sql_explanation_results["selectItems"][
                                    "withoutFunctionCallOrMathematicalOperation"
                                ],
                            )
                        ]

                        # Add the results for select items to the overall results list.
                        results += sql_analysis_result_for_select_items
                    # Process 'sortings' component if present.
                    elif preprocessed_sql_analysis_results.get(
                        "sortings", []
                    ) and sql_explanation_results.get("sortings", []):
                        for (
                            sorting,
                            sql_explanation,
                        ) in zip(
                            preprocessed_sql_analysis_results["sortings"],
                            sql_explanation_results["sortings"],
                        ):
                            # Append the sorting explanations to the results.
                            results.append(
                                {
                                    "type": "sortings",  # Specifies the type as 'sortings'.
                                    "payload": {
                                        "id": sorting["id"],  # The ID of the sorting component.
                                        "expression": sorting["values"],  # The sorting expression.
                                        "explanation": _extract_to_str(sql_explanation),  # The explanation generated for the sorting.
                                    },
                                }
                            )
        # Catch and log any exceptions that occur during processing.
        except Exception as e:
            logger.exception(f"Error in SQLExplanationGenerationPostProcessor: {e}")

        # Return the final structured results containing the explanations for each SQL component.
        return {"results": results}

## Start of Pipeline
@timer
@observe(capture_input=False)
def preprocess(
    sql_analysis_results: List[dict], pre_processor: SQLAnalysisPreprocessor
) -> dict:
    """
    Preprocesses the SQL analysis results using the SQLAnalysisPreprocessor component.

    Args:
    - sql_analysis_results (List[dict]): A list of dictionaries containing the SQL analysis results. 
      Each dictionary represents components of a SQL query (e.g., filters, group by keys, etc.).
    - pre_processor (SQLAnalysisPreprocessor): An instance of the SQLAnalysisPreprocessor component 
      responsible for transforming and extracting SQL components.

    Returns:
    - dict: A dictionary containing the preprocessed SQL components, ready for further processing.
    """
    
    # Log the input SQL analysis results in a JSON format for debugging purposes.
    logger.debug(
        f"sql_analysis_results: {orjson.dumps(sql_analysis_results, option=orjson.OPT_INDENT_2).decode()}"
    )
    
    # Execute the preprocessor's 'run' method to transform the SQL analysis results.
    return pre_processor.run(sql_analysis_results)



@timer
@observe(capture_input=False)
def prompts(
    question: str,
    sql: str,
    preprocess: dict,
    sql_summary: str,
    prompt_builder: PromptBuilder,
) -> List[dict]:
    """
    This function creates prompts by combining the original user query, SQL query, and preprocessed SQL components.
    These prompts will be used to generate SQL explanations based on the analysis of the SQL query.

    Args:
    - question (str): The original user question.
    - sql (str): The SQL query that is being analyzed and explained.
    - preprocess (dict): Preprocessed SQL components, which include filters, group by keys, relations, etc.
    - sql_summary (str): A summary of the SQL query.
    - prompt_builder (PromptBuilder): A tool to build prompts for the model to generate explanations.

    Returns:
    - List[dict]: A list of dictionaries where each dictionary represents a prompt that will be passed to the model for SQL explanation.
    """

    # Log the input data for debugging purposes.
    logger.debug(f"question: {question}")
    logger.debug(f"sql: {sql}")
    logger.debug(
        f"preprocess: {orjson.dumps(preprocess, option=orjson.OPT_INDENT_2).decode()}"
    )
    logger.debug(f"sql_summary: {sql_summary}")

    # Initialize an empty list to store the preprocessed SQL analysis results that contain values.
    preprocessed_sql_analysis_results_with_values = []

    # Iterate over the preprocessed SQL analysis results.
    for preprocessed_sql_analysis_result in preprocess[
        "preprocessed_sql_analysis_results"
    ]:
        # Loop through each key-value pair in the preprocessed result (e.g., filters, relations, etc.).
        for key, value in preprocessed_sql_analysis_result.items():
            if value:  # If the value exists, proceed.
                if key != "selectItems":  # Handle non-select items.
                    if isinstance(value, list):
                        # For list-type values, collect their 'values' field and add them to the result.
                        preprocessed_sql_analysis_results_with_values.append(
                            {key: [v["values"] for v in value]}
                        )
                    else:
                        # For non-list values, directly append their 'values' field.
                        preprocessed_sql_analysis_results_with_values.append(
                            {key: value["values"]}
                        )
                else:
                    # Handle the 'selectItems' which can include function calls or mathematical operations.
                    preprocessed_sql_analysis_results_with_values.append(
                        {
                            key: {
                                "withFunctionCallOrMathematicalOperation": [
                                    v["values"]
                                    for v in value[
                                        "withFunctionCallOrMathematicalOperation"
                                    ]
                                ],
                                "withoutFunctionCallOrMathematicalOperation": [
                                    v["values"]
                                    for v in value[
                                        "withoutFunctionCallOrMathematicalOperation"
                                    ]
                                ],
                            }
                        }
                    )

    # Log the fully processed SQL analysis results with values for debugging purposes.
    logger.debug(
        f"preprocessed_sql_analysis_results_with_values: {orjson.dumps(preprocessed_sql_analysis_results_with_values, option=orjson.OPT_INDENT_2).decode()}"
    )

    # Generate and return the prompts.
    return [
        prompt_builder.run(
            question=question,  # The original user question.
            sql=sql,  # The SQL query.
            sql_analysis_result=sql_analysis_result,  # The specific SQL component being explained.
            sql_summary=sql_summary,  # A summary of the SQL query.
        )
        for sql_analysis_result in preprocessed_sql_analysis_results_with_values
    ]


@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_explanation(prompts: List[dict], generator: Any) -> List[dict]:
    """
    This asynchronous function generates SQL explanations by sending the prompts to the model for processing.

    Args:
    - prompts (List[dict]): A list of dictionaries where each dictionary represents a prompt containing SQL-related information 
      (SQL query, question, analysis result) that will be sent to the model for generating explanations.
    - generator (Any): The generator object that will handle the actual prompt execution. Typically, it refers to the model
      that generates the explanation based on the provided prompts.

    Returns:
    - List[dict]: A list of results where each result is the output generated by the model for a given prompt.
    """

    # Log the prompts for debugging purposes, making it easier to track what is being sent to the model.
    logger.debug(
        f"prompts: {orjson.dumps(prompts, option=orjson.OPT_INDENT_2).decode()}"
    )

    # Define an asynchronous task that runs the generator's 'run' method to generate explanations.
    async def _task(prompt: str, generator: Any):
        return await generator.run(prompt=prompt.get("prompt"))

    # For each prompt, create an asynchronous task and execute the model (generator) with the prompt.
    # This creates a list of asynchronous tasks.
    tasks = [_task(prompt, generator) for prompt in prompts]

    # Gather the results of all the tasks asynchronously, meaning all tasks run concurrently.
    # The output is a list of generated explanations corresponding to each prompt.
    return await asyncio.gather(*tasks)



@timer
@observe(capture_input=False)
def post_process(
    generate_sql_explanation: List[dict],
    preprocess: dict,
    post_processor: SQLExplanationGenerationPostProcessor,
) -> dict:
    """
    This function processes the generated SQL explanations and combines them with the preprocessed SQL analysis results 
    to produce a final structured output.

    Args:
    - generate_sql_explanation (List[dict]): A list of dictionaries where each dictionary contains the model's generated SQL explanations.
    - preprocess (dict): The preprocessed SQL analysis results from earlier in the pipeline.
    - post_processor (SQLExplanationGenerationPostProcessor): The post-processing component that merges the generated explanations with the preprocessed data.

    Returns:
    - dict: A dictionary containing the final processed results after post-processing.
    """

    # Log the generated SQL explanations for debugging purposes.
    logger.debug(
        f"generate_sql_explanation: {orjson.dumps(generate_sql_explanation, option=orjson.OPT_INDENT_2).decode()}"
    )

    # Log the preprocessed SQL analysis results for debugging purposes.
    logger.debug(
        f"preprocess: {orjson.dumps(preprocess, option=orjson.OPT_INDENT_2).decode()}"
    )

    # Run the post-processing step by combining the generated SQL explanations with the preprocessed analysis results.
    return post_processor.run(
        generate_sql_explanation,  # The generated explanations from the model.
        preprocess["preprocessed_sql_analysis_results"],  # The preprocessed SQL analysis results.
    )


class SQLExplanation(BasicPipeline):
    """
    SQLExplanation is a pipeline class that integrates the components needed to process, explain, and visualize SQL queries.
    It takes a question, SQL analysis results, and other components, generating human-readable SQL explanations.
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initializes the SQLExplanation pipeline.
        
        Args:
        - llm_provider (LLMProvider): A provider of the large language model (LLM) responsible for generating the SQL explanations.
        """
        # Define the components of the pipeline, including pre-processing, prompt building, generation, and post-processing.
        self._components = {
            "pre_processor": SQLAnalysisPreprocessor(),  # Preprocesses SQL analysis results before generating explanations.
            "prompt_builder": PromptBuilder(  # Builds the prompt for LLM input using a predefined template.
                template=sql_explanation_user_prompt_template
            ),
            "generator": llm_provider.get_generator(  # Generates SQL explanations using the LLM.
                system_prompt=sql_explanation_system_prompt
            ),
            "post_processor": SQLExplanationGenerationPostProcessor(),  # Post-processes the results from the LLM.
        }

        # Initialize the parent class (BasicPipeline) with asynchronous driver support.
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(self, question: str, step_with_analysis_results: StepWithAnalysisResult) -> None:
        """
        Visualizes the execution of the SQL explanation pipeline for debugging or analysis purposes.

        Args:
        - question (str): The input question related to the SQL query.
        - step_with_analysis_results (StepWithAnalysisResult): Contains the SQL query, analysis results, and summary for visualization.
        """
        destination = "outputs/pipelines/generation"  # Directory to save visualization output.
        
        # Create the directory if it doesn't exist.
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        # Visualize the pipeline execution, showing the post-processing step and linking inputs.
        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/sql_explanation.dot",  # Output file path.
            inputs={
                "question": question,
                "sql": step_with_analysis_results.sql,  # The SQL query.
                "sql_analysis_results": step_with_analysis_results.sql_analysis_results,  # The SQL analysis results.
                "sql_summary": step_with_analysis_results.summary,  # SQL query summary.
                **self._components,  # Components like pre-processing, prompt builder, generator, and post-processing.
            },
            show_legend=True,  # Display legend in the visualization.
            orient="LR",  # Set the layout orientation (Left to Right).
        )

    @async_timer
    @observe(name="SQL Explanation Generation")
    async def run(self, question: str, step_with_analysis_results: StepWithAnalysisResult):
        """
        Executes the SQL explanation pipeline asynchronously to generate SQL explanations.

        Args:
        - question (str): The input question related to the SQL query.
        - step_with_analysis_results (StepWithAnalysisResult): Contains the SQL query, analysis results, and summary.

        Returns:
        - dict: The processed results from the pipeline.
        """
        logger.info("SQL Explanation Generation pipeline is running...")

        # Execute the pipeline asynchronously and return the final result.
        return await self._pipe.execute(
            ["post_process"],  # Step to execute.
            inputs={
                "question": question,  # Input question.
                "sql": step_with_analysis_results.sql,  # SQL query.
                "sql_analysis_results": step_with_analysis_results.sql_analysis_results,  # SQL analysis results.
                "sql_summary": step_with_analysis_results.summary,  # SQL summary.
                **self._components,  # Components used during execution.
            },
        )


if __name__ == "__main__":
    """
    Main entry point for running the SQLExplanation pipeline as a script.
    Sets up the necessary environment, initializes providers, and runs the pipeline.
    """
    from langfuse.decorators import langfuse_context
    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    # Load environment variables and initialize Langfuse context for logging and tracking.
    load_env_vars()
    init_langfuse()

    # Initialize the large language model (LLM) provider and other components.
    llm_provider, _, _, _ = init_providers(EngineConfig())
    
    # Create an instance of the SQLExplanation pipeline.
    pipeline = SQLExplanation(
        llm_provider=llm_provider,
    )

    # Visualize the execution of the pipeline with test inputs.
    pipeline.visualize(
        "this is a test question",
        StepWithAnalysisResult(
            sql="xxx",
            summary="xxx",
            sql_analysis_results=[],
        ),
    )

    # Run the pipeline asynchronously with test inputs.
    async_validate(
        lambda: pipeline.run(
            "this is a test question",
            StepWithAnalysisResult(
                sql="xxx",
                summary="xxx",
                sql_analysis_results=[],
            ),
        )
    )

    # Flush the Langfuse context to ensure logs and traces are completed.
    langfuse_context.flush()
