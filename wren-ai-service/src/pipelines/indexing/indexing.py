import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from hamilton.function_modifiers import extract_fields
from haystack import Document, component
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from langfuse.decorators import observe
from tqdm import tqdm

from src.core.pipeline import BasicPipeline
from src.core.provider import DocumentStoreProvider, EmbedderProvider
from src.utils import async_timer, timer

logger = logging.getLogger("wren-ai-service")

DATASET_NAME = os.getenv("DATASET_NAME")


@component
class DocumentCleaner:
    """
    This component is responsible for clearing all documents in the provided document store(s).
    It takes a list of document stores and an optional `id` to filter documents based on the `project_id`.

    Attributes:
    - stores: List of DocumentStore objects where the documents are stored.
    """

    def __init__(self, stores: List[DocumentStore]) -> None:
        self._stores = stores

    @component.output_types(mdl=str)
    async def run(self, mdl: str, id: Optional[str] = None) -> str:
        """
        Asynchronously clears the documents from the stores based on the given `id`.
        If `id` is provided, it only deletes documents with the specified `project_id`. If no `id` is passed, all documents will be cleared.

        Args:
        - mdl (str): A model input.
        - id (Optional[str]): A unique identifier to filter the documents by `project_id`.

        Returns:
        - str: The input model `mdl` after document cleaning.
        """

        async def _clear_documents(
            store: DocumentStore, id: Optional[str] = None
        ) -> None:
            # Constructs a filter for document deletion based on `project_id`.
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
            # Deletes documents from the document store based on the filters.
            await store.delete_documents(filters)

        logger.info("Ask Indexing pipeline is clearing old documents...")
        # Concurrently clear documents from all provided stores.
        await asyncio.gather(*[_clear_documents(store, id) for store in self._stores])
        return {"mdl": mdl}


@component
class MDLValidator:
    """
    This component is used to validate the MDL (Model Definition Language) object.
    It checks if the MDL is valid JSON and ensures that the required keys (models, views, relationships, metrics) are present.

    """

    @component.output_types(mdl=Dict[str, Any])
    def run(self, mdl: str) -> Dict[str, Any]:
        """
        Validates and processes the input MDL (Model Definition Language) string.
        Ensures that the `models`, `views`, `relationships`, and `metrics` keys exist.

        Args:
        - mdl (str): A JSON string representing the MDL.

        Returns:
        - dict: The processed and validated MDL as a dictionary.
        """
        try:
            mdl_json = orjson.loads(mdl)  # Deserialize the MDL from JSON format.
            logger.debug(f"MDL JSON: {mdl_json}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        # Ensures the required keys are present in the MDL.
        if "models" not in mdl_json:
            mdl_json["models"] = []
        if "views" not in mdl_json:
            mdl_json["views"] = []
        if "relationships" not in mdl_json:
            mdl_json["relationships"] = []
        if "metrics" not in mdl_json:
            mdl_json["metrics"] = []

        return {"mdl": mdl_json}



@component
class ViewChunker:
    """
    This component converts the view MDL into a format for indexing historical queries.
    It prepares the data by chunking the MDL views into smaller pieces, which are then stored in the document store.
    
    The converted format includes:
    - "question" (user query),
    - "summary" (LLM generated description),
    - "statement" (SQL query),
    - "viewId" (ID of the view).
    """

    @component.output_types(documents=List[Document])
    def run(self, mdl: Dict[str, Any], id: Optional[str] = None) -> Dict[str, List[Document]]:
        """
        Processes the MDL and chunks the views into a suitable format for indexing.

        Args:
        - mdl (dict): The MDL that contains the views and other metadata.
        - id (Optional[str]): The project ID to be included in the metadata.

        Returns:
        - dict: A dictionary containing a list of Document objects to be indexed.
        """
        
        # Helper function to extract content from a view's historical queries and the question.
        def _get_content(view: Dict[str, Any]) -> str:
            properties = view.get("properties", {})
            historical_queries = properties.get("historical_queries", [])
            question = properties.get("question", "")

            return " ".join(historical_queries + [question])

        # Helper function to extract meta information like summary and statement from the view.
        def _get_meta(view: Dict[str, Any]) -> Dict[str, Any]:
            properties = view.get("properties", {})
            return {
                "summary": properties.get("summary", ""),
                "statement": view.get("statement", ""),
                "viewId": properties.get("viewId", ""),
            }

        # Convert each view in the MDL into the required format.
        converted_views = [
            {"content": _get_content(view), "meta": _get_meta(view)}
            for view in mdl["views"]
        ]

        # Return documents with metadata for indexing.
        return {
            "documents": [
                Document(
                    id=str(uuid.uuid4()),
                    meta={"project_id": id, **converted_view["meta"]}
                    if id
                    else {**converted_view["meta"]},
                    content=converted_view["content"],
                )
                for converted_view in tqdm(
                    converted_views,
                    desc="indexing view into the historical view question store",
                )
            ]
        }

@component
class DDLConverter:
    """
    This component is responsible for converting a given Model Definition Language (MDL) structure into Data Definition Language (DDL) commands.
    It takes the MDL structure and transforms it into a format that can be indexed into the document store as table schema definitions.

    Attributes:
    - mdl: The input Model Definition Language (MDL) object, containing models, relationships, views, and metrics.
    - column_indexing_batch_size: The batch size for indexing columns in models, which is used to divide large sets of columns for efficient processing.
    - id: An optional identifier (e.g., project_id) to associate with each document for better metadata management.

    Methods:
    - run: Converts the MDL structure into documents representing table schema definitions for storage.
    - _get_ddl_commands: Generates a list of DDL commands from the given MDL structure.
    """

    @component.output_types(documents=List[Document])
    def run(
        self,
        mdl: Dict[str, Any],
        column_indexing_batch_size: int,
        id: Optional[str] = None,
    ):
        """
        Main entry point of the component. It transforms the MDL into DDL commands and returns them as a list of documents.
        These documents are indexed into the document store, each containing metadata about table schema definitions.

        Args:
        - mdl (Dict[str, Any]): The input MDL containing models, relationships, views, and metrics.
        - column_indexing_batch_size (int): The batch size for processing columns during indexing.
        - id (Optional[str]): An optional project identifier to tag each document.

        Returns:
        - dict: A dictionary containing the documents to be indexed in the document store.
        """

        logger.info(
            "Ask Indexing pipeline is writing new documents for table schema..."
        )
        logger.debug(f"original mdl_json: {mdl}")

        # Retrieve DDL commands generated from the MDL structure.
        ddl_commands = self._get_ddl_commands(mdl, column_indexing_batch_size)

        # Create and return the documents for each DDL command, attaching appropriate metadata.
        return {
            "documents": [
                Document(
                    id=str(uuid.uuid4()),  # Assigning a unique identifier for each document.
                    meta=(
                        {
                            "project_id": id,
                            "type": "TABLE_SCHEMA",
                            "name": ddl_command["name"],
                        }
                        if id  # If a project ID is provided, it will be added to the metadata.
                        else {
                            "type": "TABLE_SCHEMA",
                            "name": ddl_command["name"],
                        }
                    ),
                    content=ddl_command["payload"],  # The DDL command payload is set as the document content.
                )
                for ddl_command in tqdm(
                    ddl_commands,
                    desc="indexing ddl commands into the dbschema store",
                )
            ]
        }

    def _get_ddl_commands(
        self, mdl: Dict[str, Any], column_indexing_batch_size: int = 50
    ) -> List[dict]:
        """
        Converts the models, relationships, views, and metrics from the MDL into DDL commands.

        Args:
        - mdl (Dict[str, Any]): The input MDL containing models, relationships, views, and metrics.
        - column_indexing_batch_size (int): The batch size for processing columns.

        Returns:
        - List[dict]: A list of DDL commands, each representing a schema definition.
        """

        # Initialize the semantics with models, relationships, views, and metrics from the MDL.
        semantics = {
            "models": [],
            "relationships": mdl["relationships"],
            "views": mdl["views"],
            "metrics": mdl["metrics"],
        }

        # Process each model in the MDL to extract columns and their properties.
        for model in mdl["models"]:
            columns = []
            for column in model.get("columns", []):
                # Construct the DDL for each column, including properties such as name, type, and optional attributes.
                ddl_column = {
                    "name": column.get("name", ""),
                    "type": column.get("type", ""),
                }
                if "properties" in column:
                    ddl_column["properties"] = column["properties"]
                if "relationship" in column:
                    ddl_column["relationship"] = column["relationship"]
                if "expression" in column:
                    ddl_column["expression"] = column["expression"]
                if "isCalculated" in column:
                    ddl_column["isCalculated"] = column["isCalculated"]

                columns.append(ddl_column)

            # Add the processed model and its columns to the semantics.
            semantics["models"].append(
                {
                    "name": model.get("name", ""),
                    "properties": model.get("properties", {}),
                    "columns": columns,
                    "primaryKey": model.get("primaryKey", ""),
                }
            )

        # Convert the models, relationships, views, and metrics into DDL commands.
        return (
            self._convert_models_and_relationships(
                semantics["models"],
                semantics["relationships"],
                column_indexing_batch_size,
            )
            + self._convert_views(semantics["views"])
            + self._convert_metrics(semantics["metrics"])
        )

    class DDLConverter:

    COLUMN_TYPE = "COLUMN"
    FOREIGN_KEY_TYPE = "FOREIGN_KEY"
    TABLE_TYPE = "TABLE"
    TABLE_COLUMNS_TYPE = "TABLE_COLUMNS"

    def _convert_models_and_relationships(
        self,
        models: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        column_indexing_batch_size: int,
    ) -> List[str]:
        """
        Converts models and relationships into DDL commands.

        Args:
        - models: List of models containing column and table information.
        - relationships: List of relationships containing foreign key constraints.
        - column_indexing_batch_size: The batch size for column indexing.

        Returns:
        - List of DDL commands.
        """
        ddl_commands = []
        primary_keys_map = self._map_primary_keys(models)

        for model in models:
            table_name = model["name"]
            columns_ddl = self._generate_column_ddls(model)
            fk_constraints = self._generate_fk_constraints(table_name, relationships, primary_keys_map)

            # Add table DDL command
            ddl_commands.append(self._generate_table_ddl(model))

            # Batch column DDL commands
            ddl_commands.extend(self._batch_columns_ddl(table_name, columns_ddl, column_indexing_batch_size))

            # Add foreign key constraints
            if fk_constraints:
                ddl_commands.extend(fk_constraints)

        return ddl_commands

    def _map_primary_keys(self, models: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Maps model names to their primary keys.

        Args:
        - models: List of model dictionaries.

        Returns:
        - Dictionary mapping model names to their primary keys.
        """
        return {model["name"]: model["primaryKey"] for model in models}

    def _generate_column_ddls(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates DDL commands for columns in the model.

        Args:
        - model: The model containing column information.

        Returns:
        - List of DDL commands for columns.
        """
        columns_ddl = []
        for column in model["columns"]:
            if "relationship" not in column:
                comment = self._generate_column_comment(column)
                columns_ddl.append({
                    "type": self.COLUMN_TYPE,
                    "comment": comment,
                    "name": column["name"],
                    "data_type": column["type"],
                    "is_primary_key": column["name"] == model["primaryKey"],
                })
        return columns_ddl

    def _generate_column_comment(self, column: Dict[str, Any]) -> str:
        """
        Generates a comment for a column.

        Args:
        - column: The column dictionary.

        Returns:
        - Comment string for the column.
        """
        comment = ""
        if "properties" in column:
            column_properties = {
                "alias": column["properties"].pop("displayName", ""),
                "description": column["properties"].pop("description", ""),
            }
            comment += f"-- {orjson.dumps(column_properties).decode('utf-8')}\n  "
        if "isCalculated" in column and column["isCalculated"]:
            comment += f"-- This column is a Calculated Field\n  -- column expression: {column['expression']}\n  "
        return comment

    def _generate_fk_constraints(
        self, table_name: str, relationships: List[Dict[str, Any]], primary_keys_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Generates foreign key constraints based on relationships.

        Args:
        - table_name: The name of the current table.
        - relationships: List of relationship dictionaries.
        - primary_keys_map: Dictionary mapping model names to their primary keys.

        Returns:
        - List of DDL commands for foreign keys.
        """
        fk_constraints = []
        for relationship in relationships:
            condition = relationship.get("condition", "")
            join_type = relationship.get("joinType", "")
            models = relationship.get("models", [])

            if len(models) == 2:
                fk_constraint = self._determine_fk_constraint(table_name, models, condition, join_type, primary_keys_map)
                if fk_constraint:
                    fk_constraints.append({
                        "type": self.FOREIGN_KEY_TYPE,
                        "comment": f'-- {{"condition": {condition}, "joinType": {join_type}}}\n  ',
                        "constraint": fk_constraint,
                        "tables": models,
                    })
        return fk_constraints

    def _determine_fk_constraint(self, table_name: str, models: List[str], condition: str, join_type: str, primary_keys_map: Dict[str, str]) -> Optional[str]:
        """
        Determines the foreign key constraint for the table.

        Args:
        - table_name: The name of the current table.
        - models: List of models in the relationship.
        - condition: The join condition.
        - join_type: The type of join.
        - primary_keys_map: Dictionary mapping model names to their primary keys.

        Returns:
        - Foreign key constraint string or None if not applicable.
        """
        if join_type.upper() == "MANY_TO_ONE" and table_name == models[0]:
            related_table = models[1]
            fk_column = condition.split(" = ")[0].split(".")[1]
            return f"FOREIGN KEY ({fk_column}) REFERENCES {related_table}({primary_keys_map[related_table]})"
        elif join_type.upper() == "ONE_TO_MANY" and table_name == models[1]:
            related_table = models[0]
            fk_column = condition.split(" = ")[1].split(".")[1]
            return f"FOREIGN KEY ({fk_column}) REFERENCES {related_table}({primary_keys_map[related_table]})"
        elif join_type.upper() == "ONE_TO_ONE":
            index = models.index(table_name)
            related_table = [m for m in models if m != table_name][0]
            fk_column = condition.split(" = ")[index].split(".")[1]
            return f"FOREIGN KEY ({fk_column}) REFERENCES {related_table}({primary_keys_map[related_table]})"
        return None

    def _generate_table_ddl(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the DDL command for a table.

        Args:
        - model: The model dictionary.

        Returns:
        - DDL command for the table.
        """
        if "properties" in model:
            model_properties = {
                "alias": model["properties"].pop("displayName", ""),
                "description": model["properties"].pop("description", ""),
            }
            comment = f"\n/* {orjson.dumps(model_properties).decode('utf-8')} */\n"
        else:
            comment = ""

        return {
            "name": model["name"],
            "payload": str({
                "type": self.TABLE_TYPE,
                "comment": comment,
                "name": model["name"],
            }),
        }

    def _batch_columns_ddl(self, table_name: str, columns_ddl: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """
        Batches the column DDL commands for efficient processing.

        Args:
        - table_name: The name of the table.
        - columns_ddl: List of column DDL commands.
        - batch_size: The batch size for processing columns.

        Returns:
        - List of batched column DDL commands.
        """
        return [
            {
                "name": table_name,
                "payload": str({
                    "type": self.TABLE_COLUMNS_TYPE,
                    "columns": columns_ddl[i : i + batch_size],
                }),
            }
            for i in range(0, len(columns_ddl), batch_size)
        ]
    def _convert_views(self, views: List[Dict[str, Any]]) -> List[str]:
        def _format(view: Dict[str, Any]) -> dict:
            # Formats the view into a structured dictionary containing the type, name, and statement
            return {
                "type": "VIEW",
                "comment": f"/* {view['properties']} */\n" if "properties" in view else "",
                "name": view["name"],
                "statement": view["statement"],
            }

        # Converts each view into a DDL command (payload) and returns the list
        return [{"name": view["name"], "payload": str(_format(view))} for view in views]


    def _convert_metrics(self, metrics: List[Dict[str, Any]]) -> List[str]:
        ddl_commands = []

        for metric in metrics:
            table_name = metric.get("name", "")  # Retrieve the metric's name
            columns_ddl = []  # To hold columns (dimensions and measures)

            # Process dimensions
            for dimension in metric.get("dimension", []):
                comment = "-- This column is a dimension\n  "
                name = dimension.get("name", "")
                columns_ddl.append(
                    {
                        "type": "COLUMN",
                        "comment": comment,
                        "name": name,
                        "data_type": dimension.get("type", ""),
                    }
                )

            # Process measures
            for measure in metric.get("measure", []):
                comment = f"-- This column is a measure\n  -- expression: {measure['expression']}\n  "
                name = measure.get("name", "")
                columns_ddl.append(
                    {
                        "type": "COLUMN",
                        "comment": comment,
                        "name": name,
                        "data_type": measure.get("type", ""),
                    }
                )

            # Add a table-level comment and DDL command for the metric
            comment = f"\n/* This table is a metric */\n/* Metric Base Object: {metric['baseObject']} */\n"
            ddl_commands.append(
                {
                    "name": table_name,
                    "payload": str(
                        {
                            "type": "METRIC",
                            "comment": comment,
                            "name": table_name,
                            "columns": columns_ddl,
                        }
                    ),
                }
            )

        return ddl_commands



@component
class TableDescriptionConverter:
    @component.output_types(documents=List[Document])
    def run(self, mdl: Dict[str, Any], id: Optional[str] = None):
        logger.info("Ask Indexing pipeline is writing new documents for table descriptions...")

        logger.debug(f"original mdl_json: {mdl}")

        table_descriptions = self._get_table_descriptions(mdl)

        # Convert the table descriptions into documents for indexing
        return {
            "documents": [
                Document(
                    id=str(uuid.uuid4()),
                    meta={"project_id": id, "type": "TABLE_DESCRIPTION"} if id else {"type": "TABLE_DESCRIPTION"},
                    content=table_description,
                )
                for table_description in tqdm(
                    table_descriptions,
                    desc="indexing table descriptions into the table description store",
                )
            ]
        }


    def _get_table_descriptions(self, mdl: Dict[str, Any]) -> List[str]:
        """
        Extracts table descriptions from the provided MDL (Model Definition Language) structure.
        
        Parameters:
        - mdl (Dict[str, Any]): The MDL structure containing models, metrics, and views.

        Returns:
        - List[str]: A list of formatted table description strings including the name, mdl_type (MODEL, METRIC, VIEW), 
        and description for each table found in the MDL.
        """
        
        table_descriptions = []
        mdl_data = [
            {
                "mdl_type": "MODEL",
                "payload": mdl["models"],
            },
            {
                "mdl_type": "METRIC",
                "payload": mdl["metrics"],
            },
            {
                "mdl_type": "VIEW",
                "payload": mdl["views"],
            },
        ]

        # Loop through each data type (MODEL, METRIC, VIEW) and extract relevant information
        for data in mdl_data:
            payload = data["payload"]
            for unit in payload:
                # Extract name and description if available
                if name := unit.get("name", ""):
                    table_description = {
                        "name": name,
                        "mdl_type": data["mdl_type"],
                        "description": unit.get("properties", {}).get("description", "")
                    }
                    # Append the formatted string description
                    table_descriptions.append(str(table_description))

        return table_descriptions



@component
class AsyncDocumentWriter(DocumentWriter):
    @component.output_types(documents_written=int)
    async def run(
        self, documents: List[Document], policy: Optional[DuplicatePolicy] = None
    ):
        """
        Asynchronously writes documents to the document store with an optional duplicate handling policy.

        Parameters:
        - documents (List[Document]): A list of documents to be written.
        - policy (Optional[DuplicatePolicy]): A policy that dictates how to handle duplicate documents. If None, 
          the default policy is used.

        Returns:
        - int: The number of documents successfully written to the store.
        """
        if policy is None:
            policy = self.policy

        # Write documents to the document store and return the number written
        documents_written = await self.document_store.write_documents(
            documents=documents, policy=policy
        )
        return {"documents_written": documents_written}

        


## Start of Pipeline
@async_timer
@observe(capture_input=False, capture_output=False)
async def clean_document_store(
    mdl_str: str, cleaner: DocumentCleaner, id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Cleans the document store by removing old documents related to the given project.

    Parameters:
    - mdl_str (str): The MDL data in string format.
    - cleaner (DocumentCleaner): An instance of DocumentCleaner used to delete documents.
    - id (Optional[str]): An optional project ID to filter the documents being cleaned.

    Returns:
    - Dict[str, Any]: The result from the cleaner containing any relevant information about the cleanup process.
    """
    logger.debug(f"input in clean_document_store: {mdl_str}")
    return await cleaner.run(mdl=mdl_str, id=id)



@timer
@observe(capture_input=False, capture_output=False)
@extract_fields(dict(mdl=Dict[str, Any]))
def validate_mdl(
    clean_document_store: Dict[str, Any], validator: MDLValidator
) -> Dict[str, Any]:
    """
    Validates the MDL (Model Definition Language) JSON using the MDLValidator.

    Parameters:
    - clean_document_store (Dict[str, Any]): Contains the 'mdl' field with the MDL data to validate.
    - validator (MDLValidator): The validator used to ensure the MDL has the correct format and required fields.

    Returns:
    - Dict[str, Any]: A dictionary containing the validated MDL.
    """
    logger.debug(
        f"input in validate_mdl: {orjson.dumps(clean_document_store, option=orjson.OPT_INDENT_2).decode()}"
    )
    mdl = clean_document_store.get("mdl")  # Extract the 'mdl' from the input dictionary
    res = validator.run(mdl=mdl)  # Run the MDLValidator
    return dict(mdl=res["mdl"])  # Return the validated MDL


@timer
@observe(capture_input=False)
def covert_to_table_descriptions(
    mdl: Dict[str, Any],
    table_description_converter: TableDescriptionConverter,
    id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts the MDL data into table descriptions using the TableDescriptionConverter.

    Parameters:
    - mdl (Dict[str, Any]): The MDL data that contains information about the models, views, and metrics.
    - table_description_converter (TableDescriptionConverter): The component responsible for converting the MDL to table descriptions.
    - id (Optional[str]): An optional project ID used for indexing table descriptions.

    Returns:
    - Dict[str, Any]: A dictionary containing the table descriptions generated.
    """
    logger.debug(
        f"input in convert_to_table_descriptions: {orjson.dumps(mdl, option=orjson.OPT_INDENT_2).decode()}"
    )
    return table_description_converter.run(mdl=mdl, id=id)  # Convert and return table descriptions


@async_timer
@observe(capture_input=False, capture_output=False)
async def embed_table_descriptions(
    covert_to_table_descriptions: Dict[str, Any],
    document_embedder: Any,
) -> Dict[str, Any]:
    """
    Embeds the table descriptions using a document embedder.

    Parameters:
    - covert_to_table_descriptions (Dict[str, Any]): The table descriptions that need to be embedded.
    - document_embedder (Any): The embedder used to generate embeddings for the table descriptions.

    Returns:
    - Dict[str, Any]: A dictionary containing the embedded table descriptions.
    """
    logger.debug(
        f"input(covert_to_table_descriptions) in embed_table_descriptions: {orjson.dumps(covert_to_table_descriptions, option=orjson.OPT_INDENT_2).decode()}"
    )

    # Perform the embedding operation on the 'documents' field in the input dictionary
    return await document_embedder.run(covert_to_table_descriptions["documents"])


@async_timer
@observe(capture_input=False)
async def write_table_description(
    embed_table_descriptions: Dict[str, Any], table_description_writer: DocumentWriter
) -> None:
    """
    Writes the embedded table descriptions to the document store.

    Parameters:
    - embed_table_descriptions (Dict[str, Any]): The embedded table descriptions.
    - table_description_writer (DocumentWriter): The document writer component used to write the descriptions to the store.
    """
    return await table_description_writer.run(
        documents=embed_table_descriptions["documents"]
    )


@timer
@observe(capture_input=False)
def convert_to_ddl(
    mdl: Dict[str, Any],
    ddl_converter: DDLConverter,
    column_indexing_batch_size: int,
    id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts the MDL into DDL commands (Data Definition Language) using a converter.

    Parameters:
    - mdl (Dict[str, Any]): The MDL data.
    - ddl_converter (DDLConverter): The component responsible for converting the MDL into DDL.
    - column_indexing_batch_size (int): The batch size for column indexing.
    - id (Optional[str]): Optional project ID for context.

    Returns:
    - Dict[str, Any]: The DDL commands generated.
    """
    logger.debug(
        f"input in convert_to_ddl: {orjson.dumps(mdl, option=orjson.OPT_INDENT_2).decode()}"
    )
    return ddl_converter.run(
        mdl=mdl,
        column_indexing_batch_size=column_indexing_batch_size,
        id=id,
    )


@async_timer
@observe(capture_input=False, capture_output=False)
async def embed_dbschema(
    convert_to_ddl: Dict[str, Any],
    document_embedder: Any,
) -> Dict[str, Any]:
    """
    Embeds the database schema (DDL) using the provided document embedder.

    Parameters:
    - convert_to_ddl (Dict[str, Any]): The converted DDL commands.
    - document_embedder (Any): The embedder component used to embed the documents.

    Returns:
    - Dict[str, Any]: The embedded database schema documents.
    """
    logger.debug(
        f"input(convert_to_ddl) in embed_dbschema: {orjson.dumps(convert_to_ddl, option=orjson.OPT_INDENT_2).decode()}"
    )
    return await document_embedder.run(documents=convert_to_ddl["documents"])


@async_timer
@observe(capture_input=False)
async def write_dbschema(
    embed_dbschema: Dict[str, Any], dbschema_writer: DocumentWriter
) -> None:
    """
    Writes the embedded database schema documents to the document store.

    Parameters:
    - embed_dbschema (Dict[str, Any]): The embedded database schema documents.
    - dbschema_writer (DocumentWriter): The writer component used to save the schema to the store.
    """
    return await dbschema_writer.run(documents=embed_dbschema["documents"])


@timer
@observe(capture_input=False)
def view_chunk(
    mdl: Dict[str, Any], view_chunker: ViewChunker, id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Chunks the view data from the MDL for processing.

    Parameters:
    - mdl (Dict[str, Any]): The MDL data.
    - view_chunker (ViewChunker): The component responsible for chunking the views.
    - id (Optional[str]): Optional project ID for context.

    Returns:
    - Dict[str, Any]: The chunked view data.
    """
    logger.debug(
        f"input in view_chunk: {orjson.dumps(mdl, option=orjson.OPT_INDENT_2).decode()}"
    )
    return view_chunker.run(mdl=mdl, id=id)


@async_timer
@observe(capture_input=False, capture_output=False)
async def embed_view(
    view_chunk: Dict[str, Any], document_embedder: Any
) -> Dict[str, Any]:
    """
    Embeds the chunked view data using the provided document embedder.

    Parameters:
    - view_chunk (Dict[str, Any]): The chunked view data.
    - document_embedder (Any): The embedder component used to embed the view data.

    Returns:
    - Dict[str, Any]: The embedded view documents.
    """
    logger.debug(
        f"input in embed_view: {orjson.dumps(view_chunk, option=orjson.OPT_INDENT_2).decode()}"
    )
    return await document_embedder.run(documents=view_chunk["documents"])


@async_timer
@observe(capture_input=False)
async def write_view(embed_view: Dict[str, Any], view_writer: DocumentWriter) -> None:
    """
    Writes the embedded view documents to the document store.

    Parameters:
    - embed_view (Dict[str, Any]): The embedded view documents.
    - view_writer (DocumentWriter): The writer component used to save the view data to the store.
    """
    return await view_writer.run(documents=embed_view["documents"])


## End of Pipeline

class Indexing(BasicPipeline):
    def __init__(
        self,
        embedder_provider: EmbedderProvider,
        document_store_provider: DocumentStoreProvider,
        column_indexing_batch_size: Optional[int] = 50,
    ) -> None:
        """
        Initializes the Indexing pipeline that handles embedding and storing
        database schemas, table descriptions, and views.

        Parameters:
        - embedder_provider (EmbedderProvider): Provider for document embedding.
        - document_store_provider (DocumentStoreProvider): Provider for document storage.
        - column_indexing_batch_size (Optional[int]): Batch size for column indexing (default: 50).
        """
        dbschema_store = document_store_provider.get_store()
        view_store = document_store_provider.get_store(dataset_name="view_questions")
        table_description_store = document_store_provider.get_store(
            dataset_name="table_descriptions"
        )

        self._components = {
            "cleaner": DocumentCleaner(
                [dbschema_store, view_store, table_description_store]
            ),
            "validator": MDLValidator(),
            "document_embedder": embedder_provider.get_document_embedder(),
            "ddl_converter": DDLConverter(),
            "table_description_converter": TableDescriptionConverter(),
            "dbschema_writer": AsyncDocumentWriter(
                document_store=dbschema_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
            "view_chunker": ViewChunker(),
            "view_writer": AsyncDocumentWriter(
                document_store=view_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
            "table_description_writer": AsyncDocumentWriter(
                document_store=table_description_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        }

        self._configs = {
            "column_indexing_batch_size": column_indexing_batch_size,
        }

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )


def visualize(self, mdl_str: str, id: Optional[str] = None) -> None:
    """
    Visualizes the pipeline execution for document indexing, showing the processes for writing dbschema, views, and table descriptions.

    Parameters:
    - mdl_str (str): The model string input.
    - id (Optional[str]): The optional identifier for the document store.

    Returns:
    - None: It generates a .dot file that visually represents the pipeline's execution.
    """
    destination = "outputs/pipelines/indexing"
    if not Path(destination).exists():
        Path(destination).mkdir(parents=True, exist_ok=True)

    self._pipe.visualize_execution(
        ["write_dbschema", "write_view", "write_table_description"],
        output_file_path=f"{destination}/indexing.dot",
        inputs={
            "mdl_str": mdl_str,
            "id": id,
            **self._components,
            **self._configs,
        },
        show_legend=True,
        orient="LR",
    )

@async_timer
@observe(name="Document Indexing")
async def run(self, mdl_str: str, id: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the document indexing pipeline for dbschema, views, and table descriptions.

    Parameters:
    - mdl_str (str): The model string input.
    - id (Optional[str]): The optional identifier for the document store.

    Returns:
    - Dict[str, Any]: The execution results of the indexing pipeline.
    """
    logger.info("Document Indexing pipeline is running...")
    return await self._pipe.execute(
        ["write_dbschema", "write_view", "write_table_description"],
        inputs={
            "mdl_str": mdl_str,
            "id": id,
            **self._components,
            **self._configs,
        },
    )

if __name__ == "__main__":
    """
    Initializes the Indexing pipeline and runs the visualization and execution based on the input model data.
    """
    from langfuse.decorators import langfuse_context
    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    load_env_vars()
    init_langfuse()

    _, embedder_provider, document_store_provider, _ = init_providers(EngineConfig())

    pipeline = Indexing(
        embedder_provider=embedder_provider,
        document_store_provider=document_store_provider,
    )

    input = '{"models": [], "views": [], "relationships": [], "metrics": []}'
    pipeline.visualize(input)
    async_validate(lambda: pipeline.run(input))

    langfuse_context.flush()
