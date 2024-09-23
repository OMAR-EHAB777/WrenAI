import logging
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple

import aiohttp
import sqlglot
from pydantic import BaseModel

# Set up logger for the service
logger = logging.getLogger("wren-ai-service")

class EngineConfig(BaseModel):
    """
    EngineConfig defines the configuration for the Engine.
    - provider: Specifies the provider (e.g., "wren_ui").
    - config: Stores additional configuration details.
    """
    provider: str = "wren_ui"
    config: dict = {}


class Engine(metaclass=ABCMeta):
    """
    Abstract Engine class defining an interface for SQL execution.
    Subclasses must implement the `execute_sql` method.
    """
    @abstractmethod
    async def execute_sql(
        self,
        sql: str,
        session: aiohttp.ClientSession,
        dry_run: bool = True,
        **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Executes the provided SQL query asynchronously.
        :param sql: The SQL query to execute.
        :param session: An aiohttp session used for async HTTP requests.
        :param dry_run: If True, the SQL will not be executed, only simulated.
        :return: Tuple indicating success (bool) and optional result data.
        """
        ...


def clean_generation_result(result: str) -> str:
    """
    Cleans the generated SQL result by normalizing whitespace and removing 
    unnecessary SQL markers such as comments, quote markers, or new lines.
    :param result: The raw generated SQL string.
    :return: The cleaned SQL string.
    """
    def _normalize_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    return (
        _normalize_whitespace(result)
        .replace("\\n", " ")
        .replace("```sql", "")
        .replace('"""', "")
        .replace("'''", "")
        .replace("```", "")
        .replace(";", "")
    )


def remove_limit_statement(sql: str) -> str:
    """
    Removes the LIMIT statement from a SQL query, if present.
    :param sql: SQL query string with an optional LIMIT clause.
    :return: Modified SQL string without the LIMIT clause.
    """
    pattern = r"\s*LIMIT\s+\d+(\s*;?\s*--.*|\s*;?\s*)$"
    modified_sql = re.sub(pattern, "", sql, flags=re.IGNORECASE)

    return modified_sql


def add_quotes(sql: str) -> Tuple[str, bool]:
    """
    Adds quotes to SQL identifiers (such as table or column names) using the sqlglot library.
    Transpiles SQL into the Trino SQL dialect with quoted identifiers.
    :param sql: Original SQL query string.
    :return: Tuple containing quoted SQL and a success flag.
    """
    try:
        logger.debug(f"Original SQL: {sql}")

        quoted_sql = sqlglot.transpile(sql, read="trino", identify=True)[0]

        logger.debug(f"Quoted SQL: {quoted_sql}")
    except Exception as e:
        logger.exception(f"Error in sqlglot.transpile to {sql}: {e}")

        return "", False

    return quoted_sql, True
