import base64
import logging
import os
from typing import Any, Dict, Optional, Tuple

import aiohttp
import orjson

from src.core.engine import Engine, remove_limit_statement
from src.providers.loader import provider

logger = logging.getLogger("wren-ai-service")

# This file defines multiple providers for different Wren engine services (WrenUI, WrenIbis, and WrenEngine).
# These providers are responsible for executing SQL queries, each interacting with a different endpoint or service.
# The purpose of these classes is to handle various SQL operations by interacting with Wren's specific APIs.

# Provider class for WrenUI engine, which executes SQL through a GraphQL API.
@provider("wren_ui")
class WrenUI(Engine):
    def __init__(self, endpoint: str = os.getenv("WREN_UI_ENDPOINT")):
        self._endpoint = endpoint
        logger.info("Using Engine: wren_ui")

    # This method executes SQL queries asynchronously. 
    # It sends a GraphQL mutation to the Wren UI API endpoint to run or preview the SQL query.
    async def execute_sql(
        self,
        sql: str,
        session: aiohttp.ClientSession,
        project_id: str | None = None,
        dry_run: bool = True,
        **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        # Prepares the payload with the SQL query and optionally sets it to dry-run mode (with a limit of 1).
        data = {
            "sql": remove_limit_statement(sql),
            "projectId": project_id,
        }
        if dry_run:
            data["dryRun"] = True
            data["limit"] = 1
        else:
            data["limit"] = 500

        # Sends the request to the Wren UI GraphQL API for execution or preview of the SQL query.
        async with session.post(
            f"{self._endpoint}/api/graphql",
            json={
                "query": "mutation PreviewSql($data: PreviewSQLDataInput) { previewSql(data: $data) }",
                "variables": {"data": data},
            },
        ) as response:
            res = await response.json()
            if data := res.get("data"):
                return True, data, None
            return (
                False,
                None,
                res.get("errors", [{}])[0].get("message", "Unknown error"),
            )


# Provider class for WrenIbis engine, which runs SQL queries against the Wren Ibis engine using REST endpoints.
@provider("wren_ibis")
class WrenIbis(Engine):
    def __init__(
        self,
        endpoint: str = os.getenv("WREN_IBIS_ENDPOINT"),
        source: str = os.getenv("WREN_IBIS_SOURCE"),
        manifest: str = os.getenv("WREN_IBIS_MANIFEST"),
        connection_info: dict = (
            orjson.loads(base64.b64decode(os.getenv("WREN_IBIS_CONNECTION_INFO")))
            if os.getenv("WREN_IBIS_CONNECTION_INFO")
            else {}
        ),
    ):
        self._endpoint = endpoint
        self._source = source
        self._manifest = manifest
        self._connection_info = connection_info
        logger.info("Using Engine: wren_ibis")

    # Executes SQL queries asynchronously against the Wren Ibis engine. 
    # Supports dry-run and full-query execution modes.
    async def execute_sql(
        self,
        sql: str,
        session: aiohttp.ClientSession,
        dry_run: bool = True,
        **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        api_endpoint = f"{self._endpoint}/v2/connector/{self._source}/query"
        if dry_run:
            api_endpoint += "?dryRun=true&limit=1"
        else:
            api_endpoint += "?limit=500"

        async with session.post(
            api_endpoint,
            json={
                "sql": remove_limit_statement(sql),
                "manifestStr": self._manifest,
                "connectionInfo": self._connection_info,
            },
        ) as response:
            if dry_run:
                res = await response.text()
            else:
                res = await response.json()

            if response.status == 204:
                return True, None, None
            if response.status == 200:
                return True, res, None

            return False, None, res


# Provider class for WrenEngine, which provides access to Wren's main SQL execution engine.
@provider("wren_engine")
class WrenEngine(Engine):
    def __init__(self, endpoint: str = os.getenv("WREN_ENGINE_ENDPOINT")):
        self._endpoint = endpoint
        logger.info("Using Engine: wren_engine")

    # Executes SQL queries asynchronously using the Wren engine. 
    # Queries are executed in either dry-run mode or full-query mode.
    async def execute_sql(
        self,
        sql: str,
        session: aiohttp.ClientSession,
        properties: Dict[str, Any] = {
            "manifest": os.getenv("WREN_ENGINE_MANIFEST"),
        },
        dry_run: bool = True,
        **kwargs,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        # Constructs the correct API endpoint based on the type of execution (dry-run or full-run).
        api_endpoint = (
            f"{self._endpoint}/v1/mdl/dry-run"
            if dry_run
            else f"{self._endpoint}/v1/mdl/preview"
        )

        # Sends the SQL query to the API for execution.
        async with session.get(
            api_endpoint,
            json={
                "manifest": orjson.loads(base64.b64decode(properties.get("manifest")))
                if properties.get("manifest")
                else {},
                "sql": remove_limit_statement(sql),
                "limit": 1 if dry_run else 500,
            },
        ) as response:
            res = await response.json()
            if response.status == 200:
                return True, res, None

            return False, None, res
