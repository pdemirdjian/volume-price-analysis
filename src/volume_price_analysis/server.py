"""MCP Server for Volume-Price Analysis.

This module is only the MCP adapter: it advertises the tool registry, routes
one call to one registry record, and formats every result -- success or
failure -- through a single ``CallToolResult`` path. Tool behaviour lives in
:mod:`volume_price_analysis.tools`.
"""

import asyncio
import json
import logging
import math
from importlib.metadata import PackageNotFoundError, version

from mcp.server import NotificationOptions, Server, ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from .data_fetcher import DataSource, get_default_data_source
from .tools import TOOLS, TOOLS_BY_NAME, ToolContext

logger = logging.getLogger(__name__)

try:
    _SERVER_VERSION = version("volume-price-analysis-mcp")
except PackageNotFoundError:
    _SERVER_VERSION = "0.0.0"


def _sanitize_for_json(obj: object) -> object:
    """Recursively replace NaN/Infinity floats with None for RFC 8259 compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _json_response(result: object) -> str:
    """Serialize a tool result to JSON, converting NaN/Infinity to null."""
    return json.dumps(_sanitize_for_json(result), indent=2, default=str)


async def handle_list_tools() -> list[Tool]:
    """List available volume-price analysis tools, derived from the registry."""
    return [
        Tool(name=spec.name, description=spec.description, input_schema=spec.input_schema)
        for spec in TOOLS
    ]


async def dispatch(
    name: str,
    arguments: dict,
    *,
    data_source: DataSource | None = None,
) -> CallToolResult:
    """Run one registry tool and format its result.

    Args:
        name: MCP tool name; must be a key of ``TOOLS_BY_NAME``.
        arguments: Raw MCP arguments, passed through to the tool's context.
        data_source: Market-data seam. Defaults to the production
            ``DataSource`` when not injected; tests inject an in-memory one.
    """
    logger.info("Tool called: %s", name)
    logger.debug("Tool arguments: %s", arguments)

    try:
        spec = TOOLS_BY_NAME.get(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")

        ctx = ToolContext(
            args=arguments,
            data_source=data_source if data_source is not None else get_default_data_source(),
            default_period=spec.default_period,
        )
        result = await spec.run(ctx)

        return CallToolResult(content=[TextContent(type="text", text=_json_response(result))])

    except ValueError as e:
        logger.warning("Tool %s validation error: %s", name, str(e))
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))],
            is_error=True,
        )
    except Exception as e:
        logger.error("Tool %s failed: %s", name, str(e), exc_info=True)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"error": "An internal error occurred"}, indent=2),
                )
            ],
            is_error=True,
        )


async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool execution requests against the production data source."""
    return await dispatch(name, arguments)


async def _on_list_tools(
    _ctx: ServerRequestContext, _params: PaginatedRequestParams | None
) -> ListToolsResult:
    """MCP v2 list_tools handler wrapping handle_list_tools()."""
    return ListToolsResult(tools=await handle_list_tools())


async def _on_call_tool(
    _ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    """MCP v2 call_tool handler wrapping handle_call_tool()."""
    return await handle_call_tool(params.name, params.arguments or {})


server = Server(
    "volume-price-analysis",
    version=_SERVER_VERSION,
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="volume-price-analysis",
                server_version=_SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def cli() -> None:
    """Synchronous entry point for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
