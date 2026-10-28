#!/usr/bin/env python3
"""Call MCP tool handlers in-process, without an MCP client.

Run with: uv run python run_analysis_local.py
"""

import asyncio

from mcp.types import TextContent

from volume_price_analysis.server import handle_call_tool


async def main() -> None:
    print("--- COMPREHENSIVE ANALYSIS ---")
    try:
        result = await handle_call_tool("comprehensive_analysis", {"symbol": "SPY"})
        for content in result.content:
            if isinstance(content, TextContent):
                print(content.text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- OPTIONS ANALYSIS ---")
    try:
        result = await handle_call_tool("options_analysis", {"symbol": "SPY"})
        for content in result.content:
            if isinstance(content, TextContent):
                print(content.text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
