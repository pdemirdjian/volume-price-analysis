import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from mcp.types import TextContent

from volume_price_analysis.server import handle_call_tool


async def main():
    print("--- COMPREHENSIVE ANALYSIS ---")
    try:
        results = await handle_call_tool("comprehensive_analysis", {"symbol": "SPY"})
        for content in results:
            if isinstance(content, TextContent):
                print(content.text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- OPTIONS ANALYSIS ---")
    try:
        results = await handle_call_tool("options_analysis", {"symbol": "SPY"})
        for content in results:
            if isinstance(content, TextContent):
                print(content.text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
