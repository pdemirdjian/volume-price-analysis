"""Tests for the MCP tool registry and the single dispatcher built on it.

These tests drive every registry record through the public dispatcher seam
(``server.dispatch``) with an injected ``DataSource`` -- no patching of any
module-level fetch.
"""

import datetime
import json

import pandas as pd
import pytest

from volume_price_analysis.data_fetcher import DEFAULT_TIMEOUT, InMemoryDataSource
from volume_price_analysis.server import dispatch, handle_list_tools
from volume_price_analysis.tools import TOOLS, ToolContext, ToolSpec

SYMBOL = "AAPL"


def make_frame(n: int = 120) -> pd.DataFrame:
    """OHLCV frame long enough for every indicator in the registry."""
    return pd.DataFrame(
        {
            "Date": pd.date_range(start="2024-01-01", periods=n, freq="D"),
            "Open": [100.0 + i * 0.5 - 0.5 for i in range(n)],
            "High": [100.0 + i * 0.5 + 2 for i in range(n)],
            "Low": [100.0 + i * 0.5 - 2 for i in range(n)],
            "Close": [100.0 + i * 0.5 for i in range(n)],
            "Volume": [1_000_000 + i * 20_000 for i in range(n)],
        }
    )


class NonCopyingDataSource:
    """A ``DataSource`` that hands out the *same* frame object it stores.

    ``InMemoryDataSource`` returns a defensive copy, which would hide any
    mutation a tool performs. This adapter deliberately does not, so the
    immutability test observes what the tools actually did to the frame.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def fetch(
        self,
        symbol: str,
        *,
        period: str = "1mo",
        start: str | None = None,
        end: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> pd.DataFrame:
        return self.frames[symbol]

    def earnings_date(self, symbol: str) -> datetime.datetime | None:
        return None


# Minimal valid arguments per tool. scan_candidates is the one tool that does
# not take a symbol and does not use the context's lazy fetch.
MINIMAL_ARGS: dict[str, dict] = {
    "get_stock_data": {"symbol": SYMBOL},
    "calculate_obv": {"symbol": SYMBOL},
    "calculate_vwap": {"symbol": SYMBOL},
    "calculate_volume_profile": {"symbol": SYMBOL},
    "calculate_mfi": {"symbol": SYMBOL},
    "calculate_ad_line": {"symbol": SYMBOL},
    "calculate_cmf": {"symbol": SYMBOL},
    "analyze_volume_trends": {"symbol": SYMBOL},
    "comprehensive_analysis": {"symbol": SYMBOL},
    "options_analysis": {"symbol": SYMBOL},
    "scan_candidates": {"symbols": [SYMBOL]},
    "calculate_rsi_divergence": {"symbol": SYMBOL},
}


# The top-level keys each tool's payload must carry. These are the response
# contract MCP clients depend on -- a registry record that quietly stops
# emitting one of them is a breaking change, not a refactor.
REQUIRED_KEYS: dict[str, set[str]] = {
    "get_stock_data": {
        "symbol",
        "period",
        "data_points",
        "date_range",
        "latest_close",
        "latest_volume",
        "sample_data",
    },
    "calculate_obv": {
        "symbol",
        "indicator",
        "latest_obv",
        "obv_trend",
        "data_points",
        "recent_values",
    },
    "calculate_vwap": {
        "symbol",
        "indicator",
        "latest_vwap",
        "latest_close",
        "price_vs_vwap",
        "position",
        "recent_values",
    },
    "calculate_volume_profile": {
        "symbol",
        "indicator",
        "num_price_levels",
        "point_of_control",
        "poc_volume",
        "price_range",
        "profile_data",
    },
    "calculate_mfi": {"symbol", "indicator", "latest_mfi", "condition", "recent_values"},
    "calculate_ad_line": {
        "symbol",
        "indicator",
        "latest_ad_line",
        "ad_trend",
        "data_points",
        "recent_values",
    },
    "calculate_cmf": {"symbol", "indicator", "latest_cmf", "condition", "recent_values"},
    "analyze_volume_trends": {
        "symbol",
        "analysis",
        "current_volume",
        "average_volume",
        "volume_vs_average",
        "volume_trend",
        "price_direction",
        "divergence_detected",
        "divergence_type",
    },
    "comprehensive_analysis": {
        "symbol",
        "analysis_type",
        "period",
        "latest_price",
        "headline",
        "volume_indicators",
        "price_indicators",
        "volatility_indicators",
        "volume_profile",
        "volume_trends",
        "summary",
    },
    "options_analysis": {
        "symbol",
        "analysis_type",
        "period",
        "latest_price",
        "headline",
        "parameters",
        "trend_analysis",
        "volatility_analysis",
        "volume_indicators",
        "price_indicators",
        "volume_profile",
        "volume_trends",
        "options_insights",
        "composite_signal",
        "time_decay",
    },
    "scan_candidates": {
        "scan_parameters",
        "summary",
        "top_bullish",
        "top_bearish",
        "high_conviction_setups",
        "errors",
    },
    "calculate_rsi_divergence": {
        "symbol",
        "rsi",
        "condition",
        "period",
        "bullish_divergence",
        "bearish_divergence",
        "divergence_type",
        "signal",
        "interpretation",
        "current_rsi",
    },
}


class TestRegistryShape:
    """The registry is the single source of truth for the tool list."""

    def test_registry_lists_twelve_tools(self):
        assert len(TOOLS) == 12
        assert all(isinstance(spec, ToolSpec) for spec in TOOLS)

    def test_registry_names_are_unique(self):
        names = [spec.name for spec in TOOLS]
        assert len(set(names)) == len(names)

    @pytest.mark.asyncio
    async def test_list_tools_names_match_registry_in_order(self):
        """The list-tools response is derived from the registry, order included."""
        tools = await handle_list_tools()
        assert [t.name for t in tools] == [spec.name for spec in TOOLS]
        assert [t.description for t in tools] == [spec.description for spec in TOOLS]
        assert [t.input_schema for t in tools] == [spec.input_schema for spec in TOOLS]

    def test_every_registry_tool_has_minimal_args(self):
        """Guard: MINIMAL_ARGS must cover the registry so the sweep is complete."""
        assert set(MINIMAL_ARGS) == {spec.name for spec in TOOLS}

    def test_every_registry_tool_has_required_keys(self):
        """Guard: REQUIRED_KEYS must cover the registry so the sweep is complete."""
        assert set(REQUIRED_KEYS) == {spec.name for spec in TOOLS}


class TestDispatchEveryTool:
    """A parametrized sweep of every registry record through the dispatcher."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spec", TOOLS, ids=lambda s: s.name)
    async def test_dispatch_returns_well_formed_json(self, spec):
        source = InMemoryDataSource({SYMBOL: make_frame()})

        result = await dispatch(spec.name, MINIMAL_ARGS[spec.name], data_source=source)

        assert result.is_error in (None, False), result.content[0].text
        assert len(result.content) == 1
        payload = json.loads(result.content[0].text)
        assert isinstance(payload, dict)
        assert "error" not in payload
        missing = REQUIRED_KEYS[spec.name] - set(payload)
        assert not missing, f"{spec.name} dropped top-level keys: {sorted(missing)}"
        if spec.name != "scan_candidates":
            assert payload["symbol"] == SYMBOL

    @pytest.mark.asyncio
    async def test_scan_tool_never_uses_the_context_fetch(self):
        """scan_candidates fetches per symbol inside run_scan, not via ctx.fetch()."""
        source = InMemoryDataSource({SYMBOL: make_frame()})
        ctx = ToolContext(args={"symbols": [SYMBOL]}, data_source=source)

        spec = next(s for s in TOOLS if s.name == "scan_candidates")
        await spec.run(ctx)

        assert ctx.fetched_frame is None


class TestFrameImmutability:
    """No tool may mutate the OHLCV frame handed out by the data source."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spec", TOOLS, ids=lambda s: s.name)
    async def test_fetched_frame_is_not_mutated(self, spec):
        frame = make_frame()
        original = frame.copy()
        source = NonCopyingDataSource({SYMBOL: frame})

        await dispatch(spec.name, MINIMAL_ARGS[spec.name], data_source=source)

        assert list(frame.columns) == list(original.columns)
        assert frame.equals(original)


class TestDispatchErrors:
    """Error paths go through the one formatting path."""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result = await dispatch("nope", {"symbol": SYMBOL})
        assert result.is_error is True
        assert json.loads(result.content[0].text)["error"] == "Unknown tool: nope"

    @pytest.mark.asyncio
    async def test_dispatch_uses_injected_source_not_production(self):
        """The injected source is the only place data comes from."""
        source = InMemoryDataSource({SYMBOL: make_frame(30)})

        result = await dispatch("get_stock_data", {"symbol": SYMBOL}, data_source=source)

        assert source.fetch_calls == [SYMBOL]
        assert json.loads(result.content[0].text)["data_points"] == 30


class TestToolContext:
    """The context is the tools' only route to arguments and market data."""

    def test_fetch_is_lazy_and_cached(self):
        source = InMemoryDataSource({SYMBOL: make_frame(20)})
        ctx = ToolContext(args={"symbol": SYMBOL}, data_source=source)

        assert source.fetch_calls == []
        first = ctx.fetch()
        second = ctx.fetch()
        assert source.fetch_calls == [SYMBOL]
        assert first.equals(second)

    def test_fetch_hands_out_a_copy_each_call(self):
        source = InMemoryDataSource({SYMBOL: make_frame(20)})
        ctx = ToolContext(args={"symbol": SYMBOL}, data_source=source)

        first = ctx.fetch()
        first["Close"] = 0.0
        assert ctx.fetch()["Close"].iloc[-1] != 0.0

    def test_symbol_is_upper_cased(self):
        ctx = ToolContext(args={"symbol": "aapl"}, data_source=InMemoryDataSource())
        assert ctx.require_symbol() == "AAPL"

    def test_blank_symbol_is_rejected(self):
        ctx = ToolContext(args={"symbol": "  "}, data_source=InMemoryDataSource())
        with pytest.raises(ValueError, match="symbol parameter is required"):
            ctx.require_symbol()

    def test_period_falls_back_to_the_spec_default(self):
        ctx = ToolContext(args={}, data_source=InMemoryDataSource(), default_period="3mo")
        assert ctx.period == "3mo"
        assert ToolContext(args={"period": "1y"}, data_source=InMemoryDataSource()).period == "1y"
