"""Tests for the MCP tool registry and the single dispatcher built on it.

These tests drive every registry record through the public dispatcher seam
(``server.dispatch``) with an injected ``DataSource`` -- no patching of any
module-level fetch.
"""

import copy
import datetime
import json
import math

import jsonschema
import pandas as pd
import pytest
from mcp.client.client import Client
from mcp.server import Server, ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
)

from volume_price_analysis.data_fetcher import DEFAULT_TIMEOUT, InMemoryDataSource
from volume_price_analysis.server import dispatch, handle_list_tools
from volume_price_analysis.tools import (
    TOOLS,
    TOOLS_BY_NAME,
    ToolContext,
    ToolSpec,
    validate_arguments,
)

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
    "calculate_anchored_vwap": {"symbol": SYMBOL, "anchor": "period_start"},
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
    "calculate_anchored_vwap": {
        "symbol",
        "latest_avwap",
        "distance_percent",
        "distance_atr",
        "anchor",
        "recent_values",
    },
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

    def test_registry_lists_thirteen_tools(self):
        assert len(TOOLS) == 13
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


SCHEMA_TOOLS = {"scan_candidates", "options_analysis", "comprehensive_analysis"}
SCHEMA_SPECS = [spec for spec in TOOLS if spec.output_schema is not None]

# Under MINIMAL_ARGS the sample frame scores below the default min_score, so every
# scan list is empty. This scan yields a candidate and, for the symbol the
# in-memory source lacks, an errors entry -- so list-item sub-schemas get exercised.
MISSING_SYMBOL = "ZZZZ"
POPULATED_SCAN_ARGS = {"symbols": [SYMBOL, MISSING_SYMBOL], "min_score": 0, "min_adx": 0}

# (case id, tool name, arguments): every tool's minimal call plus the populated scan.
OUTPUT_CASES = [(name, name, args) for name, args in MINIMAL_ARGS.items()] + [
    ("scan_candidates_populated", "scan_candidates", POPULATED_SCAN_ARGS)
]
SCHEMA_CASES = [case for case in OUTPUT_CASES if TOOLS_BY_NAME[case[1]].output_schema]
_CASE_IDS = [case[0] for case in OUTPUT_CASES]
_SCHEMA_CASE_IDS = [case[0] for case in SCHEMA_CASES]

# Consumer-read keys below the top level, by (path, key). The rename sweep must
# reach each of them, so a schema that stops requiring one fails loudly.
_CANDIDATE_KEYS = (
    "symbol",
    "composite_score",
    "adx",
    "iv_percentile",
    "hv_percentile",
    "latest_price",
    "atr",
    "signal_quality",
    "key_levels",
)
_PROFILE_READS = {
    (("volume_profile",), key)
    for key in ("point_of_control", "value_area_high", "value_area_low", "current_position")
}


def _reads(path: tuple, *keys: str) -> set[tuple[tuple, str]]:
    return {(path, key) for key in keys}


# What agent/ai_client.py's _project_deep_analysis reads from an options_analysis.
_BRIEFING_PROJECTION_READS = (
    _PROFILE_READS
    | _reads(("trend_analysis",), "adx", "rsi")
    | _reads(("trend_analysis", "adx"), "value", "trend_strength", "trend_direction")
    | _reads(("trend_analysis", "rsi"), "value", "condition", "divergence_type")
    | _reads(
        ("volatility_analysis",), "iv_percentile_proxy", "expected_move", "atr", "bollinger_bands"
    )
    | _reads(
        ("volatility_analysis", "iv_percentile_proxy"),
        "percentile",
        "hv_percentile",
        "options_implication",
    )
    | _reads(("volatility_analysis", "expected_move"), "percent", "upper_target", "lower_target")
    | _reads(("volatility_analysis", "atr"), "daily_range", "stop_loss_suggestion")
    | _reads(("volatility_analysis", "bollinger_bands"), "position", "squeeze_detected")
    | _reads(("volume_indicators", "obv"), "trend")
    | _reads(("volume_indicators", "accumulation_distribution"), "signal")
    | _reads(("volume_indicators", "mfi"), "condition")
    | _reads(("volume_indicators", "cmf"), "signal")
    | _reads(("volume_indicators", "relative_volume"), "current_rvol")
    | _reads(("volume_indicators", "volume_breakout"), "is_breakout")
)
NESTED_REQUIRED: dict[str, set[tuple[tuple, str]]] = {
    "scan_candidates_populated": {(("top_bullish", 0), key) for key in _CANDIDATE_KEYS}
    | {
        (("errors", 0), "symbol"),
        (("errors", 0), "error"),
        (("scan_parameters",), "symbols_scanned"),
        (("summary",), "total_candidates"),
    },
    "options_analysis": {
        (("composite_signal",), "action"),
        (("composite_signal",), "signal_quality"),
        (("headline",), "rationale"),
    }
    | _BRIEFING_PROJECTION_READS,
    "comprehensive_analysis": {(("headline",), "rationale")}
    | _PROFILE_READS
    | {
        (("volume_indicators", "obv"), "trend"),
        (("volume_indicators", "mfi"), "condition"),
        (("volume_indicators", "relative_volume"), "current_rvol"),
        (("volume_indicators", "volume_breakout"), "is_breakout"),
    },
}


def _legacy_text(payload: object) -> str:
    """The text block every tool returned before structured output (PDE-197).

    An independent copy of the pre-change formatting, so a change to the
    server's serialisation shows up here rather than being mirrored.
    """

    def scrub(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    return json.dumps(scrub(payload), indent=2, default=str)


def _validator(schema: dict) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _source() -> InMemoryDataSource:
    return InMemoryDataSource({SYMBOL: make_frame()})


async def _structured(tool: str, args: dict) -> dict:
    result = await dispatch(tool, dict(args), data_source=_source())
    assert not result.is_error, result.content[0].text
    return result.structured_content


def _required_paths(schema: dict, payload: dict, path: tuple = ()):
    """Yield ``(path, key)`` for every required key the schema imposes on ``payload``.

    Descends into object properties and the first item of each array of objects.
    """
    for key in schema.get("required", []):
        yield path, key
    for key, sub in schema.get("properties", {}).items():
        value = payload.get(key)
        if sub.get("type") == "object" and isinstance(value, dict):
            yield from _required_paths(sub, value, (*path, key))
        elif "items" in sub and isinstance(value, list) and value and isinstance(value[0], dict):
            yield from _required_paths(sub["items"], value[0], (*path, key, 0))


def _dig(payload: object, path: tuple) -> dict:
    for step in path:
        payload = payload[step]  # type: ignore[index]
    assert isinstance(payload, dict)
    return payload


def _all_dicts(obj: object):
    """Every dict in ``obj``, including dicts inside lists, outermost first."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _all_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _all_dicts(item)


class TestStructuredOutput:
    """High-value tools declare an output schema and return structured content (PDE-197)."""

    def test_schema_tools_are_the_high_value_three(self):
        assert {spec.name for spec in SCHEMA_SPECS} == SCHEMA_TOOLS

    @pytest.mark.asyncio
    async def test_list_tools_advertises_output_schema(self):
        tools = await handle_list_tools()
        assert [t.output_schema for t in tools] == [spec.output_schema for spec in TOOLS]

    @pytest.mark.parametrize("spec", SCHEMA_SPECS, ids=lambda s: s.name)
    def test_schemas_are_lenient_objects(self, spec):
        schema = spec.output_schema
        _validator(schema)
        assert schema["type"] == "object"
        assert set(schema["required"]) <= REQUIRED_KEYS[spec.name]
        for sub in _all_dicts(schema):
            if sub.get("type") == "object":
                assert sub.get("additionalProperties") is True, sub

    @pytest.mark.asyncio
    async def test_populated_scan_has_candidates_and_errors(self):
        """Guard: the populated case really exercises the list-item sub-schemas."""
        payload = await _structured("scan_candidates", POPULATED_SCAN_ARGS)
        assert payload["top_bullish"] or payload["top_bearish"]
        assert payload["errors"] == [
            {"symbol": MISSING_SYMBOL, "error": f"No data found for symbol {MISSING_SYMBOL}"}
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("case", "tool", "args"), OUTPUT_CASES, ids=_CASE_IDS)
    async def test_output_matches_schema_and_text_is_unchanged(self, case, tool, args):
        spec = TOOLS_BY_NAME[tool]
        source = _source()
        raw = await spec.run(ToolContext(args=dict(args), data_source=source))

        result = await dispatch(tool, dict(args), data_source=source)

        assert not result.is_error, result.content[0].text
        assert result.content[0].text == _legacy_text(raw)
        if spec.output_schema is None:
            assert result.structured_content is None
            return
        assert result.structured_content == json.loads(result.content[0].text)
        _validator(spec.output_schema).validate(result.structured_content)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("case", "tool", "args"), SCHEMA_CASES, ids=_SCHEMA_CASE_IDS)
    async def test_renaming_a_required_key_fails_validation(self, case, tool, args):
        schema = TOOLS_BY_NAME[tool].output_schema
        payload = await _structured(tool, args)
        validator = _validator(schema)
        paths = list(_required_paths(schema, payload))

        assert NESTED_REQUIRED.get(case, set()) <= set(paths)
        unnoticed = []
        for path, key in paths:
            drifted = copy.deepcopy(payload)
            target = _dig(drifted, path)
            target[f"{key}_renamed"] = target.pop(key)
            if validator.is_valid(drifted):
                unnoticed.append((path, key))
        assert not unnoticed, f"{case}: renames went unnoticed: {unnoticed}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("case", "tool", "args"), SCHEMA_CASES, ids=_SCHEMA_CASE_IDS)
    async def test_adding_a_field_keeps_output_valid(self, case, tool, args):
        payload = await _structured(tool, args)
        for obj in list(_all_dicts(payload)):
            obj["new_field"] = {"anything": [1, "two", None]}

        _validator(TOOLS_BY_NAME[tool].output_schema).validate(payload)
        if case == "scan_candidates_populated":
            assert "new_field" in payload["errors"][0]
            assert "new_field" in (payload["top_bullish"] or payload["top_bearish"])[0]

    @pytest.mark.asyncio
    async def test_error_result_carries_no_structured_content(self):
        source = InMemoryDataSource()
        result = await dispatch("options_analysis", {"symbol": SYMBOL}, data_source=source)

        assert result.is_error is True
        assert result.structured_content is None
        assert "error" in json.loads(result.content[0].text)


def _client_server(source: InMemoryDataSource) -> Server:
    """A lowlevel server wired to the real list-tools and dispatcher, with an injected source."""

    async def on_list(_ctx: ServerRequestContext, _p: PaginatedRequestParams | None):
        return ListToolsResult(tools=await handle_list_tools())

    async def on_call(_ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        return await dispatch(params.name, params.arguments or {}, data_source=source)

    return Server("test", on_list_tools=on_list, on_call_tool=on_call)


class TestClientValidation:
    """An MCP client re-validates structured content against the advertised schema."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("case", "tool", "args"), SCHEMA_CASES, ids=_SCHEMA_CASE_IDS)
    async def test_client_accepts_structured_result(self, case, tool, args):
        async with Client(_client_server(_source())) as client:
            result = await client.call_tool(tool, dict(args))

        assert not result.is_error
        assert result.structured_content == json.loads(result.content[0].text)

    @pytest.mark.asyncio
    async def test_client_does_not_reject_error_result(self):
        async with Client(_client_server(InMemoryDataSource())) as client:
            result = await client.call_tool("options_analysis", {"symbol": SYMBOL})

        assert result.is_error is True
        assert result.structured_content is None


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


def _bounded_params() -> list[tuple[str, str, int | float, int | float]]:
    """Every (tool, parameter, minimum, maximum) a registry schema declares."""
    return [
        (spec.name, param, prop["minimum"], prop["maximum"])
        for spec in TOOLS
        for param, prop in spec.input_schema["properties"].items()
        if "minimum" in prop and "maximum" in prop
    ]


class TestSchemaBounds:
    """Numeric ranges come from each tool's schema, not hand-written checks (PDE-153)."""

    def test_sweep_covers_every_bounded_parameter(self):
        # Guards the parametrised sweep below against silently testing nothing.
        assert {(tool, param) for tool, param, _, _ in _bounded_params()} == {
            ("calculate_volume_profile", "num_bins"),
            ("calculate_mfi", "mfi_period"),
            ("calculate_cmf", "cmf_period"),
            ("analyze_volume_trends", "window"),
            ("options_analysis", "holding_period"),
            ("options_analysis", "days_to_expiration"),
            ("scan_candidates", "holding_period"),
            ("scan_candidates", "max_results"),
            ("scan_candidates", "max_iv_percentile"),
            ("calculate_rsi_divergence", "rsi_period"),
            ("calculate_rsi_divergence", "divergence_lookback"),
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("tool", "param", "low", "high"), _bounded_params())
    async def test_out_of_range_rejected_before_any_fetch(self, tool, param, low, high):
        source = InMemoryDataSource({SYMBOL: make_frame()})
        for bad in (low - 1, high + 1):
            result = await dispatch(tool, {**MINIMAL_ARGS[tool], param: bad}, data_source=source)

            assert result.is_error is True
            assert json.loads(result.content[0].text)["error"] == (
                f"{param} must be between {low} and {high}, got {bad}"
            )
        assert source.fetch_calls == []

    @pytest.mark.parametrize(("tool", "param", "low", "high"), _bounded_params())
    def test_boundaries_are_accepted(self, tool, param, low, high):
        for edge in (low, high):
            validate_arguments({param: edge}, TOOLS_BY_NAME[tool].input_schema)

    def test_absent_and_unbounded_arguments_pass(self):
        schema = TOOLS_BY_NAME["scan_candidates"].input_schema
        validate_arguments({}, schema)
        validate_arguments({"min_score": 0, "min_adx": 1e9}, schema)

    def test_one_sided_bounds(self):
        schema = {
            "properties": {
                "floor": {"type": "integer", "minimum": 1},
                "ceiling": {"type": "integer", "maximum": 9},
            }
        }
        with pytest.raises(ValueError, match="^floor must be at least 1, got 0$"):
            validate_arguments({"floor": 0}, schema)
        with pytest.raises(ValueError, match="^ceiling must be at most 9, got 10$"):
            validate_arguments({"ceiling": 10}, schema)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["20", 20.0])
    async def test_numeric_num_bins_is_coerced(self, value):
        source = InMemoryDataSource({SYMBOL: make_frame()})
        result = await dispatch(
            "calculate_volume_profile", {"symbol": SYMBOL, "num_bins": value}, data_source=source
        )
        assert not result.is_error, result.content[0].text
        assert json.loads(result.content[0].text)["num_price_levels"] == 20


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


class TestBoundaryValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("spec", TOOLS, ids=lambda s: s.name)
    async def test_optional_nulls_behave_like_omission_for_every_tool(self, spec):
        source = InMemoryDataSource({SYMBOL: make_frame()})
        args = dict(MINIMAL_ARGS[spec.name])
        baseline = await dispatch(spec.name, args.copy(), data_source=source)
        for name in spec.input_schema["properties"]:
            if name not in spec.input_schema.get("required", []) and name not in args:
                args[name] = None
        result = await dispatch(spec.name, args, data_source=source)
        assert not result.is_error, result.content[0].text
        assert json.loads(result.content[0].text) == json.loads(baseline.content[0].text)
        if spec.name == "calculate_volume_profile":
            assert json.loads(result.content[0].text)["num_price_levels"] == 20

    def test_null_symbols_is_omitted_before_scan_universe_selection(self):
        args = {"symbols": None, "universe": "sp500"}
        validate_arguments(args, TOOLS_BY_NAME["scan_candidates"].input_schema)
        assert args == {"universe": "sp500"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("args", [{}, {"symbol": None}])
    async def test_required_symbol_is_rejected_when_absent_or_null(self, args):
        source = InMemoryDataSource()
        result = await dispatch("get_stock_data", args, data_source=source)
        assert result.is_error is True
        error = json.loads(result.content[0].text)["error"]
        assert "symbol" in error
        assert "internal error" not in error
        assert source.fetch_calls == []

    @pytest.mark.asyncio
    async def test_stock_data_optional_nulls_use_defaults(self):
        source = InMemoryDataSource({SYMBOL: make_frame()})
        result = await dispatch(
            "get_stock_data",
            {"symbol": SYMBOL, "start_date": None, "end_date": None, "period": None},
            data_source=source,
        )
        assert not result.is_error, result.content[0].text
        assert json.loads(result.content[0].text)["period"] == "1mo"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool", "param", "value"),
        [
            ("calculate_volume_profile", "num_bins", value)
            for value in (True, False, "abc", [], 20.5, "20.5", "nan", "inf", float("nan"))
        ]
        + [("get_stock_data", "symbol", value) for value in (None, 123, [])]
        + [("scan_candidates", "symbols", value) for value in ("AAPL", [None], [1], [""])]
        + [("scan_candidates", "symbols", ["  "])]
        + [("scan_candidates", "min_adx", True)],
    )
    async def test_invalid_types_return_clear_errors_before_fetch(self, tool, param, value):
        source = InMemoryDataSource()
        result = await dispatch(tool, {**MINIMAL_ARGS[tool], param: value}, data_source=source)
        assert result.is_error is True
        error = json.loads(result.content[0].text)["error"]
        assert param in error
        assert "internal error" not in error
        assert source.fetch_calls == []

    @pytest.mark.asyncio
    async def test_scan_normalizes_symbols_and_applies_volume_filter(self):
        source = InMemoryDataSource({SYMBOL: make_frame()})
        args = {"symbols": ["aapl"], "min_score": "0", "min_adx": "0"}
        baseline = await dispatch("scan_candidates", args.copy(), data_source=source)
        assert not baseline.is_error, baseline.content[0].text
        payload = json.loads(baseline.content[0].text)
        assert payload["summary"]["total_candidates"] == 1
        assert payload["scan_parameters"]["min_avg_daily_volume"] == 0
        filtered = await dispatch(
            "scan_candidates", {**args, "min_avg_daily_volume": "10000000"}, data_source=source
        )
        assert not filtered.is_error, filtered.content[0].text
        payload = json.loads(filtered.content[0].text)
        assert payload["scan_parameters"]["min_avg_daily_volume"] == 10_000_000
        assert payload["summary"]["total_candidates"] == 0
        assert source.fetch_calls == [SYMBOL, SYMBOL]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("min_score", -1),
            ("min_adx", -1),
            ("max_iv_percentile", -1),
            ("max_iv_percentile", 101),
            ("min_avg_daily_volume", -1),
            ("min_avg_daily_volume", "-1"),
        ],
    )
    async def test_scan_filter_ranges_are_enforced_before_fetch(self, param, value):
        source = InMemoryDataSource()
        result = await dispatch(
            "scan_candidates", {"symbols": [SYMBOL], param: value}, data_source=source
        )
        assert result.is_error is True
        assert param in json.loads(result.content[0].text)["error"]
        assert source.fetch_calls == []


@pytest.mark.asyncio
async def test_anchored_vwap_default_and_tail():
    frame = make_frame(30)
    # One strict low, confirmed three bars later; default must select it.
    frame.loc[20, "Low"] = 50.0
    source = InMemoryDataSource({SYMBOL: frame})
    response = await dispatch("calculate_anchored_vwap", {"symbol": SYMBOL}, data_source=source)
    assert not response.is_error
    payload = json.loads(response.content[0].text)
    assert payload["anchor"]["type"] == "swing_low"
    assert payload["anchor"]["date"] == frame.loc[20, "Date"].isoformat()
    assert len(payload["recent_values"]) == 10
    assert payload["recent_values"][-1]["AVWAP"] == payload["latest_avwap"]
    assert payload["recent_values"][0]["AVWAP"] == pytest.approx(272 / 3)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,message",
    [
        ({"anchor": "bad"}, "anchor must be one of"),
        ({"anchor": "date"}, "date is required"),
        ({"anchor": "date", "date": "2024-02"}, "valid calendar date"),
        ({"anchor": "date", "date": "2024-02-30"}, "valid calendar date"),
    ],
)
async def test_anchored_vwap_invalid_args_rejected_before_fetch(args, message):
    source = InMemoryDataSource({SYMBOL: make_frame()})
    result = await dispatch(
        "calculate_anchored_vwap", {"symbol": SYMBOL, **args}, data_source=source
    )
    assert result.is_error is True
    assert message in json.loads(result.content[0].text)["error"]
    assert source.fetch_calls == []
