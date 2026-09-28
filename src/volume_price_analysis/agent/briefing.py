"""The Briefing module: the one owner of the delivered morning document.

Everything that decides what the reader sees in the email lives here: the
dated title, the regime header, the pick table, the model's prose (or the
fallback body when the model was unavailable), and the stats footer. The
orchestrator gathers a :class:`BriefingInputs` and makes one call;
``email_sender`` only turns the returned markdown into a MIME message.
"""

import json
from dataclasses import dataclass
from datetime import date

from .ai_client import format_briefing_date
from .picks import Pick, build_picks, render_picks_table
from .regime import annotate_regime_conflicts, format_regime_header


@dataclass(frozen=True)
class BriefingInputs:
    """Everything a briefing is rendered from.

    ``model_text`` is the AI provider's prose, or None when the provider was
    unavailable (the render then falls back to a programmatic body).
    """

    scan_results: dict
    deep_analyses: list[dict]
    regime: dict
    briefing_date: date
    elapsed_s: float
    model_text: str | None


def render(inputs: BriefingInputs) -> str:
    """Render the complete markdown body of the delivered briefing.

    The template (dated title, regime verdict, fixed pick table, footer) is
    rendered here, never by the model, so the date is always real and the pick
    block always parses (PDE-69). ``tests/golden/briefing_body.md`` pins it.
    """
    picks = _picks(inputs)
    text = inputs.model_text if inputs.model_text is not None else _fallback_body(inputs)
    return (
        f"# Morning Market Briefing — {format_briefing_date(inputs.briefing_date)}\n\n"
        f"{_regime_header(inputs, picks)}\n\n"
        f"## Pick Summary\n\n{render_picks_table(picks)}\n\n"
        f"{text}"
        f"{_stats_line(inputs)}"
    )


def render_raw(inputs: BriefingInputs) -> str:
    """Render the raw-data body sent in --no-ai mode.

    The regime header, then the regime-annotated scan result and every deep
    analysis as JSON, for a reader who wants the numbers without prose.
    """
    scan_results = annotate_regime_conflicts(inputs.scan_results, inputs.regime)
    body = f"{_regime_header(inputs, _picks(inputs))}\n\n"
    body += "# Morning Market Scan Results\n\n"
    body += f"```json\n{json.dumps(scan_results, indent=2, default=str)}\n```\n\n"

    if inputs.deep_analyses:
        body += "# Deep Analysis Results\n\n"
        for analysis in inputs.deep_analyses:
            symbol = analysis.get("symbol", "Unknown")
            body += f"## {symbol}\n"
            body += f"```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"
    return body


def _fallback_body(inputs: BriefingInputs) -> str:
    """The programmatic body used when the AI provider was unavailable.

    Rendered beneath the dated title, so it opens at H2.
    """
    scan_results = annotate_regime_conflicts(inputs.scan_results, inputs.regime)
    lines = ["## Fallback Briefing (AI unavailable)\n"]

    summary = scan_results.get("summary", {})
    lines.append(f"**Candidates found:** {summary.get('total_candidates', 0)}")
    lines.append(f"**Bullish setups:** {summary.get('bullish_setups', 0)}")
    lines.append(f"**Bearish setups:** {summary.get('bearish_setups', 0)}")
    lines.append(f"**High conviction:** {summary.get('high_conviction', 0)}\n")

    conflict_symbols = {
        c.get("symbol")
        for c in scan_results.get("high_conviction_setups", [])
        if isinstance(c, dict) and c.get("regime_conflict")
    }
    if inputs.deep_analyses:
        lines.append("## Top Candidates\n")
        for a in inputs.deep_analyses:
            sym = a.get("symbol", "?")
            score = a.get("composite_signal", {}).get("score", 0)
            rec = a.get("composite_signal", {}).get("recommendation", "?")
            price = a.get("latest_price", 0)
            line = f"- **{sym}** @ ${price:.2f} | Score: {score:.1f} | {rec}"
            if sym in conflict_symbols:
                line += " | ⚠️ counter-regime setup"
            lines.append(line)

    return "\n".join(lines)


def _picks(inputs: BriefingInputs) -> list[Pick]:
    return build_picks(inputs.scan_results, inputs.deep_analyses, regime=inputs.regime)


def _regime_header(inputs: BriefingInputs, picks: list[Pick]) -> str:
    """The regime verdict line; the conflict count is the table's flagged rows."""
    return format_regime_header(inputs.regime, sum(p.regime_conflict for p in picks))


def _stats_line(inputs: BriefingInputs) -> str:
    """The footer appended to every rendered briefing."""
    symbols_scanned = inputs.scan_results.get("scan_parameters", {}).get("symbols_scanned")
    total_candidates = inputs.scan_results.get("summary", {}).get("total_candidates", 0)
    scanned_part = f"{symbols_scanned} symbols scanned | " if symbols_scanned else ""
    return (
        f"\n\n---\n"
        f"**14-day holding period** - Indicators, expected moves, and strategies "
        f"are calibrated for approx. 14 DTE options. Shorter-duration plays (0-5 DTE) "
        f"may need different setups.\n\n"
        f"*Generated in {inputs.elapsed_s:.1f}s | "
        f"{scanned_part}"
        f"{total_candidates} candidates found | "
        f"{len(inputs.deep_analyses)} deep analyses*"
    )
