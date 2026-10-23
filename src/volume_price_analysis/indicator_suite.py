"""One analysis-local snapshot of shared, adaptively tuned indicators.

Different period requests are deliberately distinct computations: scan RSI and
comprehensive displays retain their historical tuning.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import indicators as ind


def adaptive_periods(holding_period: int) -> dict[str, int]:
    """Indicator lookbacks scaled to the options holding period.

    Breakpoints: <=14 days (fast), <=21 days (balanced), else standard.
    """
    if holding_period <= 14:
        return {
            "mfi_period": 7,
            "volume_window": 10,
            "rsi_period": 7,
            "adx_period": 10,
            "hv_window": 10,
        }
    if holding_period <= 21:
        return {
            "mfi_period": 10,
            "volume_window": 14,
            "rsi_period": 10,
            "adx_period": 14,
            "hv_window": 14,
        }
    # 22-30 days
    return {
        "mfi_period": 14,
        "volume_window": 20,
        "rsi_period": 14,
        "adx_period": 14,
        "hv_window": 20,
    }


@dataclass
class Snapshot:
    """Shared indicator results and latest-value defaults for one symbol."""

    data: pd.DataFrame = field(repr=False)
    periods: dict[str, int]
    obv: pd.Series
    ad_line: pd.Series
    vwap: pd.Series
    vwma: pd.Series
    mfi: pd.Series
    cmf: pd.Series
    rsi_data: dict[str, Any]
    adx_data: dict[str, Any]
    breakout: dict[str, Any]

    _rsi_variants: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def latest_mfi(self) -> float:
        value = self.mfi.iloc[-1]
        return float(value) if not pd.isna(value) else 50.0

    @property
    def latest_cmf(self) -> float:
        value = self.cmf.iloc[-1]
        return float(value) if not pd.isna(value) else 0.0

    def rsi(self, period: int, window: int) -> dict[str, Any]:
        """Reuse adaptive RSI when tuning matches; preserve other requests."""
        if (period, window) == (self.periods["rsi_period"], self.periods["volume_window"]):
            return self.rsi_data
        key = (period, window)
        if key not in self._rsi_variants:
            self._rsi_variants[key] = ind.calculate_rsi_with_divergence(self.data, period, window)
        return self._rsi_variants[key]


def build_snapshot(data: pd.DataFrame, holding_period: int = 14) -> Snapshot:
    """Compute the shared suite once; consumers own their presentation rules."""
    periods = adaptive_periods(holding_period)
    window = periods["volume_window"]
    return Snapshot(
        data=data,
        periods=periods,
        obv=ind.calculate_obv(data),
        ad_line=ind.calculate_accumulation_distribution(data),
        vwap=ind.calculate_vwap(data),
        vwma=ind.calculate_vwma(data, window),
        mfi=ind.calculate_mfi(data, periods["mfi_period"]),
        cmf=ind.calculate_chaikin_money_flow(data, window),
        rsi_data=ind.calculate_rsi_with_divergence(data, periods["rsi_period"], window),
        adx_data=ind.calculate_adx(data, periods["adx_period"]),
        breakout=ind.detect_volume_breakout(data, 2.0, window),
    )
