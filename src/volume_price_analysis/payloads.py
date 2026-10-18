"""Type-only contracts for analysis results; runtime payloads remain plain dictionaries."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

# Nested indicator-specific structures remain open where their full contracts
# are outside this pass; analysis-owned fields are declared explicitly.


class ScanCandidateKeyLevels(TypedDict):
    upper_target: float
    lower_target: float


class ScanCandidate(TypedDict):
    conviction: NotRequired[str]
    regime_conflict: NotRequired[str]
    earnings_warning: NotRequired[str]
    symbol: str
    composite_score: float
    recommendation: str
    signal_quality: str
    adx: float
    adx_period: int
    trend_strength: str
    trend_direction: str
    rsi: float
    rsi_divergence: str
    iv_percentile: float
    hv_percentile: float
    iv_implication: str
    expected_move_pct: float
    rvol: float
    latest_price: float
    atr: float
    key_levels: ScanCandidateKeyLevels


class Headline(TypedDict):
    recommendation: str
    composite_score: float
    signal_quality: str
    rationale: str


class ScanResultScanParameters(TypedDict):
    universe: str
    symbols_in_universe: int
    symbols_scanned: int
    holding_period: int
    min_score: float
    min_adx: float
    max_iv_percentile: float
    min_avg_daily_volume: float
    direction_filter: str
    adx_period: int
    volatility_basis: str


class ScanResultSummary(TypedDict):
    total_candidates: int
    bullish_setups: int
    bearish_setups: int
    high_conviction: int
    skipped: int
    errors: int
    rate_limit_errors: int


class ScanResult(TypedDict):
    market_regime: NotRequired[dict[str, Any]]
    scan_parameters: ScanResultScanParameters
    summary: ScanResultSummary
    high_conviction_setups: list[ScanCandidate]
    top_bullish: list[ScanCandidate]
    top_bearish: list[ScanCandidate]
    errors: list[dict[str, str]]
    rate_limited: bool


class ComprehensiveAnalysisVolumeIndicatorsObv(TypedDict):
    value: float
    trend: str
    interpretation: str


class ComprehensiveAnalysisVolumeIndicatorsAccumulationDistribution(TypedDict):
    value: float
    trend: str
    interpretation: str


class ComprehensiveAnalysisVolumeIndicatorsVpt(TypedDict):
    value: float
    trend: str


class ComprehensiveAnalysisVolumeIndicatorsMfi(TypedDict):
    value: float
    condition: str


class ComprehensiveAnalysisVolumeIndicatorsCmf(TypedDict):
    value: float
    condition: str
    interpretation: str


class ComprehensiveAnalysisVolumeIndicatorsRelativeVolume(TypedDict):
    current_rvol: float
    average_volume: float
    current_volume: float
    significance: str


class ComprehensiveAnalysisVolumeIndicators(TypedDict):
    obv: ComprehensiveAnalysisVolumeIndicatorsObv
    accumulation_distribution: ComprehensiveAnalysisVolumeIndicatorsAccumulationDistribution
    vpt: ComprehensiveAnalysisVolumeIndicatorsVpt
    mfi: ComprehensiveAnalysisVolumeIndicatorsMfi
    cmf: ComprehensiveAnalysisVolumeIndicatorsCmf
    relative_volume: ComprehensiveAnalysisVolumeIndicatorsRelativeVolume
    volume_breakout: dict[str, Any]


class ComprehensiveAnalysisPriceIndicatorsVwap(TypedDict):
    value: float
    price_vs_vwap: str
    position: str


class ComprehensiveAnalysisPriceIndicatorsVwma20(TypedDict):
    value: float
    price_vs_vwma: str
    position: str


class ComprehensiveAnalysisPriceIndicatorsPriceRoc(TypedDict):
    current_roc: float
    direction: str
    strength: str
    volume_confirmed: bool
    signal: str


class ComprehensiveAnalysisPriceIndicators(TypedDict):
    vwap: ComprehensiveAnalysisPriceIndicatorsVwap
    vwma_20: ComprehensiveAnalysisPriceIndicatorsVwma20
    price_roc: ComprehensiveAnalysisPriceIndicatorsPriceRoc


class ComprehensiveAnalysisVolatilityIndicatorsHistoricalVolatility20D(TypedDict):
    value: float
    annualized: bool
    interpretation: str


class ComprehensiveAnalysisVolatilityIndicatorsAtr14D(TypedDict):
    value: float
    interpretation: str


class BollingerSnapshot(TypedDict):
    upper: float
    middle: float
    lower: float
    percent_b: float
    bandwidth: float


class ComprehensiveAnalysisVolatilityIndicatorsBollingerBands(BollingerSnapshot):
    squeeze_status: str


class ComprehensiveAnalysisVolatilityIndicators(TypedDict):
    historical_volatility_20d: ComprehensiveAnalysisVolatilityIndicatorsHistoricalVolatility20D
    atr_14d: ComprehensiveAnalysisVolatilityIndicatorsAtr14D
    bollinger_bands: ComprehensiveAnalysisVolatilityIndicatorsBollingerBands


class ComprehensiveAnalysisVolumeProfile(TypedDict):
    point_of_control: float
    value_area_high: float
    value_area_low: float
    current_position: str
    interpretation: str
    poc_distance: str
    vah_distance: str
    val_distance: str


class ComprehensiveAnalysis(TypedDict):
    symbol: str
    analysis_type: str
    period: str
    latest_price: float
    headline: Headline
    volume_indicators: ComprehensiveAnalysisVolumeIndicators
    price_indicators: ComprehensiveAnalysisPriceIndicators
    volatility_indicators: ComprehensiveAnalysisVolatilityIndicators
    volume_profile: ComprehensiveAnalysisVolumeProfile
    volume_trends: dict[str, Any]
    summary: list[str]


class OptionsAnalysisParameters(TypedDict):
    holding_period: int
    days_to_expiration: int
    mfi_period: int
    volume_window: int
    rsi_period: int
    adx_period: int
    hv_window: int
    squeeze_window: int
    optimization: str


class OptionsAnalysisCompositeSignal(TypedDict):
    score: float
    recommendation: str
    action: str
    signal_quality: str
    quality_note: str
    score_breakdown: dict[str, Any]


class OptionsAnalysisTrendAnalysisAdx(TypedDict):
    value: float
    plus_di: float
    minus_di: float
    trend_strength: str
    trend_direction: str
    adx_slope: str
    interpretation: str


class OptionsAnalysisTrendAnalysisRsi(TypedDict):
    value: float
    condition: str
    divergence_type: str
    divergence_signal: str
    interpretation: str


class OptionsAnalysisTrendAnalysis(TypedDict):
    adx: OptionsAnalysisTrendAnalysisAdx
    rsi: OptionsAnalysisTrendAnalysisRsi


class OptionsAnalysisVolumeIndicatorsObv(TypedDict):
    value: float
    trend: str
    short_term_momentum: str


class OptionsAnalysisVolumeIndicatorsAccumulationDistribution(TypedDict):
    value: float
    trend: str
    signal: str


class OptionsAnalysisVolumeIndicatorsVpt(TypedDict):
    value: float
    trend: str
    volume_conviction: str


class OptionsAnalysisVolumeIndicatorsMfi(TypedDict):
    value: float
    condition: str
    options_signal: str


class OptionsAnalysisVolumeIndicatorsCmf(TypedDict):
    value: float
    signal: str


class OptionsAnalysisVolumeIndicatorsRelativeVolume(TypedDict):
    current_rvol: float
    significance: str


class OptionsAnalysisVolumeIndicators(TypedDict):
    obv: OptionsAnalysisVolumeIndicatorsObv
    accumulation_distribution: OptionsAnalysisVolumeIndicatorsAccumulationDistribution
    vpt: OptionsAnalysisVolumeIndicatorsVpt
    mfi: OptionsAnalysisVolumeIndicatorsMfi
    cmf: OptionsAnalysisVolumeIndicatorsCmf
    relative_volume: OptionsAnalysisVolumeIndicatorsRelativeVolume
    volume_breakout: dict[str, Any]


class OptionsAnalysisPriceIndicatorsVwap(TypedDict):
    value: float
    price_vs_vwap: str
    position: str
    signal: str


class OptionsAnalysisPriceIndicatorsVwma(TypedDict):
    value: float
    price_vs_vwma: str
    trend: str


class OptionsAnalysisPriceIndicatorsPriceRoc(TypedDict):
    current_roc: float
    direction: str
    strength: str
    volume_confirmed: bool


class OptionsAnalysisPriceIndicators(TypedDict):
    vwap: OptionsAnalysisPriceIndicatorsVwap
    vwma: OptionsAnalysisPriceIndicatorsVwma
    price_roc: OptionsAnalysisPriceIndicatorsPriceRoc


class OptionsAnalysisVolatilityAnalysisIvPercentileProxy(TypedDict):
    percentile: float
    hv_percentile: float
    basis: str
    is_proxy: bool
    current_hv: float
    hv_range: str
    interpretation: str
    options_implication: str
    strategy_suggestion: str


class OptionsAnalysisVolatilityAnalysisExpectedMove(TypedDict):
    dollars: float
    percent: float
    upper_target: float
    lower_target: float
    targets: dict[str, Any]
    strike_guidance: dict[str, Any]
    interpretation: str


class OptionsAnalysisVolatilityAnalysisAtr(TypedDict):
    value: float
    daily_range: str
    stop_loss_suggestion: str


class OptionsAnalysisVolatilityAnalysisBollingerBands(BollingerSnapshot):
    squeeze_detected: bool
    position: str


class OptionsAnalysisVolatilityAnalysis(TypedDict):
    iv_percentile_proxy: OptionsAnalysisVolatilityAnalysisIvPercentileProxy
    expected_move: OptionsAnalysisVolatilityAnalysisExpectedMove
    atr: OptionsAnalysisVolatilityAnalysisAtr
    bollinger_bands: OptionsAnalysisVolatilityAnalysisBollingerBands


class OptionsAnalysisVolumeProfileStrikeSelectionGuidance(TypedDict):
    poc_strike: str
    vah_strike: str
    val_strike: str
    current_vs_poc: str


class OptionsAnalysisVolumeProfile(TypedDict):
    point_of_control: float
    value_area_high: float
    value_area_low: float
    current_position: str
    interpretation: str
    strike_selection_guidance: OptionsAnalysisVolumeProfileStrikeSelectionGuidance


class OptionsAnalysisTimeDecay(TypedDict):
    days_to_expiration: int
    theta_risk: str
    theta_note: str


class OptionsAnalysis(TypedDict):
    conviction: NotRequired[str]
    earnings_warning: NotRequired[str]
    symbol: str
    analysis_type: str
    period: str
    latest_price: float
    headline: Headline
    parameters: OptionsAnalysisParameters
    composite_signal: OptionsAnalysisCompositeSignal
    trend_analysis: OptionsAnalysisTrendAnalysis
    volume_indicators: OptionsAnalysisVolumeIndicators
    price_indicators: OptionsAnalysisPriceIndicators
    volatility_analysis: OptionsAnalysisVolatilityAnalysis
    volume_profile: OptionsAnalysisVolumeProfile
    time_decay: OptionsAnalysisTimeDecay
    volume_trends: dict[str, Any]
    options_insights: list[str]


class CompositeScore(TypedDict):
    composite_score: float
    raw_score: float
    max_score: float
    recommendation: str
    action: str
    signal_quality: str
    quality_note: str
    score_breakdown: dict[str, Any]
    adx_period: int
    adx_summary: dict[str, Any]
    indicator_summary: dict[str, Any]
