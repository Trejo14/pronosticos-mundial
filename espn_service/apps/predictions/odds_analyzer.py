"""Análisis de cuotas: edge, valor, Sharpe ratio, Kellyopt, bankroll management.

Todas las funciones son matemáticas puras — sin dependencias de API ni BD.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class Analysis:
    outcome: str
    our_probability: float
    best_odds: float
    implied_prob: float
    edge: float
    expected_value: float
    kelly_fraction: float
    risk_label: str
    sharpe_ratio: float = 0.0
    kelly_half: float = 0.0       # Kelly fraccionario (half-Kelly)
    kelly_quarter: float = 0.0    # Quarter-Kelly (ultraconservador)


def american_to_decimal(american_odds: int) -> float:
    if american_odds > 0:
        return 1 + american_odds / 100
    return 1 + 100 / abs(american_odds)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return 1.0
    return 1 / decimal_odds


def remove_vig(home_prob: float, draw_prob: float, away_prob: float) -> tuple[float, float, float]:
    total = home_prob + draw_prob + away_prob
    if total <= 0:
        return home_prob, draw_prob, away_prob
    return home_prob / total, draw_prob / total, away_prob / total


def calculate_edge(our_prob: float, market_odds: float) -> float:
    if market_odds <= 1:
        return 0.0
    implied = 1 / market_odds
    if implied <= 0:
        return 0.0
    return (our_prob - implied) / implied


def calculate_expected_value(our_prob: float, decimal_odds: float) -> float:
    return (our_prob * decimal_odds) - 1


def calculate_kelly(our_prob: float, decimal_odds: float) -> float:
    """Kelly completo (peligroso — usar fraccional)."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - our_prob
    kelly = (our_prob * b - q) / b
    return max(kelly, 0.0)


def calculate_sharpe(our_prob: float, decimal_odds: float, kelly: float) -> float:
    """Sharpe ratio aproximado para la apuesta.

    Retorno esperado / desviación estándar.
    """
    if kelly <= 0 or decimal_odds <= 1:
        return 0.0
    ev = calculate_expected_value(our_prob, decimal_odds)
    b = decimal_odds - 1
    variance = (b ** 2) * our_prob * (1 - our_prob)
    if variance <= 0:
        return 0.0
    std_dev = math.sqrt(variance)
    return ev / std_dev if std_dev > 0 else 0.0


def calculate_risk_label(our_prob: float, edge: float, kelly: float) -> str:
    if kelly <= 0:
        return "high"
    if edge > 0.15 and kelly > 0.03:
        return "low"
    if edge > 0.05 and kelly > 0.01:
        return "medium"
    return "high"


def calculate_risk_score(
    our_prob: float,
    edge: float,
    kelly: float,
    num_simulations: int = 1,
    std_dev: float | None = None,
) -> float:
    base_risk = 1.0 - abs(edge)
    if kelly <= 0:
        return 1.0
    prob_risk = 1.0 - our_prob
    if std_dev is not None:
        sim_risk = min(std_dev, 1.0)
        return (base_risk * 0.3 + prob_risk * 0.3 + sim_risk * 0.4) * (1 - min(kelly, 0.1))
    return (base_risk * 0.5 + prob_risk * 0.5) * (1 - min(kelly, 0.1))


def analyze_outcome(
    outcome_name: str,
    our_prob: float,
    best_decimal_odds: float,
) -> Analysis:
    implied = decimal_to_implied_prob(best_decimal_odds)
    edge = calculate_edge(our_prob, best_decimal_odds)
    ev = calculate_expected_value(our_prob, best_decimal_odds)
    kelly = calculate_kelly(our_prob, best_decimal_odds)
    kelly_half = kelly * 0.5
    kelly_quarter = kelly * 0.25
    risk = calculate_risk_label(our_prob, edge, kelly)
    sharpe = calculate_sharpe(our_prob, best_decimal_odds, kelly)
    return Analysis(
        outcome=outcome_name,
        our_probability=round(our_prob, 4),
        best_odds=best_decimal_odds,
        implied_prob=round(implied, 4),
        edge=round(edge, 4),
        expected_value=round(ev, 4),
        kelly_fraction=round(kelly, 4),
        risk_label=risk,
        sharpe_ratio=round(sharpe, 4),
        kelly_half=round(kelly_half, 4),
        kelly_quarter=round(kelly_quarter, 4),
    )


def find_best_odds(odds_list: list[dict]) -> dict[str, float]:
    best: dict[str, float] = {}
    for provider_odds in odds_list:
        for outcome in ["home", "draw", "away"]:
            val = provider_odds.get(f"{outcome}Odds") or provider_odds.get(outcome, {}).get("odds")
            if val and (outcome not in best or val > best[outcome]):
                best[outcome] = float(val)
    return best


def league_margin_to_prob(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> tuple[float, float, float]:
    raw_home = decimal_to_implied_prob(home_odds)
    raw_draw = decimal_to_implied_prob(draw_odds)
    raw_away = decimal_to_implied_prob(away_odds)
    return remove_vig(raw_home, raw_draw, raw_away)
