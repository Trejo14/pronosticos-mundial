"""Motor de predicción de clase mundial.

Modelos multi-deporte a través de engines especializados.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from apps.predictions.engines import get_engine
from apps.predictions.engines.base_engine import TeamContext
from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    compute_team_strength,
    expected_score,
    extract_team_stats_from_event,
)
from apps.predictions.odds_analyzer import remove_vig

LEAGUE_AVG_GOALS = 2.5


@dataclass
class MatchPrediction:
    home_win: float
    draw: float
    away_win: float
    expected_goals_home: float
    expected_goals_away: float
    home_strength: float
    away_strength: float
    confidence: float
    model_agreement: float = 0.0  # qué tanto coinciden los modelos
    home_xg: float = 0.0
    away_xg: float = 0.0
    # Add generic extra for other sports
    extra: dict[str, Any] = None


@dataclass
class TeamInfo:
    espn_id: str
    name: str
    abbreviation: str
    elo: float = INITIAL_ELO
    attacking: float = 1.0
    defensive: float = 1.0
    form_pts: float = 0.5
    recent_gf: float = 1.0
    recent_ga: float = 1.0
    squad_value: float = 50.0
    xg_per_match: float | None = None
    sport: str = "soccer"  # Added to identify the sport
    stats: dict[str, Any] = None


# ──────────────────────── predict_match ────────────────────────

def predict_match(
    home: TeamInfo,
    away: TeamInfo,
    espn_win_probs: tuple[float, float, float] | None = None,
    league_avg_goals: float = LEAGUE_AVG_GOALS,
    elo_weight: float = 0.20,
    market_probs: tuple[float, float, float] | None = None,
) -> MatchPrediction:
    """Predicción completa multi-modelo."""
    
    # Create TeamContext for engine
    home_ctx = TeamContext(
        espn_id=home.espn_id,
        name=home.name,
        abbreviation=home.abbreviation,
        elo=home.elo,
        offensive_rating=home.attacking,
        defensive_rating=home.defensive,
        form_pts=home.form_pts,
        recent_scores_for=home.recent_gf,
        recent_scores_against=home.recent_ga,
        stats=home.stats or {},
    )
    home_ctx.stats["squad_value"] = home.squad_value
    home_ctx.stats["xg_per_match"] = home.xg_per_match

    away_ctx = TeamContext(
        espn_id=away.espn_id,
        name=away.name,
        abbreviation=away.abbreviation,
        elo=away.elo,
        offensive_rating=away.attacking,
        defensive_rating=away.defensive,
        form_pts=away.form_pts,
        recent_scores_for=away.recent_gf,
        recent_scores_against=away.recent_ga,
        stats=away.stats or {},
    )
    away_ctx.stats["squad_value"] = away.squad_value
    away_ctx.stats["xg_per_match"] = away.xg_per_match

    # ESPN probabilities can be treated as market probs if we don't have explicit market probs
    if not market_probs and espn_win_probs:
        market_probs = remove_vig(*espn_win_probs)

    engine = get_engine(home.sport)
    pred = engine.predict(home_ctx, away_ctx, market_probs=market_probs)

    # Convert GenericPrediction to MatchPrediction to maintain backwards compatibility
    return MatchPrediction(
        home_win=pred.home_win,
        draw=pred.draw,
        away_win=pred.away_win,
        expected_goals_home=pred.expected_score_home,
        expected_goals_away=pred.expected_score_away,
        home_strength=pred.home_strength,
        away_strength=pred.away_strength,
        confidence=pred.confidence,
        model_agreement=pred.model_agreement,
        home_xg=pred.extra.get("xg_home", pred.expected_score_home),
        away_xg=pred.extra.get("xg_away", pred.expected_score_away),
        extra=pred.extra,
    )


# ──────────────────────── extractores ESPN ────────────────────────

def extract_espn_win_probs(event_data: dict[str, Any]) -> tuple[float, float, float] | None:
    competitions = event_data.get("competitions", [])
    if not competitions and "events" in event_data:
        events = event_data.get("events", [])
        if events:
            competitions = events[0].get("competitions", [])
    if not competitions:
        return None
    for comp in competitions:
        probs = comp.get("probabilities") or comp.get("predictor") or {}
        home_team_data = probs.get("homeTeam") or {}
        away_team_data = probs.get("awayTeam") or {}
        home_win = home_team_data.get("gameProjection") or home_team_data.get("homeWinPercentage")
        away_win = away_team_data.get("teamChanceLoss") or away_team_data.get("awayWinPercentage")
        if home_win is not None:
            home_val = float(home_win) / 100 if float(home_win) > 1 else float(home_win)
            if away_win is not None:
                away_val = float(away_win) / 100 if float(away_win) > 1 else float(away_win)
            else:
                away_val = 1.0 - home_val
            draw_val = 1.0 - home_val - away_val
            return (home_val, draw_val, away_val)
    return None


def extract_odds(event_data: dict[str, Any]) -> list[dict[str, Any]] | None:
    competitions = event_data.get("competitions", [])
    if not competitions and "events" in event_data:
        events = event_data.get("events", [])
        if events:
            competitions = events[0].get("competitions", [])
    if not competitions:
        return None
    for comp in competitions:
        odds = comp.get("odds") or comp.get("oddsData")
        if odds:
            return odds if isinstance(odds, list) else [odds]
        competitors = comp.get("competitors", [])
        if len(competitors) >= 2:
            return None
    return None


def parse_odds_into_probs(odds_list: list[dict[str, Any]]) -> tuple[float, float, float] | None:
    for provider in odds_list:
        home_ml = provider.get("homeTeamOdds", {}).get("moneyLine") or provider.get("homeOdds")
        away_ml = provider.get("awayTeamOdds", {}).get("moneyLine") or provider.get("awayOdds")
        draw_odds = provider.get("drawOdds") or provider.get("drawOdds")
        if home_ml and away_ml:
            from apps.predictions.odds_analyzer import american_to_decimal, decimal_to_implied_prob
            home_dec = american_to_decimal(home_ml) if isinstance(home_ml, int) and abs(home_ml) > 100 else float(home_ml)
            away_dec = american_to_decimal(away_ml) if isinstance(away_ml, int) and abs(away_ml) > 100 else float(away_ml)
            draw_dec = None
            if draw_odds:
                draw_dec = american_to_decimal(draw_odds) if isinstance(draw_odds, int) and abs(draw_odds) > 100 else float(draw_odds)
            h_prob = decimal_to_implied_prob(home_dec)
            a_prob = decimal_to_implied_prob(away_dec)
            d_prob = decimal_to_implied_prob(draw_dec) if draw_dec else (1.0 - h_prob - a_prob)
            if draw_dec:
                return remove_vig(h_prob, d_prob, a_prob)
            return h_prob, d_prob, a_prob
    return None
