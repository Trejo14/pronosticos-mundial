"""Match prediction engine: combines ESPN data with statistical models.

Uses:
- ESPN win probabilities as a baseline
- Elo ratings for head-to-head adjustment
- Poisson-based expected goals model
- Market odds for calibration
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    compute_team_strength,
    expected_score,
    extract_team_stats_from_event,
)
from apps.predictions.odds_analyzer import (
    remove_vig,
)


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


@dataclass
class TeamInfo:
    espn_id: str
    name: str
    abbreviation: str
    elo: float = INITIAL_ELO
    attacking: float = 1.0
    defensive: float = 1.0


def poisson_prob(goals: int, expected: float) -> float:
    if expected <= 0:
        return 1.0 if goals == 0 else 0.0
    return (math.exp(-expected) * (expected ** goals)) / math.factorial(goals)


def poisson_match_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for i in range(max_goals + 1):
        p_i = poisson_prob(i, lambda_home)
        for j in range(max_goals + 1):
            p_j = poisson_prob(j, lambda_away)
            prob = p_i * p_j
            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    return home_win, draw, away_win


def adjust_probabilities_with_elo(
    base_home: float,
    base_draw: float,
    base_away: float,
    elo_home: float,
    elo_away: float,
    home_advantage: float = 0.0,
    weight: float = 0.3,
) -> tuple[float, float, float]:
    expected_home = expected_score(elo_home + home_advantage, elo_away)
    elo_home_prob = expected_home
    elo_draw_prob = 0.0
    elo_away_prob = 1.0 - expected_home
    if base_draw > 0.05:
        draw_share = base_draw / (1.0 - base_home - base_away) if base_home + base_away < 1 else 0.25
        elo_draw_prob = (1 - elo_home_prob - elo_away_prob) * draw_share * 2 if (1 - elo_home_prob - elo_away_prob) > 0 else 0.25
        residual = 1.0 - elo_draw_prob
        elo_home_prob *= residual
        elo_away_prob *= residual
    blended_home = base_home * (1 - weight) + elo_home_prob * weight
    blended_draw = base_draw * (1 - weight) + elo_draw_prob * weight
    blended_away = base_away * (1 - weight) + elo_away_prob * weight
    return remove_vig(blended_home, blended_draw, blended_away)


def calculate_expected_goals(
    home_strength: TeamStrength,
    away_strength: TeamStrength,
    league_avg_goals: float = 2.5,
) -> tuple[float, float]:
    home_attack = home_strength.attacking
    home_defense = home_strength.defensive
    away_attack = away_strength.attacking
    away_defense = away_strength.defensive
    expected_home = league_avg_goals * home_attack * away_defense * (1 + home_strength.home_advantage)
    expected_away = league_avg_goals * away_attack * home_defense
    expected_home = max(expected_home, 0.1)
    expected_away = max(expected_away, 0.1)
    return expected_home, expected_away


def predict_match(
    home: TeamInfo,
    away: TeamInfo,
    espn_win_probs: tuple[float, float, float] | None = None,
    league_avg_goals: float = 2.5,
    elo_weight: float = 0.3,
) -> MatchPrediction:
    home_strength = TeamStrength(
        name=home.name,
        espn_id=home.espn_id,
        elo=home.elo,
        attacking=home.attacking,
        defensive=home.defensive,
        home_advantage=0.05,
    )
    away_strength = TeamStrength(
        name=away.name,
        espn_id=away.espn_id,
        elo=away.elo,
        attacking=away.attacking,
        defensive=away.defensive,
        home_advantage=0.0,
    )
    exp_goals_home, exp_goals_away = calculate_expected_goals(
        home_strength, away_strength, league_avg_goals
    )
    poisson_home, poisson_draw, poisson_away = poisson_match_probabilities(
        exp_goals_home, exp_goals_away
    )
    if espn_win_probs:
        espn_home, espn_draw, espn_away = espn_win_probs
        blended_home, blended_draw, blended_away = adjust_probabilities_with_elo(
            espn_home, espn_draw, espn_away,
            home.elo, away.elo,
            home_advantage=0.05,
            weight=elo_weight,
        )
    else:
        blended_home, blended_draw, blended_away = poisson_home, poisson_draw, poisson_away
    final_home, final_draw, final_away = remove_vig(blended_home, blended_draw, blended_away)
    if espn_win_probs:
        prob_diff = max(
            abs(final_home - espn_win_probs[0]),
            abs(final_draw - espn_win_probs[1]),
            abs(final_away - espn_win_probs[2]),
        )
        confidence = max(0.5, 1.0 - prob_diff * 2)
    else:
        confidence = 0.6
    return MatchPrediction(
        home_win=round(final_home, 4),
        draw=round(final_draw, 4),
        away_win=round(final_away, 4),
        expected_goals_home=round(exp_goals_home, 2),
        expected_goals_away=round(exp_goals_away, 2),
        home_strength=round(compute_team_strength(home.elo, home.attacking, home.defensive, True), 2),
        away_strength=round(compute_team_strength(away.elo, away.attacking, away.defensive, False), 2),
        confidence=round(confidence, 4),
    )


def extract_espn_win_probs(
    event_data: dict[str, Any],
) -> tuple[float, float, float] | None:
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


def extract_odds(
    event_data: dict[str, Any],
) -> list[dict[str, Any]] | None:
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


def parse_odds_into_probs(
    odds_list: list[dict[str, Any]],
) -> tuple[float, float, float] | None:
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
