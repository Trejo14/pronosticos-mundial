"""Feature engineering: FIFA rankings, Elo ratings, and team strength calculations.

Uses the ESPN API client to gather data and compute team strength metrics
for the prediction model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TeamStrength:
    name: str
    espn_id: str
    elo: float
    attacking: float
    defensive: float
    home_advantage: float = 0.0


INITIAL_ELO = 1500.0
ELO_K = 32.0
HOME_ADVANTAGE_BASE = 0.05


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    score_b: float,
    k: float = ELO_K,
    margin: float = 0.0,
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    margin_multiplier = 1.0 + math.log(abs(margin) + 1.0) * 0.1 if margin else 1.0
    actual_a = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
    k_adjusted = k * margin_multiplier
    new_a = rating_a + k_adjusted * (actual_a - expected_a)
    new_b = rating_b + k_adjusted * ((1.0 - actual_a) - expected_b)
    return new_a, new_b


def fifa_ranking_to_elo(fifa_rank: int | None, max_rank: int = 211) -> float:
    if fifa_rank is None or fifa_rank <= 0:
        return INITIAL_ELO
    normalized = 1.0 - (fifa_rank - 1) / max_rank
    return 1000.0 + normalized * 1000.0


def calculate_attacking_strength(
    goals_scored: float,
    league_avg_goals: float = 1.5,
) -> float:
    if league_avg_goals <= 0:
        return 1.0
    return goals_scored / league_avg_goals


def calculate_defensive_strength(
    goals_conceded: float,
    league_avg_goals: float = 1.5,
) -> float:
    if league_avg_goals <= 0:
        return 1.0
    return goals_conceded / league_avg_goals


def extract_team_stats_from_standings(standings_data: dict[str, Any]) -> dict[str, dict]:
    teams: dict[str, dict] = {}
    children = standings_data.get("children", [])
    for group in children:
        entries = group.get("standings", {}).get("entries", [])
        for entry in entries:
            team = entry.get("team", {})
            tid = str(team.get("id", ""))
            stats = {}
            for s in entry.get("stats", []):
                name = s.get("name", "")
                val = s.get("displayValue", "0")
                try:
                    stats[name] = float(val) if val.replace(".", "").replace("-", "").isdigit() else val
                except (ValueError, TypeError):
                    stats[name] = val
            teams[tid] = {
                "id": tid,
                "name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "draws": stats.get("ties", 0) or stats.get("draws", 0),
                "points": stats.get("points", 0),
                "goals_for": stats.get("goalsFor", 0) or stats.get("pointsFor", 0),
                "goals_against": stats.get("goalsAgainst", 0) or stats.get("pointsAgainst", 0),
                "goal_differential": stats.get("goalDifferential", 0) or stats.get("pointDifferential", 0),
                "form": stats.get("form", ""),
                "games_played": stats.get("gamesPlayed", 0) or (stats.get("wins", 0) + stats.get("losses", 0) + stats.get("ties", 0)),
                "rank": stats.get("rank", 0) or stats.get("position", 0),
            }
    return teams


def extract_team_stats_from_event(event_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    events = event_data.get("events", [event_data])
    for event in events:
        for comp in event.get("competitions", []):
            for competitor in comp.get("competitors", []):
                team = competitor.get("team", {})
                tid = str(team.get("id", ""))
                home_away = competitor.get("homeAway", "away")
                score = competitor.get("score", "0")
                records = competitor.get("records", [])
                record_str = ""
                for r in records:
                    if r.get("name") in ("overall", "all"):
                        record_str = r.get("summary", "")
                        break
                teams[tid] = {
                    "id": tid,
                    "name": team.get("displayName", ""),
                    "abbreviation": team.get("abbreviation", ""),
                    "home_away": home_away,
                    "score": score,
                    "record": record_str,
                    "logo": team.get("logo", ""),
                }
    return teams


def compute_team_strength(
    elo: float = INITIAL_ELO,
    attacking: float = 1.0,
    defensive: float = 1.0,
    home: bool = False,
) -> float:
    strength = elo / 1000.0
    strength *= attacking
    strength /= max(defensive, 0.1)
    if home:
        strength *= (1.0 + HOME_ADVANTAGE_BASE)
    return strength
