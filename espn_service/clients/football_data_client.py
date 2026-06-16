"""Client for Football-data.org API v4.

Provides match, team, competition, and standings data for World Cup 2026.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from django.conf import settings

logger = structlog.get_logger(__name__)

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
WC_COMPETITION_ID = 2000


@dataclass
class FootballDataMatch:
    id: int
    utc_date: str
    status: str
    matchday: int
    stage: str
    group: str | None
    home_team_id: int
    home_team_name: str
    home_team_short: str
    home_team_tla: str
    home_team_crest: str
    away_team_id: int
    away_team_name: str
    away_team_short: str
    away_team_tla: str
    away_team_crest: str
    score_home: int | None
    score_away: int | None
    winner: str | None


@dataclass
class FootballDataStanding:
    position: int
    team_id: int
    team_name: str
    team_short: str
    team_tla: str
    team_crest: str
    played: int
    won: int
    draw: int
    lost: int
    points: int
    goals_for: int
    goals_against: int
    goal_difference: int


class FootballDataClient:
    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or getattr(settings, "FOOTBALL_DATA_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-Auth-Token": self.api_key}

    def _request(self, path: str) -> dict[str, Any]:
        url = f"{FOOTBALL_DATA_BASE}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    logger.warning("football_data_rate_limited")
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("football_data_http_error", url=url, status=e.response.status_code, error=str(e))
            raise
        except httpx.RequestError as e:
            logger.error("football_data_request_error", url=url, error=str(e))
            raise

    def get_competition_matches(self, competition_id: int = WC_COMPETITION_ID) -> list[FootballDataMatch]:
        data = self._request(f"competitions/{competition_id}/matches")
        matches = []
        for m in data.get("matches", []):
            matches.append(FootballDataMatch(
                id=m["id"],
                utc_date=m["utcDate"],
                status=m["status"],
                matchday=m.get("matchday", 0),
                stage=m.get("stage", ""),
                group=m.get("group"),
                home_team_id=m["homeTeam"]["id"],
                home_team_name=m["homeTeam"]["name"],
                home_team_short=m["homeTeam"].get("shortName", m["homeTeam"]["name"]),
                home_team_tla=m["homeTeam"].get("tla", ""),
                home_team_crest=m["homeTeam"].get("crest", ""),
                away_team_id=m["awayTeam"]["id"],
                away_team_name=m["awayTeam"]["name"],
                away_team_short=m["awayTeam"].get("shortName", m["awayTeam"]["name"]),
                away_team_tla=m["awayTeam"].get("tla", ""),
                away_team_crest=m["awayTeam"].get("crest", ""),
                score_home=m.get("score", {}).get("fullTime", {}).get("home"),
                score_away=m.get("score", {}).get("fullTime", {}).get("away"),
                winner=m.get("score", {}).get("winner"),
            ))
        return matches

    def get_competition_standings(self, competition_id: int = WC_COMPETITION_ID) -> dict[str, list[FootballDataStanding]]:
        data = self._request(f"competitions/{competition_id}/standings")
        groups: dict[str, list[FootballDataStanding]] = {}
        for s in data.get("standings", []):
            group_name = s.get("group", "TOTAL")
            table = []
            for entry in s.get("table", []):
                t = entry["team"]
                table.append(FootballDataStanding(
                    position=entry["position"],
                    team_id=t["id"],
                    team_name=t["name"],
                    team_short=t.get("shortName", t["name"]),
                    team_tla=t.get("tla", ""),
                    team_crest=t.get("crest", ""),
                    played=entry.get("playedGames", 0),
                    won=entry.get("won", 0),
                    draw=entry.get("draw", 0),
                    lost=entry.get("lost", 0),
                    points=entry.get("points", 0),
                    goals_for=entry.get("goalsFor", 0),
                    goals_against=entry.get("goalsAgainst", 0),
                    goal_difference=entry.get("goalDifference", 0),
                ))
            groups[group_name] = table
        return groups

    def get_competition_teams(self, competition_id: int = WC_COMPETITION_ID) -> list[dict[str, Any]]:
        data = self._request(f"competitions/{competition_id}/teams")
        return data.get("teams", [])

    def get_match(self, match_id: int) -> dict[str, Any]:
        return self._request(f"matches/{match_id}")
