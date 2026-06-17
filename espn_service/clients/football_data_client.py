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
class FootballDataGoal:
    minute: int
    injury_time: int | None
    type: str
    team_id: int
    team_name: str
    scorer_id: int | None
    scorer_name: str | None
    assist_id: int | None
    assist_name: str | None
    score_home: int
    score_away: int


@dataclass
class FootballDataBooking:
    minute: int
    team_id: int
    team_name: str
    player_id: int | None
    player_name: str | None
    card: str


@dataclass
class FootballDataSubstitution:
    minute: int
    team_id: int
    team_name: str
    player_out_id: int | None
    player_out_name: str | None
    player_in_id: int | None
    player_in_name: str | None


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
    score_halftime_home: int | None = None
    score_halftime_away: int | None = None
    goals: list[FootballDataGoal] | None = None
    bookings: list[FootballDataBooking] | None = None
    substitutions: list[FootballDataSubstitution] | None = None


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

    def _request(self, path: str) -> dict[str, Any] | None:
        url = f"{FOOTBALL_DATA_BASE}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
                resp = client.get(url)
                if resp.status_code == 429:
                    logger.warning("football_data_rate_limited")
                    return None
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("football_data_http_error", url=url, status=e.response.status_code, error=str(e))
            return None
        except httpx.RequestError as e:
            logger.error("football_data_request_error", url=url, error=str(e))
            return None

    def get_competition_matches(self, competition_id: int = WC_COMPETITION_ID) -> list[FootballDataMatch]:
        data = self._request(f"competitions/{competition_id}/matches")
        if data is None:
            return []
        matches = []
        for m in data.get("matches", []):
            goals_list = None
            raw_goals = m.get("goals")
            if raw_goals:
                goals_list = []
                for g in raw_goals:
                    scorer = g.get("scorer") or {}
                    assist = g.get("assist") or {}
                    team = g.get("team") or {}
                    goals_list.append(FootballDataGoal(
                        minute=g.get("minute", 0),
                        injury_time=g.get("injuryTime"),
                        type=g.get("type", "GOAL"),
                        team_id=team.get("id", 0),
                        team_name=team.get("name", ""),
                        scorer_id=scorer.get("id"),
                        scorer_name=scorer.get("name"),
                        assist_id=assist.get("id"),
                        assist_name=assist.get("name"),
                        score_home=(g.get("score") or {}).get("home", 0),
                        score_away=(g.get("score") or {}).get("away", 0),
                    ))
            bookings_list = None
            raw_bookings = m.get("bookings")
            if raw_bookings:
                bookings_list = []
                for b in raw_bookings:
                    player = b.get("player") or {}
                    team = b.get("team") or {}
                    bookings_list.append(FootballDataBooking(
                        minute=b.get("minute", 0),
                        team_id=team.get("id", 0),
                        team_name=team.get("name", ""),
                        player_id=player.get("id"),
                        player_name=player.get("name"),
                        card=b.get("card", "YELLOW_CARD"),
                    ))
            subs_list = None
            raw_subs = m.get("substitutions")
            if raw_subs:
                subs_list = []
                for s in raw_subs:
                    player_out = s.get("playerOut") or {}
                    player_in = s.get("playerIn") or {}
                    team = s.get("team") or {}
                    subs_list.append(FootballDataSubstitution(
                        minute=s.get("minute", 0),
                        team_id=team.get("id", 0),
                        team_name=team.get("name", ""),
                        player_out_id=player_out.get("id"),
                        player_out_name=player_out.get("name"),
                        player_in_id=player_in.get("id"),
                        player_in_name=player_in.get("name"),
                    ))
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
                score_halftime_home=m.get("score", {}).get("halfTime", {}).get("home"),
                score_halftime_away=m.get("score", {}).get("halfTime", {}).get("away"),
                goals=goals_list,
                bookings=bookings_list,
                substitutions=subs_list,
            ))
        return matches

    def get_competition_standings(self, competition_id: int = WC_COMPETITION_ID) -> dict[str, list[FootballDataStanding]]:
        data = self._request(f"competitions/{competition_id}/standings")
        if data is None:
            return {}
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
        return data.get("teams", []) if data else []

    def get_match(self, match_id: int) -> dict[str, Any] | None:
        return self._request(f"matches/{match_id}")
