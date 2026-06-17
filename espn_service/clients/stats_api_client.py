"""Client for TheStatsAPI (thestatsapi.com).

Provides football match data, odds, standings, player stats, and xG.
"""
from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

STATS_API_BASE = "https://api.thestatsapi.com/api/football"
DEFAULT_API_KEY = "fapi_I1AksxZz9ZbpF4pMEZJuPpJKAvj9RzlU"


class StatsAPIClient:
    """Client for TheStatsAPI (football data + odds)."""

    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.api_key = api_key or DEFAULT_API_KEY
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        url = f"{STATS_API_BASE}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout, headers=self._headers()) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    logger.warning("stats_api_rate_limited")
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("stats_api_http_error", url=url, status=e.response.status_code, body=e.response.text[:500])
            raise
        except httpx.TimeoutException:
            logger.error("stats_api_timeout", url=url)
            raise
        except Exception as e:
            logger.error("stats_api_error", url=url, error=str(e))
            raise

    # ── Competitions ──

    def get_competitions(
        self,
        page: int = 1,
        per_page: int = 50,
        country: str | None = None,
    ) -> dict[str, Any]:
        """List football competitions."""
        params = {"page": page, "per_page": per_page}
        if country:
            params["country"] = country
        return self._request("competitions", params)

    def get_competition(self, competition_id: str) -> dict[str, Any]:
        return self._request(f"competitions/{competition_id}")

    def get_competition_seasons(self, competition_id: str) -> dict[str, Any]:
        return self._request(f"competitions/{competition_id}/seasons")

    def get_standings(
        self, competition_id: str, season_id: str, group: str | None = None
    ) -> dict[str, Any]:
        params = {}
        if group:
            params["group"] = group
        return self._request(f"competitions/{competition_id}/seasons/{season_id}/standings", params)

    def get_groups(self, competition_id: str, season_id: str) -> dict[str, Any]:
        return self._request(f"competitions/{competition_id}/seasons/{season_id}/groups")

    def get_teams(
        self, competition_id: str | None = None, page: int = 1, per_page: int = 50
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if competition_id:
            params["competition_id"] = competition_id
        return self._request("teams", params)

    def get_team(self, team_id: str) -> dict[str, Any]:
        return self._request(f"teams/{team_id}")

    def get_players(
        self,
        team_id: str | None = None,
        competition_id: str | None = None,
        page: int = 1,
        per_page: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if team_id:
            params["team_id"] = team_id
        if competition_id:
            params["competition_id"] = competition_id
        return self._request("players", params)

    def get_player(self, player_id: str) -> dict[str, Any]:
        return self._request(f"players/{player_id}")

    def get_player_stats(
        self, player_id: str, season_id: str | None = None
    ) -> dict[str, Any]:
        params = {}
        if season_id:
            params["season_id"] = season_id
        return self._request(f"players/{player_id}/stats", params)

    # ── Matches ──

    def get_matches(
        self,
        competition_id: str | None = None,
        team_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        per_page: int = 50,
        status: str | None = None,
        group: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if competition_id:
            params["competition_id"] = competition_id
        if team_id:
            params["team_id"] = team_id
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if status:
            params["status"] = status
        if group:
            params["group"] = group
        return self._request("matches", params)

    def get_match(self, match_id: str) -> dict[str, Any]:
        return self._request(f"matches/{match_id}")

    def get_match_stats(self, match_id: str) -> dict[str, Any]:
        return self._request(f"matches/{match_id}/stats")

    def get_match_events(self, match_id: str) -> dict[str, Any]:
        return self._request(f"matches/{match_id}/events")

    def get_match_lineups(self, match_id: str) -> dict[str, Any]:
        return self._request(f"matches/{match_id}/lineups")

    def get_match_head_to_head(self, match_id: str) -> dict[str, Any]:
        return self._request(f"matches/{match_id}/h2h")

    # ── Odds ──

    def get_match_odds(self, match_id: str) -> dict[str, Any]:
        """Pre-match odds from all bookmakers."""
        return self._request(f"matches/{match_id}/odds")

    def get_match_live_odds(self, match_id: str) -> dict[str, Any]:
        """Live in-play odds."""
        return self._request(f"matches/{match_id}/odds/live")

    def get_historical_odds(
        self,
        match_id: str,
        bookmaker: str | None = None,
    ) -> dict[str, Any]:
        """Historical odds timeline for a match."""
        params = {}
        if bookmaker:
            params["bookmaker"] = bookmaker
        return self._request(f"matches/{match_id}/odds/history", params)

    def search_matches(self, query: str, page: int = 1, per_page: int = 20) -> dict[str, Any]:
        """Search matches by team or competition name."""
        return self._request("matches", {"search": query, "page": page, "per_page": per_page})
