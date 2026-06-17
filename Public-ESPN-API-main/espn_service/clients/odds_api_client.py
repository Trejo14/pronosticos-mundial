"""Client for The Odds API (the-odds-api.com).

Provides betting odds from multiple bookmakers for use in edge/value analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_API_KEY = "fapi_I1AksxZz9ZbpF4pMEZJuPpJKAvj9RzlU"


@dataclass
class OddsApiResponse:
    data: list[dict[str, Any]] | dict[str, Any]
    status_code: int
    remaining_requests: int = 0


class OddsApiClient:
    """Client for The Odds API v4."""

    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or DEFAULT_API_KEY
        self.timeout = timeout

    def _request(self, path: str, params: dict[str, Any] | None = None) -> OddsApiResponse:
        url = f"{ODDS_API_BASE}/{path.lstrip('/')}"
        all_params: dict[str, Any] = {"apiKey": self.api_key}
        if params:
            all_params.update(params)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, params=all_params)
                remaining = int(resp.headers.get("x-requests-remaining", 0))
                if resp.status_code == 429:
                    logger.warning("odds_api_rate_limited", remaining=remaining)
                resp.raise_for_status()
                data = resp.json()
                return OddsApiResponse(
                    data=data,
                    status_code=resp.status_code,
                    remaining_requests=remaining,
                )
        except httpx.HTTPStatusError as e:
            logger.error("odds_api_http_error", url=url, status=e.response.status_code)
            raise
        except httpx.TimeoutException:
            logger.error("odds_api_timeout", url=url)
            raise
        except Exception as e:
            logger.error("odds_api_error", url=url, error=str(e))
            raise

    def get_sports(self) -> OddsApiResponse:
        """List all available sports."""
        return self._request("sports")

    def get_sport_odds(
        self,
        sport_key: str,
        regions: str = "us,uk,eu",
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
        event_ids: str | None = None,
    ) -> OddsApiResponse:
        """Get odds for a sport.

        Args:
            sport_key: e.g. 'soccer_fifa_world_cup', 'soccer_epl'
            regions: Comma-separated ('us', 'uk', 'eu', 'au')
            markets: 'h2h' (moneyline), 'spreads', 'totals'
            bookmakers: Comma-separated bookmaker keys (optional)
            event_ids: Comma-separated event IDs (optional)
        """
        params: dict[str, Any] = {
            "regions": regions,
            "markets": markets,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        if event_ids:
            params["eventIds"] = event_ids
        return self._request(f"sports/{sport_key}/odds", params)

    def get_sport_scores(
        self,
        sport_key: str,
        days_from: int | None = None,
        event_ids: str | None = None,
    ) -> OddsApiResponse:
        """Get scores for a sport."""
        params: dict[str, Any] = {}
        if days_from is not None:
            params["daysFrom"] = days_from
        if event_ids:
            params["eventIds"] = event_ids
        return self._request(f"sports/{sport_key}/scores", params)

    def get_event_odds(
        self,
        sport_key: str,
        event_id: str,
        regions: str = "us,uk,eu",
        markets: str = "h2h,spreads,totals",
    ) -> OddsApiResponse:
        """Get odds for a single event."""
        params = {"regions": regions, "markets": markets}
        return self._request(f"sports/{sport_key}/events/{event_id}/odds", params)

    def get_event_scores(
        self,
        sport_key: str,
        event_id: str,
    ) -> OddsApiResponse:
        """Get scores for a single event."""
        return self._request(f"sports/{sport_key}/events/{event_id}/scores")


# Mapping from ESPN league slugs to The Odds API sport keys
LEAGUE_TO_ODDS_API_KEY: dict[str, str] = {
    # Soccer
    "fifa.world": "soccer_fifa_world_cup",
    "fifa.wwc": "soccer_fifa_womens_world_cup",
    "uefa.champions": "soccer_uefa_champions_league",
    "uefa.europa": "soccer_uefa_europa_league",
    "eng.1": "soccer_epl",
    "esp.1": "soccer_spain_la_liga",
    "ger.1": "soccer_germany_bundesliga",
    "ita.1": "soccer_italy_serie_a",
    "fra.1": "soccer_france_ligue_one",
    "usa.1": "soccer_usa_mls",
    "mex.1": "soccer_mexico_liga_mx",
    "uefa.euro": "soccer_european_championship",
    "conmebol.america": "soccer_copa_america",
    # Basketball
    "nba": "basketball_nba",
    "wnba": "basketball_wnba",
    "mens-college-basketball": "basketball_ncaab",
    "euroleague": "basketball_euroleague",
    # Football
    "nfl": "americanfootball_nfl",
    "college-football": "americanfootball_ncaaf",
    "cfl": "americanfootball_cfl",
    # Baseball
    "mlb": "baseball_mlb",
    "college-baseball": "baseball_ncaa",
    # Hockey
    "nhl": "icehockey_nhl",
}

def get_odds_api_key_for_league(league_slug: str) -> str | None:
    """Convert ESPN league slug to The Odds API sport key."""
    return LEAGUE_TO_ODDS_API_KEY.get(league_slug)
