"""Client for 365Scores public API.

Provides live match statistics (player stats, xG chart events, match events)
for football matches. No API key required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

SCORES365_BASE = "https://webws.365scores.com/web"
WC_COMPETITION_ID = 5930

# Stat type IDs mapped to meaningful names
STAT_TYPE_SHOULDER = 3    # total shots
STAT_TYPE_SOT = 4         # shots on target
STAT_TYPE_SOFT = 5        # shots off target
STAT_TYPE_BLOCKED = 6     # blocked shots
STAT_TYPE_OFFSIDES = 9
STAT_TYPE_PASSES = 19     # completed passes (string like "23/30 (77%)")
STAT_TYPE_SAVES = 23
STAT_TYPE_FOULS = 42      # fouls committed
STAT_TYPE_FOULS_SUF = 37  # fouls suffered
STAT_TYPE_TACKLES = 39
STAT_TYPE_CLEAR = 40
STAT_TYPE_INTER = 41
STAT_TYPE_XG = 76
STAT_TYPE_XA = 78
STAT_TYPE_RECOVERIES = 86
STAT_TYPE_DRIBBLES = 54   # dribbles completed
STAT_TYPE_CROSSES = 52    # crosses completed
STAT_TYPE_KEY_PASSES = 46
STAT_TYPE_CORNERS = 6     # from events, not player stats — computed from chartEvents subType=2

# Chart event subtypes
CHART_SUBTYPE_CORNER = 2
CHART_SUBTYPE_OWN_GOAL = 10


@dataclass
class Scores365PlayerStat:
    type_id: int
    value: str
    name: str
    category_id: int
    is_top: bool = False


@dataclass
class Scores365Player:
    id: int
    name: str
    shirt_num: int
    position: int
    position_name: str
    stats: list[Scores365PlayerStat] = field(default_factory=list)
    is_substitute: bool = False


@dataclass
class Scores365ChartEvent:
    xg: float
    xgot: float
    body_part: str
    time: str
    competitor_num: int
    player_id: int
    outcome_id: int
    outcome_name: str
    sub_type: int
    status: int  # 6=first half, 8=second half


@dataclass
class Scores365MatchEvent:
    id: int
    type: int
    minute: int
    add_time: int | None
    competitor_id: int
    player_id: int | None
    player_name: str | None
    is_major: bool
    extra_info: dict[str, Any] | None = None


@dataclass
class Scores365GameStats:
    game_id: int
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_score: int | None
    away_score: int | None
    status_group: int
    game_time: float
    game_time_display: str
    players: list[Scores365Player] = field(default_factory=list)
    chart_events: list[Scores365ChartEvent] = field(default_factory=list)
    match_events: list[Scores365MatchEvent] = field(default_factory=list)

    @property
    def home_players(self) -> list:
        return [p for p in self.players if not hasattr(p, '_is_away')]

    @property
    def away_players(self) -> list:
        return [p for p in self.players if hasattr(p, '_is_away') and p._is_away]

    def _parse_stat_value(self, raw: str) -> float:
        """Parse stat values like '5', '5/6 (83.3%)', '23/30 (77%)' into a number."""
        val = raw.split()[0] if " " in raw else raw
        # Handle "5/6" fractions — take the first number
        if "/" in val:
            val = val.split("/")[0]
        val = val.replace(",", ".").strip()
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _aggregate_stat(self, team_competitor_num: int, stat_type: int) -> int:
        total = 0
        for p in self.players:
            comp_num = getattr(p, '_competitor_num', 0)
            if comp_num != team_competitor_num:
                continue
            for s in p.stats:
                if s.type_id == stat_type:
                    total += int(self._parse_stat_value(s.value))
        return total

    def _aggregate_stat_float(self, team_competitor_num: int, stat_type: int) -> float:
        total = 0.0
        for p in self.players:
            comp_num = getattr(p, '_competitor_num', 0)
            if comp_num != team_competitor_num:
                continue
            for s in p.stats:
                if s.type_id == stat_type:
                    total += self._parse_stat_value(s.value)
        return total

    def home_stats(self) -> dict[str, Any]:
        return {
            "shots": self._aggregate_stat(1, STAT_TYPE_SHOULDER),
            "shots_on_target": self._aggregate_stat(1, STAT_TYPE_SOT),
            "shots_off_target": self._aggregate_stat(1, STAT_TYPE_SOFT),
            "blocked_shots": self._aggregate_stat(1, STAT_TYPE_BLOCKED),
            "fouls": self._aggregate_stat(1, STAT_TYPE_FOULS),
            "offsides": self._aggregate_stat(1, STAT_TYPE_OFFSIDES),
            "saves": self._aggregate_stat(1, STAT_TYPE_SAVES),
            "tackles": self._aggregate_stat(1, STAT_TYPE_TACKLES),
            "clearances": self._aggregate_stat(1, STAT_TYPE_CLEAR),
            "interceptions": self._aggregate_stat(1, STAT_TYPE_INTER),
            "recoveries": self._aggregate_stat(1, STAT_TYPE_RECOVERIES),
            "xG": round(self._aggregate_stat_float(1, STAT_TYPE_XG), 2),
            "corner_kicks": sum(1 for c in self.chart_events if c.competitor_num == 1 and c.sub_type == CHART_SUBTYPE_CORNER),
            "yellow_cards": self._count_cards(1, "YELLOW"),
            "red_cards": self._count_cards(1, "RED"),
        }

    def away_stats(self) -> dict[str, Any]:
        return {
            "shots": self._aggregate_stat(2, STAT_TYPE_SHOULDER),
            "shots_on_target": self._aggregate_stat(2, STAT_TYPE_SOT),
            "shots_off_target": self._aggregate_stat(2, STAT_TYPE_SOFT),
            "blocked_shots": self._aggregate_stat(2, STAT_TYPE_BLOCKED),
            "fouls": self._aggregate_stat(2, STAT_TYPE_FOULS),
            "offsides": self._aggregate_stat(2, STAT_TYPE_OFFSIDES),
            "saves": self._aggregate_stat(2, STAT_TYPE_SAVES),
            "tackles": self._aggregate_stat(2, STAT_TYPE_TACKLES),
            "clearances": self._aggregate_stat(2, STAT_TYPE_CLEAR),
            "interceptions": self._aggregate_stat(2, STAT_TYPE_INTER),
            "recoveries": self._aggregate_stat(2, STAT_TYPE_RECOVERIES),
            "xG": round(self._aggregate_stat_float(2, STAT_TYPE_XG), 2),
            "corner_kicks": sum(1 for c in self.chart_events if c.competitor_num == 2 and c.sub_type == CHART_SUBTYPE_CORNER),
            "yellow_cards": self._count_cards(2, "YELLOW"),
            "red_cards": self._count_cards(2, "RED"),
        }

    def _count_cards(self, competitor_num: int, card_type: str) -> int:
        count = 0
        for e in self.match_events:
            comp = getattr(e, '_competitor_num', 0)
            if comp != competitor_num:
                continue
            # 365Scores eventType.id: 2=Yellow Card, 3=Red Card (if exists)
            if card_type == "YELLOW" and e.type == 2:
                count += 1
            elif card_type == "RED" and e.type == 3:
                count += 1
        return count


class Scores365Client:
    """Client for the public 365Scores API."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def _request(self, path: str) -> dict[str, Any] | None:
        url = f"{SCORES365_BASE}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning("scores365_http_error", url=url, status=e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.warning("scores365_request_error", url=url, error=str(e))
            return None

    def _parse_value(self, v: str | None) -> str:
        return v or ""

    def get_game_stats(self, game_id: int, lang_id: int = 31) -> Scores365GameStats | None:
        path = f"/game/?gameId={game_id}&langId={lang_id}&timezoneName=America/Mexico_City"
        data = self._request(path)
        if not data or "game" not in data:
            return None

        g = data["game"]
        home = g.get("homeCompetitor", {})
        away = g.get("awayCompetitor", {})
        stages = g.get("stages", [])
        current_stage = None
        for s in stages:
            if s.get("isCurrent"):
                current_stage = s
                break
        if not current_stage and stages:
            current_stage = stages[-1]

        result = Scores365GameStats(
            game_id=g["id"],
            home_team_id=home.get("id", 0),
            home_team_name=home.get("name", ""),
            away_team_id=away.get("id", 0),
            away_team_name=away.get("name", ""),
            home_score=current_stage.get("homeCompetitorScore") if current_stage else home.get("score"),
            away_score=current_stage.get("awayCompetitorScore") if current_stage else away.get("score"),
            status_group=g.get("statusGroup", 0),
            game_time=g.get("gameTime", -1.0),
            game_time_display=g.get("gameTimeDisplay", ""),
        )

        # Build player name lookup from game-level members
        player_names: dict[int, dict[str, Any]] = {}
        for gm in g.get("members", []):
            pid = gm.get("id", 0)
            if pid:
                player_names[pid] = gm

        # Parse player stats
        home_team_id = home.get("id", 0)
        away_team_id = away.get("id", 0)
        for competitor_num, competitor_key in [(1, "homeCompetitor"), (2, "awayCompetitor")]:
            comp = g.get(competitor_key, {})
            team_id = comp.get("id", 0)
            lineups = comp.get("lineups", {})
            if not lineups:
                continue
            members = lineups.get("members", [])
            for m in members:
                player_id = m.get("id", 0)
                pdata = player_names.get(player_id, {})
                player_name = pdata.get("name", "")
                shirt_num = pdata.get("jerseyNumber", 0)
                pos = m.get("position", {})
                position = pos.get("id", 0) if isinstance(pos, dict) else pos
                position_name = pos.get("name", "") if isinstance(pos, dict) else ""
                is_sub = m.get("status", 0) != 1
                has_stats = m.get("hasStats", False)

                player = Scores365Player(
                    id=player_id,
                    name=player_name,
                    shirt_num=shirt_num,
                    position=position,
                    position_name=position_name,
                    is_substitute=is_sub,
                )
                player._competitor_num = competitor_num
                player._is_away = competitor_num == 2
                player._team_id = team_id

                if has_stats:
                    raw_stats = m.get("stats", [])
                    for s in raw_stats:
                        player.stats.append(Scores365PlayerStat(
                            type_id=s.get("type", 0),
                            value=self._parse_value(s.get("value")),
                            name=s.get("name", ""),
                            category_id=s.get("categoryId", 0),
                            is_top=s.get("isTop", False),
                        ))
                result.players.append(player)

        # Parse chart events (xG shots)
        chart_events = g.get("chartEvents", {})
        for ce in chart_events.get("events", []):
            outcome = ce.get("outcome", {})
            result.chart_events.append(Scores365ChartEvent(
                xg=float(ce.get("xg", 0)),
                xgot=float(ce.get("xgot", 0)),
                body_part=self._parse_value(ce.get("bodyPart")),
                time=self._parse_value(ce.get("time")),
                competitor_num=ce.get("competitorNum", 0),
                player_id=ce.get("playerId", 0),
                outcome_id=outcome.get("id", -1),
                outcome_name=outcome.get("name", ""),
                sub_type=ce.get("subType", -1),
                status=ce.get("status", 0),
            ))

        # Parse match events (goals, cards, subs)
        # eventType.id: 1=Goal, 2=Yellow Card, 1000=Substitution
        match_events = g.get("events", [])
        for e in match_events:
            et = e.get("eventType", {})
            event_type_id = et.get("id", 0) if isinstance(et, dict) else et

            gtd = e.get("gameTimeDisplay", "")
            minute_int = 0
            add_time = None
            if gtd:
                gtd_clean = gtd.replace("'", "")
                parts = gtd_clean.split("+")
                try:
                    minute_int = int(parts[0])
                except (ValueError, TypeError):
                    minute_int = 0
                if len(parts) > 1:
                    try:
                        add_time = int(parts[1])
                    except (ValueError, TypeError):
                        pass
            else:
                minute_int = int(e.get("gameTime", 0))

            player_id = e.get("playerId")
            pdata = player_names.get(player_id, {}) if player_id else {}
            player_name = pdata.get("name", "")

            # Determine competitor_num from team_id
            comp_id = e.get("competitorId", 0)
            comp_num = 1 if comp_id == home_team_id else (2 if comp_id == away_team_id else 0)

            result.match_events.append(Scores365MatchEvent(
                id=e.get("id", 0),
                type=event_type_id,
                minute=minute_int,
                add_time=add_time,
                competitor_id=comp_id,
                player_id=player_id,
                player_name=player_name,
                is_major=e.get("isMajor", False),
                extra_info={"extraPlayers": e.get("extraPlayers", [])},
            ))
            # Tag with competitor_num for card counting
            result.match_events[-1]._competitor_num = comp_num

        return result

    def find_game_by_teams(
        self,
        home_team_name: str,
        away_team_name: str,
        competition_id: int = WC_COMPETITION_ID,
        lang_id: int = 31,
    ) -> int | None:
        """Find a 365Scores game ID by matching team names."""
        from difflib import SequenceMatcher

        def _similar(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        def _best_name_score(name_a: str, name_b: str) -> float:
            clean = name_b.replace("Seleção ", "").replace("Seleção do ", "").replace("Seleção da ", "").strip()
            return max(_similar(name_a, name_b), _similar(name_a, clean))

        def _score_game(g: dict) -> float:
            h = g.get("homeCompetitor", {}).get("name", "")
            a = g.get("awayCompetitor", {}).get("name", "")
            return (_best_name_score(home_team_name, h) + _best_name_score(away_team_name, a)) / 2

        best_match = None
        best_score = 0.0

        for status_group in (2, 1):
            for g in self.get_competition_games(competition_id, status_group, lang_id):
                score = _score_game(g)
                if score > best_score and score > 0.6:
                    best_score, best_match = score, g.get("id")
            if best_match:
                break

        if not best_match:
            for g in self.get_competition_results(competition_id, lang_id):
                score = _score_game(g)
                if score > best_score and score > 0.6:
                    best_score, best_match = score, g.get("id")

        return best_match

    def get_competition_games(
        self,
        competition_id: int = WC_COMPETITION_ID,
        status_group: int = 1,
        lang_id: int = 31,
    ) -> list[dict[str, Any]]:
        path = (
            f"/games/?langId={lang_id}&timezoneName=America/Mexico_City"
            f"&competitionId={competition_id}&statusGroup={status_group}"
        )
        data = self._request(path)
        if not data:
            return []
        games = []
        for sport in data.get("sports", []):
            for country in sport.get("countries", []):
                for comp in country.get("competitions", []):
                    games.extend(comp.get("games", []))
        return games

    def get_competition_results(
        self,
        competition_id: int = WC_COMPETITION_ID,
        lang_id: int = 31,
    ) -> list[dict[str, Any]]:
        path = (
            f"/games/results/?langId={lang_id}&timezoneName=America/Mexico_City"
            f"&competitions={competition_id}"
        )
        data = self._request(path)
        if not data:
            return []
        return data.get("games", [])
