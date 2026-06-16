"""Orchestration service: ties ESPN API client with prediction engine.

This is the main service that fetches data from ESPN, runs predictions,
and returns structured results.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from django.conf import settings

from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    extract_team_stats_from_event,
    extract_team_stats_from_standings,
    fifa_ranking_to_elo,
)
from apps.predictions.odds_analyzer import (
    analyze_outcome,
    find_best_odds,
    league_margin_to_prob,
)
from apps.predictions.prediction_engine import (
    TeamInfo,
    extract_espn_win_probs,
    extract_odds,
    parse_odds_into_probs,
    predict_match,
)
from apps.predictions.tournament_simulator import (
    Team,
    build_world_cup_2026_group_stage,
    simulate_tournament,
)

logger = structlog.get_logger(__name__)


class PredictionService:
    """Main prediction service that coordinates data fetching and analysis."""

    def __init__(self, espn_client=None, stats_api_client=None):
        self.client = espn_client
        self.stats_api = stats_api_client
        self._team_elo_cache: dict[str, float] = {}
        self._team_att_def_cache: dict[str, dict[str, float]] = {}
        self._stats_api_available: bool | None = None

    def set_client(self, client):
        self.client = client

    def set_stats_api(self, client):
        self.stats_api = client
        self._stats_api_available = None

    def _check_stats_api(self) -> bool:
        if self._stats_api_available is not None:
            return self._stats_api_available
        if not self.stats_api:
            self._stats_api_available = False
            return False
        try:
            self.stats_api.get_competitions(per_page=1)
            self._stats_api_available = True
            logger.info("stats_api_available")
        except Exception:
            self._stats_api_available = False
            logger.info("stats_api_not_available, falling back to ESPN odds")
        return self._stats_api_available

    def _enrich_with_stats_api_odds(self, prediction: dict, home_team: str, away_team: str) -> dict:
        if not self._check_stats_api():
            return prediction
        try:
            matches_resp = self.stats_api.get_matches(
                date_from=datetime.now().strftime("%Y-%m-%d"),
                per_page=20,
            )
            all_matches = matches_resp if isinstance(matches_resp, list) else matches_resp.get("data", [])
            target_match = None
            for m in all_matches:
                mt = m.get("home_team", {}).get("name", "") or m.get("homeTeam", "") or ""
                at = m.get("away_team", {}).get("name", "") or m.get("awayTeam", "") or ""
                if (home_team.lower() in mt.lower() or mt.lower() in home_team.lower()) and \
                   (away_team.lower() in at.lower() or at.lower() in away_team.lower()):
                    target_match = m
                    break
            if not target_match:
                return prediction
            match_id = target_match.get("id", "")
            if not match_id:
                return prediction
            odds_data = self.stats_api.get_match_odds(match_id)
            odds = odds_data.get("odds", odds_data) if isinstance(odds_data, dict) else {}
            for bookmaker, markets in odds.items():
                if not isinstance(markets, dict):
                    continue
                for market_key, outcomes in markets.items():
                    if market_key in ("1x2", "1X2", "h2h", "match_result"):
                        home_odds = outcomes.get("home")
                        draw_odds = outcomes.get("draw")
                        away_odds = outcomes.get("away")
                        if home_odds and draw_odds and away_odds:
                            from apps.predictions.odds_analyzer import league_margin_to_prob
                            h_prob, d_prob, a_prob = league_margin_to_prob(
                                float(home_odds), float(draw_odds), float(away_odds)
                            )
                            prediction.setdefault("predictions", {})
                            prediction["predictions"]["home_win"]["best_odds"] = float(home_odds)
                            prediction["predictions"]["home_win"]["implied_prob"] = round(1 / float(home_odds), 4)
                            prediction["predictions"]["draw"]["best_odds"] = float(draw_odds)
                            prediction["predictions"]["draw"]["implied_prob"] = round(1 / float(draw_odds), 4)
                            prediction["predictions"]["away_win"]["best_odds"] = float(away_odds)
                            prediction["predictions"]["away_win"]["implied_prob"] = round(1 / float(away_odds), 4)
                            prediction.setdefault("market_probabilities", {})
                            prediction["market_probabilities"] = {
                                "home": round(h_prob, 4),
                                "draw": round(d_prob, 4),
                                "away": round(a_prob, 4),
                            }
                            from apps.predictions.odds_analyzer import analyze_outcome
                            ha = analyze_outcome("Home Win", prediction["predictions"]["home_win"]["probability"], float(home_odds))
                            da = analyze_outcome("Draw", prediction["predictions"]["draw"]["probability"], float(draw_odds))
                            aa = analyze_outcome("Away Win", prediction["predictions"]["away_win"]["probability"], float(away_odds))
                            for outcome_key, analysis in [("home_win", ha), ("draw", da), ("away_win", aa)]:
                                prediction["predictions"][outcome_key]["edge"] = analysis.edge
                                prediction["predictions"][outcome_key]["expected_value"] = analysis.expected_value
                                prediction["predictions"][outcome_key]["kelly"] = analysis.kelly_fraction
                                prediction["predictions"][outcome_key]["risk"] = analysis.risk_label
                            best_analyses = [a for a in [ha, da, aa] if a.edge > 0.05 and a.kelly_fraction > 0]
                            if best_analyses:
                                best = max(best_analyses, key=lambda a: a.edge)
                                prediction["recommendation"] = {
                                    "action": "bet",
                                    "outcome": best.outcome,
                                    "edge": best.edge,
                                    "kelly_fraction": best.kelly_fraction,
                                    "message": f"Bet on {best.outcome} (edge: {best.edge:.1%}, Kelly: {best.kelly_fraction:.1%})",
                                }
                                prediction["risk"]["label"] = best.risk_label
                            break
            return prediction
        except Exception as e:
            logger.debug("stats_api_enrich_failed", error=str(e))
            return prediction

    def _get_espn_team_id_from_event(self, event_data: dict, home_away: str = "home") -> str | None:
        competitions = event_data.get("competitions", []) or (
            event_data.get("events", [{}])[0].get("competitions", []) if event_data.get("events") else []
        )
        if not competitions:
            return None
        for competitor in competitions[0].get("competitors", []):
            if competitor.get("homeAway") == home_away:
                return str(competitor.get("team", {}).get("id", ""))
        return None

    def predict_match(self, event_data: dict) -> dict[str, Any]:
        """Run a full prediction on a single event."""
        competitions = event_data.get("competitions", [])
        if not competitions and "events" in event_data:
            events = event_data.get("events", [])
            if events:
                competitions = events[0].get("competitions", [])

        if not competitions:
            return {"error": "No competition data found"}

        comp = competitions[0]
        competitors_data = comp.get("competitors", [])
        home_data = None
        away_data = None
        for c in competitors_data:
            if c.get("homeAway") == "home":
                home_data = c
            else:
                away_data = c

        if not home_data or not away_data:
            return {"error": "Could not find home/away teams"}

        home_team_data = home_data.get("team", {})
        away_team_data = away_data.get("team", {})

        home_id = str(home_team_data.get("id", ""))
        away_id = str(away_team_data.get("id", ""))
        home_name = home_team_data.get("displayName", home_team_data.get("name", "Home"))
        away_name = away_team_data.get("displayName", away_team_data.get("name", "Away"))

        home_elo = self._team_elo_cache.get(home_id, INITIAL_ELO)
        away_elo = self._team_elo_cache.get(away_id, INITIAL_ELO)
        home_att = self._team_att_def_cache.get(home_id, {}).get("attacking", 1.0)
        home_def = self._team_att_def_cache.get(home_id, {}).get("defensive", 1.0)
        away_att = self._team_att_def_cache.get(away_id, {}).get("attacking", 1.0)
        away_def = self._team_att_def_cache.get(away_id, {}).get("defensive", 1.0)

        home_info = TeamInfo(
            espn_id=home_id, name=home_name, abbreviation=home_team_data.get("abbreviation", ""),
            elo=home_elo, attacking=home_att, defensive=home_def,
        )
        away_info = TeamInfo(
            espn_id=away_id, name=away_name, abbreviation=away_team_data.get("abbreviation", ""),
            elo=away_elo, attacking=away_att, defensive=away_def,
        )

        espn_probs = extract_espn_win_probs(event_data)
        market_probs = None
        odds_list = extract_odds(event_data)

        # Try to fetch odds from separate endpoint if not in event data
        if not odds_list and self.client:
            try:
                sport_slug = "soccer"
                league_slug = "fifa.world"
                event_id = event_data.get("id", "")
                comp_id = ""
                competitions = event_data.get("competitions", [])
                if competitions:
                    comp_id = competitions[0].get("id", event_id)
                if event_id:
                    odds_resp = self.client.get_odds(sport_slug, league_slug, event_id, comp_id)
                    if odds_resp and odds_resp.data:
                        odds_items = odds_resp.data.get("items", [])
                        if odds_items:
                            odds_list = odds_items
            except Exception:
                pass

        if odds_list:
            market_probs = parse_odds_into_probs(odds_list)

        prediction = predict_match(home_info, away_info, espn_probs)

        raw_odds = find_best_odds(odds_list) if odds_list else {}

        home_decimal = raw_odds.get("home") or raw_odds.get("homeOdds")
        draw_decimal = raw_odds.get("draw") or raw_odds.get("drawOdds")
        away_decimal = raw_odds.get("away") or raw_odds.get("awayOdds")

        if home_decimal and draw_decimal and away_decimal:
            home_analysis = analyze_outcome("Home Win", prediction.home_win, float(home_decimal))
            draw_analysis = analyze_outcome("Draw", prediction.draw, float(draw_decimal))
            away_analysis = analyze_outcome("Away Win", prediction.away_win, float(away_decimal))
        elif home_decimal and away_decimal:
            home_analysis = analyze_outcome("Home Win", prediction.home_win, float(home_decimal))
            draw_analysis = None
            away_analysis = analyze_outcome("Away Win", prediction.away_win, float(away_decimal))
        else:
            home_analysis = None
            draw_analysis = None
            away_analysis = None

        from apps.predictions.odds_analyzer import calculate_risk_score

        analyses_vals = [(a.edge if a else 0) for a in [home_analysis, draw_analysis, away_analysis] if a is not None]
        kelly_vals = [(a.kelly_fraction if a else 0) for a in [home_analysis, draw_analysis, away_analysis] if a is not None]

        risk_score = calculate_risk_score(
            max(prediction.home_win, prediction.draw, prediction.away_win),
            max(analyses_vals) if analyses_vals else 0,
            max(kelly_vals) if kelly_vals else 0,
        )

        risk_label = "low" if risk_score < 0.3 else ("medium" if risk_score < 0.6 else "high")

        result = {
            "match": f"{home_name} vs {away_name}",
            "home_team": home_name,
            "away_team": away_name,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "predictions": {
                "home_win": {
                    "probability": prediction.home_win,
                    "expected_odds": round(1 / prediction.home_win, 2) if prediction.home_win > 0 else None,
                    **({"best_odds": home_analysis.best_odds, "implied_prob": home_analysis.implied_prob,
                        "edge": home_analysis.edge, "expected_value": home_analysis.expected_value,
                        "kelly": home_analysis.kelly_fraction, "risk": home_analysis.risk_label}
                       if home_analysis else {}),
                },
                "draw": {
                    "probability": prediction.draw,
                    "expected_odds": round(1 / prediction.draw, 2) if prediction.draw > 0 else None,
                    **({"best_odds": draw_analysis.best_odds, "implied_prob": draw_analysis.implied_prob,
                        "edge": draw_analysis.edge, "expected_value": draw_analysis.expected_value,
                        "kelly": draw_analysis.kelly_fraction, "risk": draw_analysis.risk_label}
                       if draw_analysis else {}),
                },
                "away_win": {
                    "probability": prediction.away_win,
                    "expected_odds": round(1 / prediction.away_win, 2) if prediction.away_win > 0 else None,
                    **({"best_odds": away_analysis.best_odds, "implied_prob": away_analysis.implied_prob,
                        "edge": away_analysis.edge, "expected_value": away_analysis.expected_value,
                        "kelly": away_analysis.kelly_fraction, "risk": away_analysis.risk_label}
                       if away_analysis else {}),
                },
            },
            "expected_goals": {
                "home": prediction.expected_goals_home,
                "away": prediction.expected_goals_away,
            },
            "team_strength": {
                "home": prediction.home_strength,
                "away": prediction.away_strength,
            },
            "risk": {
                "score": round(risk_score, 3),
                "label": risk_label,
            },
            "confidence": prediction.confidence,
            "recommendation": self._generate_recommendation(
                prediction, home_analysis, draw_analysis, away_analysis
            ),
        }

        if market_probs:
            result["market_probabilities"] = {
                "home": round(market_probs[0], 4),
                "draw": round(market_probs[1], 4),
                "away": round(market_probs[2], 4),
            }

        return result

    def _generate_recommendation(self, prediction, *analyses):
        from apps.predictions.odds_analyzer import Analysis
        best = None
        for a in analyses:
            if a and a.edge > 0.05 and a.kelly_fraction > 0:
                if best is None or a.edge > best.edge:
                    best = a
        if best is None:
            return {
                "action": "no_bet",
                "message": "No value bets detected. Market odds are efficient.",
                "reason": f"Best edge across outcomes is below 5% threshold",
            }
        return {
            "action": "bet",
            "outcome": best.outcome,
            "edge": best.edge,
            "kelly_fraction": best.kelly_fraction,
            "message": f"Moderate bet on {best.outcome} (edge: {best.edge:.1%}, Kelly: {best.kelly_fraction:.1%})",
        }

    def predict_tournament(
        self,
        teams_data: list[dict[str, Any]],
        num_simulations: int = 10000,
    ) -> dict[str, Any]:
        teams, groups = build_world_cup_2026_group_stage(
            teams_data, get_elo=self._team_elo_cache.get
        )
        results = simulate_tournament(
            teams, groups, num_simulations=num_simulations
        )
        sorted_results = sorted(
            results.values(),
            key=lambda r: r.win_tournament / max(r.total_simulations, 1),
            reverse=True,
        )
        predictions = []
        for r in sorted_results[:32]:
            predictions.append({
                "team": r.team_name,
                "team_id": r.team_espn_id,
                "win_probability": round(r.win_prob, 4),
                "reach_final_probability": round(r.final_prob, 4),
                "reach_semis_probability": round(r.semis_prob, 4),
                "reach_quarters_probability": round(r.quarters_prob, 4),
            })
        return {
            "simulations": num_simulations,
            "predictions": predictions,
        }

    def predict_upcoming_matches(
        self,
        league: str = "fifa.world",
        days_ahead: int = 7,
    ) -> list[dict[str, Any]]:
        if not self.client:
            return [{"error": "ESPN client not configured"}]
        try:
            scoreboard = self.client.get_scoreboard("soccer", league)
            data = scoreboard.data
        except Exception as e:
            logger.error("Failed to fetch scoreboard", league=league, error=str(e))
            return [{"error": f"Failed to fetch scoreboard: {e}"}]

        events = data.get("events", [])
        results = []
        now = datetime.now()

        from django.utils import timezone

        for event in events:
            event_date_str = event.get("date", "")
            try:
                event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                if event_date_str:
                    try:
                        event_date = datetime.strptime(event_date_str, "%Y-%m-%dT%H:%MZ")
                    except ValueError:
                        continue
                else:
                    continue

            if event_date < now:
                continue
            if (event_date - now).days > days_ahead:
                continue

            prediction = self.predict_match(event)
            prediction["event_id"] = event.get("id", "")
            prediction["match_date"] = event_date.isoformat()
            if self.stats_api:
                prediction = self._enrich_with_stats_api_odds(
                    prediction,
                    prediction.get("home_team", ""),
                    prediction.get("away_team", ""),
                )
            results.append(prediction)

        results.sort(key=lambda x: x.get("match_date", ""))
        return results

    def predict_worldcup(
        self,
        football_data_client=None,
    ) -> dict[str, Any]:
        from clients.football_data_client import FootballDataClient
        from apps.predictions.prediction_engine import predict_match as engine_predict_match

        fb_client = football_data_client or FootballDataClient()
        if not fb_client.api_key:
            return {"error": "FOOTBALL_DATA_API_KEY not configured"}

        try:
            matches = fb_client.get_competition_matches()
            standings_raw = fb_client.get_competition_standings()
        except Exception as e:
            logger.error("football_data_fetch_failed", error=str(e))
            return {"error": f"Failed to fetch World Cup data: {e}"}

        def normalize_group(name: str) -> str:
            n = name.upper().replace(" ", "_")
            if not n.startswith("GROUP_"):
                n = "GROUP_" + n
            return n

        standings = {}
        for group_name, table in standings_raw.items():
            normalized = normalize_group(group_name)
            standings[normalized] = [
                {
                    "position": s.position,
                    "team_id": s.team_id,
                    "team_name": s.team_name,
                    "team_short": s.team_short,
                    "team_tla": s.team_tla,
                    "team_crest": s.team_crest,
                    "played": s.played,
                    "won": s.won,
                    "draw": s.draw,
                    "lost": s.lost,
                    "points": s.points,
                    "goals_for": s.goals_for,
                    "goals_against": s.goals_against,
                    "goal_difference": s.goal_difference,
                }
                for s in table
            ]

        teams_cache: dict[int, TeamInfo] = {}
        for m in matches:
            for side, tid, tname, tabbr in [
                ("home", m.home_team_id, m.home_team_name, m.home_team_tla),
                ("away", m.away_team_id, m.away_team_name, m.away_team_tla),
            ]:
                sid = str(tid)
                if sid not in teams_cache:
                    teams_cache[sid] = TeamInfo(
                        espn_id=sid,
                        name=tname,
                        abbreviation=tabbr or tname[:3].upper(),
                        elo=self._team_elo_cache.get(sid, INITIAL_ELO),
                        attacking=self._team_att_def_cache.get(sid, {}).get("attacking", 1.0),
                        defensive=self._team_att_def_cache.get(sid, {}).get("defensive", 1.0),
                    )

        def match_to_dict(m) -> dict:
            d: dict[str, Any] = {
                "id": m.id,
                "matchday": m.matchday,
                "status": m.status,
                "utc_date": m.utc_date,
                "stage": m.stage or "",
                "group": m.group or "",
                "home_team": {
                    "id": m.home_team_id,
                    "name": m.home_team_name,
                    "short": m.home_team_short,
                    "tla": m.home_team_tla,
                    "crest": m.home_team_crest,
                },
                "away_team": {
                    "id": m.away_team_id,
                    "name": m.away_team_name,
                    "short": m.away_team_short,
                    "tla": m.away_team_tla,
                    "crest": m.away_team_crest,
                },
                "score": {"home": m.score_home, "away": m.score_away, "winner": m.winner},
            }
            if m.status in ("TIMED", "SCHEDULED"):
                home_info = teams_cache.get(str(m.home_team_id))
                away_info = teams_cache.get(str(m.away_team_id))
                if home_info and away_info:
                    pred = engine_predict_match(home_info, away_info, espn_win_probs=None)
                    d["prediction"] = {
                        "home_win": pred.home_win,
                        "draw": pred.draw,
                        "away_win": pred.away_win,
                        "expected_goals_home": pred.expected_goals_home,
                        "expected_goals_away": pred.expected_goals_away,
                        "home_strength": pred.home_strength,
                        "away_strength": pred.away_strength,
                        "confidence": pred.confidence,
                    }
            return d

        groups: dict[str, dict] = {}
        knockout: dict[str, list] = {}
        for m in matches:
            md = match_to_dict(m)
            if m.stage == "GROUP_STAGE" and m.group:
                g = m.group
                if g not in groups:
                    display = g.replace("GROUP_", "Grupo ")
                    groups[g] = {"name": display, "standings": standings.get(g, []), "matches": []}
                groups[g]["matches"].append(md)
            else:
                stage = m.stage or "UNKNOWN"
                if stage not in knockout:
                    knockout[stage] = []
                knockout[stage].append(md)

        for g in groups.values():
            g["matches"].sort(key=lambda x: (x["matchday"], x["utc_date"]))

        result: dict[str, Any] = {
            "groups": groups,
            "standings": standings,
            "last_updated": datetime.now().isoformat(),
        }
        if knockout:
            result["knockout"] = knockout

        return result
