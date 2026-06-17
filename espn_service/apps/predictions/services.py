"""Orchestration service: prediction engine with persistent Elo and form tracking.

Class-world prediction pipeline:
- Persistent Elo ratings with temporal decay
- Recent form tracking (last 5 matches weighted)
- Attack/defense strength from recent results
- Multi-model blending (Poisson, Dixon-Coles, Elo, form, market odds)
- 365Scores live stats enrichment
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from django.conf import settings

from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    build_team_strength,
    extract_team_stats_from_event,
    extract_team_stats_from_standings,
    fifa_ranking_to_elo,
    get_elo,
    save_elo,
    update_elo,
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
from clients.football_data_client import (
    FootballDataMatch,
    FootballDataGoal,
    FootballDataBooking,
    FootballDataSubstitution,
)

logger = structlog.get_logger(__name__)


def _events_to_dict(goals, bookings, substitutions) -> dict:
    out: dict[str, list] = {"goals": [], "bookings": [], "substitutions": []}
    if goals:
        for g in goals:
            out["goals"].append({
                "minute": g.minute, "injury_time": g.injury_time, "type": g.type,
                "team_id": g.team_id, "team_name": g.team_name,
                "scorer_id": g.scorer_id, "scorer_name": g.scorer_name,
                "assist_id": g.assist_id, "assist_name": g.assist_name,
                "score_home": g.score_home, "score_away": g.score_away,
            })
    if bookings:
        for b in bookings:
            out["bookings"].append({
                "minute": b.minute, "card": b.card,
                "team_id": b.team_id, "team_name": b.team_name,
                "player_id": b.player_id, "player_name": b.player_name,
            })
    if substitutions:
        for s in substitutions:
            out["substitutions"].append({
                "minute": s.minute,
                "team_id": s.team_id, "team_name": s.team_name,
                "player_out_id": s.player_out_id, "player_out_name": s.player_out_name,
                "player_in_id": s.player_in_id, "player_in_name": s.player_in_name,
            })
    return out


class PredictionService:
    """Main prediction service with persistent Elo and form tracking."""

    def __init__(self, espn_client=None, stats_api_client=None):
        self.client = espn_client
        self.stats_api = stats_api_client
        self._team_elo_cache: dict[str, float] = {}
        self._team_att_def_cache: dict[str, dict[str, float]] = {}
        self._team_recent_results: dict[str, list[dict]] = {}
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

        home_elo = self._team_elo_cache.get(home_id) or get_elo(home_id)
        away_elo = self._team_elo_cache.get(away_id) or get_elo(away_id)
        self._team_elo_cache[home_id] = home_elo
        self._team_elo_cache[away_id] = away_elo

        home_att = self._team_att_def_cache.get(home_id, {}).get("attacking", 1.0)
        home_def = self._team_att_def_cache.get(home_id, {}).get("defensive", 1.0)
        away_att = self._team_att_def_cache.get(away_id, {}).get("attacking", 1.0)
        away_def = self._team_att_def_cache.get(away_id, {}).get("defensive", 1.0)

        # Form data from recent results
        home_results = self._team_recent_results.get(home_id, [])
        away_results = self._team_recent_results.get(away_id, [])
        home_form_pts = (sum(1 for r in home_results if r.get("gf", 0) > r.get("ga", 0)) * 3 +
                         sum(1 for r in home_results if r.get("gf", 0) == r.get("ga", 0))) / max(len(home_results), 1) / 3
        away_form_pts = (sum(1 for r in away_results if r.get("gf", 0) > r.get("ga", 0)) * 3 +
                         sum(1 for r in away_results if r.get("gf", 0) == r.get("ga", 0))) / max(len(away_results), 1) / 3
        home_recent_gf = sum(r.get("gf", 0) for r in home_results) / max(len(home_results), 1)
        away_recent_gf = sum(r.get("gf", 0) for r in away_results) / max(len(away_results), 1)
        home_recent_ga = sum(r.get("ga", 0) for r in home_results) / max(len(home_results), 1)
        away_recent_ga = sum(r.get("ga", 0) for r in away_results) / max(len(away_results), 1)

        from apps.predictions.feature_engineering import get_team_xg, estimate_squad_value
        hxgf, hxga = get_team_xg(home_id)
        axgf, axga = get_team_xg(away_id)
        home_info = TeamInfo(
            espn_id=home_id, name=home_name, abbreviation=home_team_data.get("abbreviation", ""),
            elo=home_elo, attacking=home_att, defensive=home_def,
            form_pts=home_form_pts, recent_gf=home_recent_gf, recent_ga=home_recent_ga,
            squad_value=estimate_squad_value(None),
            xg_per_match=hxgf,
        )
        away_info = TeamInfo(
            espn_id=away_id, name=away_name, abbreviation=away_team_data.get("abbreviation", ""),
            elo=away_elo, attacking=away_att, defensive=away_def,
            form_pts=away_form_pts, recent_gf=away_recent_gf, recent_ga=away_recent_ga,
            squad_value=estimate_squad_value(None),
            xg_per_match=axgf,
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

        prediction = predict_match(home_info, away_info, espn_win_probs=espn_probs, market_probs=market_probs)

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

        # Mejora 4: Mercados especiales
        if getattr(prediction, "extra", None) and "xg_home" in prediction.extra:
            xg_h = prediction.extra["xg_home"]
            xg_a = prediction.extra["xg_away"]
            
            # Helper for exact scores using Poisson
            import math
            def pois(k, lam):
                if lam <= 0: return 1.0 if k == 0 else 0.0
                return (math.exp(-lam) * (lam**k)) / math.factorial(k)
                
            exact_scores = []
            for i in range(5):
                for j in range(5):
                    prob = pois(i, xg_h) * pois(j, xg_a)
                    # Simple Dixon-Coles adjustment
                    if i==0 and j==0: prob *= (1 - xg_h * xg_a * 0.15)
                    elif i==1 and j==0: prob *= (1 + xg_a * 0.15)
                    elif i==0 and j==1: prob *= (1 + xg_h * 0.15)
                    elif i==1 and j==1: prob *= (1 - 0.15)
                    if prob > 0.01:
                        exact_scores.append({"score": f"{i}-{j}", "probability": round(prob, 4)})
            exact_scores.sort(key=lambda x: x["probability"], reverse=True)
            
            # Halftime (approx 45% of goals in 1st half, 55% in 2nd half)
            ht_xg_h, ht_xg_a = xg_h * 0.45, xg_a * 0.45
            ht_home = 1.0 - pois(0, ht_xg_h)
            ht_away = 1.0 - pois(0, ht_xg_a)
            ht_draw = sum(pois(k, ht_xg_h) * pois(k, ht_xg_a) for k in range(3))
            
            # Asian handicap approximation
            spread = xg_h - xg_a
            
            # Corners (rough estimate based on xG: ~3.5 corners per 1.0 xG)
            corners_h = round(max(3.5, xg_h * 3.5), 1)
            corners_a = round(max(3.5, xg_a * 3.5), 1)

            result["special_markets"] = {
                "exact_scores": exact_scores[:5],
                "halftime": {
                    "draw_prob": round(ht_draw, 4),
                    "btts_first_half": round((1 - pois(0, ht_xg_h)) * (1 - pois(0, ht_xg_a)), 4)
                },
                "asian_handicap": {
                    "line": round(spread * 2) / 2, # round to nearest 0.5
                },
                "expected_corners": {
                    "home": corners_h,
                    "away": corners_a,
                    "total": corners_h + corners_a
                }
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
        from django.utils import timezone
        now = timezone.now()

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

    def get_match_detail(self, match_id: int) -> dict[str, Any]:
        from clients.football_data_client import FootballDataClient
        fb_client = FootballDataClient()
        raw = fb_client.get_match(match_id)
        if not raw:
            return {"error": "Match not found"}
        match_data = raw.get("match", raw)
        from apps.predictions.feature_engineering import INITIAL_ELO
        m = FootballDataMatch(
            id=match_data["id"],
            utc_date=match_data["utcDate"],
            status=match_data["status"],
            matchday=match_data.get("matchday", 0),
            stage=match_data.get("stage", ""),
            group=match_data.get("group"),
            home_team_id=match_data["homeTeam"]["id"],
            home_team_name=match_data["homeTeam"]["name"],
            home_team_short=match_data["homeTeam"].get("shortName", match_data["homeTeam"]["name"]),
            home_team_tla=match_data["homeTeam"].get("tla", ""),
            home_team_crest=match_data["homeTeam"].get("crest", ""),
            away_team_id=match_data["awayTeam"]["id"],
            away_team_name=match_data["awayTeam"]["name"],
            away_team_short=match_data["awayTeam"].get("shortName", match_data["awayTeam"]["name"]),
            away_team_tla=match_data["awayTeam"].get("tla", ""),
            away_team_crest=match_data["awayTeam"].get("crest", ""),
            score_home=match_data.get("score", {}).get("fullTime", {}).get("home"),
            score_away=match_data.get("score", {}).get("fullTime", {}).get("away"),
            winner=match_data.get("score", {}).get("winner"),
            score_halftime_home=match_data.get("score", {}).get("halfTime", {}).get("home"),
            score_halftime_away=match_data.get("score", {}).get("halfTime", {}).get("away"),
            goals=[
                FootballDataGoal(
                    minute=g.get("minute", 0), injury_time=g.get("injuryTime"),
                    type=g.get("type", "GOAL"),
                    team_id=(g.get("team") or {}).get("id", 0),
                    team_name=(g.get("team") or {}).get("name", ""),
                    scorer_id=(g.get("scorer") or {}).get("id"),
                    scorer_name=(g.get("scorer") or {}).get("name"),
                    assist_id=(g.get("assist") or {}).get("id"),
                    assist_name=(g.get("assist") or {}).get("name"),
                    score_home=(g.get("score") or {}).get("home", 0),
                    score_away=(g.get("score") or {}).get("away", 0),
                ) for g in (match_data.get("goals") or [])
            ] if match_data.get("goals") else None,
            bookings=[
                FootballDataBooking(
                    minute=b.get("minute", 0),
                    team_id=(b.get("team") or {}).get("id", 0),
                    team_name=(b.get("team") or {}).get("name", ""),
                    player_id=(b.get("player") or {}).get("id"),
                    player_name=(b.get("player") or {}).get("name"),
                    card=b.get("card", "YELLOW_CARD"),
                ) for b in (match_data.get("bookings") or [])
            ] if match_data.get("bookings") else None,
            substitutions=[
                FootballDataSubstitution(
                    minute=s.get("minute", 0),
                    team_id=(s.get("team") or {}).get("id", 0),
                    team_name=(s.get("team") or {}).get("name", ""),
                    player_out_id=(s.get("playerOut") or {}).get("id"),
                    player_out_name=(s.get("playerOut") or {}).get("name"),
                    player_in_id=(s.get("playerIn") or {}).get("id"),
                    player_in_name=(s.get("playerIn") or {}).get("name"),
                ) for s in (match_data.get("substitutions") or [])
            ] if match_data.get("substitutions") else None,
        )
        sid_home = str(m.home_team_id)
        sid_away = str(m.away_team_id)

        def _info(tid, name, tla, crest):
            return {
                "id": tid, "name": name, "short": name, "tla": tla, "crest": crest,
            }

        result = {
            "id": m.id,
            "status": m.status,
            "utc_date": m.utc_date,
            "stage": m.stage,
            "group": m.group,
            "matchday": m.matchday,
            "home_team": _info(m.home_team_id, m.home_team_name, m.home_team_tla, m.home_team_crest),
            "away_team": _info(m.away_team_id, m.away_team_name, m.away_team_tla, m.away_team_crest),
            "score": {"home": m.score_home, "away": m.score_away, "winner": m.winner},
            "halftime_score": {"home": m.score_halftime_home, "away": m.score_halftime_away},
        }
        if m.goals or m.bookings or m.substitutions:
            result["events"] = _events_to_dict(m.goals, m.bookings, m.substitutions)

        # Run prediction with persistent Elo and form data
        home_elo = self._team_elo_cache.get(sid_home) or get_elo(sid_home)
        away_elo = self._team_elo_cache.get(sid_away) or get_elo(sid_away)
        self._team_elo_cache[sid_home] = home_elo
        self._team_elo_cache[sid_away] = away_elo

        home_att = self._team_att_def_cache.get(sid_home, {}).get("attacking", 1.0)
        home_def = self._team_att_def_cache.get(sid_home, {}).get("defensive", 1.0)
        away_att = self._team_att_def_cache.get(sid_away, {}).get("attacking", 1.0)
        away_def = self._team_att_def_cache.get(sid_away, {}).get("defensive", 1.0)

        home_results = self._team_recent_results.get(sid_home, [])
        away_results = self._team_recent_results.get(sid_away, [])
        home_form_pts = (sum(3 for r in home_results if r.get("gf", 0) > r.get("ga", 0)) +
                         sum(1 for r in home_results if r.get("gf", 0) == r.get("ga", 0))) / max(len(home_results), 1) / 3
        away_form_pts = (sum(3 for r in away_results if r.get("gf", 0) > r.get("ga", 0)) +
                         sum(1 for r in away_results if r.get("gf", 0) == r.get("ga", 0))) / max(len(away_results), 1) / 3
        home_rgf = sum(r.get("gf", 0) for r in home_results) / max(len(home_results), 1)
        away_rgf = sum(r.get("gf", 0) for r in away_results) / max(len(away_results), 1)
        home_rga = sum(r.get("ga", 0) for r in home_results) / max(len(home_results), 1)
        away_rga = sum(r.get("ga", 0) for r in away_results) / max(len(away_results), 1)

        from apps.predictions.feature_engineering import get_team_xg, estimate_squad_value
        hxgf, _ = get_team_xg(sid_home)
        axgf, _ = get_team_xg(sid_away)
        home_info = TeamInfo(
            espn_id=sid_home, name=m.home_team_name,
            abbreviation=m.home_team_tla or m.home_team_name[:3].upper(),
            elo=home_elo, attacking=home_att, defensive=home_def,
            form_pts=home_form_pts, recent_gf=home_rgf, recent_ga=home_rga,
            squad_value=estimate_squad_value(None),
            xg_per_match=hxgf,
        )
        away_info = TeamInfo(
            espn_id=sid_away, name=m.away_team_name,
            abbreviation=m.away_team_tla or m.away_team_name[:3].upper(),
            elo=away_elo, attacking=away_att, defensive=away_def,
            form_pts=away_form_pts, recent_gf=away_rgf, recent_ga=away_rga,
            squad_value=estimate_squad_value(None),
            xg_per_match=axgf,
        )
        pred = predict_match(home_info, away_info, espn_win_probs=None)
        result["prediction"] = {
            "home_win": pred.home_win,
            "draw": pred.draw,
            "away_win": pred.away_win,
            "expected_goals_home": pred.expected_goals_home,
            "expected_goals_away": pred.expected_goals_away,
            "home_strength": pred.home_strength,
            "away_strength": pred.away_strength,
            "confidence": pred.confidence,
            "model_agreement": pred.model_agreement,
            "xG": {"home": pred.home_xg, "away": pred.away_xg},
        }
        # Build exact scores list
        import math
        def pp(k, lam):
            if lam <= 0: return 1.0 if k == 0 else 0.0
            return (math.exp(-lam) * (lam**k)) / math.factorial(k)
            
        expH = pred.expected_goals_home
        expA = pred.expected_goals_away
        exact_scores = []
        for i in range(9):
            for j in range(9):
                p = pp(i, expH) * pp(j, expA)
                if p > 0.005:
                    exact_scores.append({"home": i, "away": j, "probability": round(p, 4)})
        exact_scores.sort(key=lambda x: x["probability"], reverse=True)
        result["exact_scores"] = exact_scores[:10]

        # Build specials / markets
        pp2 = pp
        result["markets"] = {
            "over_under": {
                "over_2_5": round(sum(pp2(i, expH) * pp2(j, expA) for i in range(9) for j in range(9) if i + j > 2.5), 4),
                "under_2_5": round(sum(pp2(i, expH) * pp2(j, expA) for i in range(9) for j in range(9) if i + j < 2.5), 4),
                "over_3_5": round(sum(pp2(i, expH) * pp2(j, expA) for i in range(9) for j in range(9) if i + j > 3.5), 4),
                "under_3_5": round(sum(pp2(i, expH) * pp2(j, expA) for i in range(9) for j in range(9) if i + j < 3.5), 4),
            },
            "both_teams_score": {
                "yes": round(sum(pp2(i, expH) * pp2(j, expA) for i in range(1, 9) for j in range(1, 9)), 4),
                "no": round(sum(pp2(i, expH) * pp2(j, expA) for i in range(9) for j in range(9) if i == 0 or j == 0) - (pp2(0, expH) * pp2(0, expA)), 4),
            },
            "double_chance": {
                "home_or_draw": round(pred.home_win + pred.draw, 4),
                "home_or_away": round(pred.home_win + pred.away_win, 4),
                "draw_or_away": round(pred.draw + pred.away_win, 4),
            },
            "exact_goals": {},
        }
        for total_g in range(7):
            p = sum(pp2(i, expH) * pp2(j, expA) for i in range(9) for j in range(9) if i + j == total_g)
            if p > 0.01:
                result["markets"]["exact_goals"][str(total_g)] = round(p, 4)

        # Enrich with 365Scores live stats
        try:
            from clients.scores365_client import Scores365Client
            s365 = Scores365Client()
            s365_game_id = s365.find_game_by_teams(m.home_team_name, m.away_team_name)
            if s365_game_id:
                stats = s365.get_game_stats(s365_game_id)
                if stats:
                    result["live_stats"] = {
                        "home": stats.home_stats(),
                        "away": stats.away_stats(),
                        "game_time": stats.game_time_display,
                        "game_minute": int(stats.game_time) if stats.game_time >= 0 else None,
                    }
                    xg_data = []
                    for ce in stats.chart_events:
                        xg_data.append({
                            "xg": round(ce.xg, 2),
                            "xgot": round(ce.xgot, 2),
                            "body_part": ce.body_part,
                            "time": ce.time,
                            "team": "home" if ce.competitor_num == 1 else "away",
                            "outcome": ce.outcome_name,
                            "is_goal": ce.outcome_id == 0,
                        })
                    if xg_data:
                        result["xg_chart"] = xg_data
        except Exception as exc:
            logger.debug("scores365_enrich_failed", error=str(exc))

        # Update Elo for finished matches
        if m.status in ("FINISHED",) and m.score_home is not None and m.score_away is not None:
            try:
                from apps.predictions.feature_engineering import update_elo
                new_home_elo, new_away_elo = update_elo(
                    home_elo, away_elo,
                    float(m.score_home), float(m.score_away),
                    margin=abs(m.score_home - m.score_away),
                )
                save_elo(sid_home, new_home_elo)
                save_elo(sid_away, new_away_elo)
                self._team_elo_cache[sid_home] = new_home_elo
                self._team_elo_cache[sid_away] = new_away_elo
                # Track results for form
                for sid, gf, ga in [(sid_home, m.score_home, m.score_away), (sid_away, m.score_away, m.score_home)]:
                    if sid not in self._team_recent_results:
                        self._team_recent_results[sid] = []
                    self._team_recent_results[sid].insert(0, {"gf": gf, "ga": ga})
                    self._team_recent_results[sid] = self._team_recent_results[sid][:10]
            except Exception as exc:
                logger.debug("elo_update_failed", error=str(exc))

            # Save xG data from 365Scores to cache
            try:
                xg_chart = result.get("xg_chart", [])
                if xg_chart:
                    home_xg_total = sum(ev["xg"] for ev in xg_chart if ev["team"] == "home")
                    away_xg_total = sum(ev["xg"] for ev in xg_chart if ev["team"] == "away")
                    if home_xg_total > 0 or away_xg_total > 0:
                        from apps.predictions.feature_engineering import save_team_xg
                        save_team_xg(sid_home, home_xg_total, away_xg_total)
                        save_team_xg(sid_away, away_xg_total, home_xg_total)
            except Exception as exc:
                logger.debug("xg_save_failed", error=str(exc))

            # Record prediction results for calibration
            try:
                from apps.predictions.model_calibration import record_prediction as cal_record
                if "prediction" in result:
                    hp = result["prediction"].get("home_win", 0.33)
                    dp = result["prediction"].get("draw", 0.33)
                    ap = result["prediction"].get("away_win", 0.33)
                    cal_record(
                        match_id=str(m.id),
                        home_team=m.home_team_name,
                        away_team=m.away_team_name,
                        home_proba=hp, draw_proba=dp, away_proba=ap,
                        actual_home=m.score_home, actual_away=m.score_away,
                    )
            except Exception as exc:
                logger.debug("calibration_record_failed", error=str(exc))

        result["specials"] = {
            "anytime_goalscorer": f"Jugador de {home_info.name if expH > expA else away_info.name} (mayor probabilidad de gol)",
            "player_cards": "Depende de las cuotas en vivo",
            "corners": "Disponible con datos de estadísticas en vivo",
        }
        return result

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
                ("home", m.home_team_id, m.home_team_name or "", m.home_team_tla or ""),
                ("away", m.away_team_id, m.away_team_name or "", m.away_team_tla or ""),
            ]:
                if not tname or not tid:
                    continue
                sid = str(tid)
                if sid not in teams_cache:
                    from apps.predictions.feature_engineering import get_team_xg, estimate_squad_value
                    xgf, _ = get_team_xg(sid)
                    teams_cache[sid] = TeamInfo(
                        espn_id=sid,
                        name=tname,
                        abbreviation=tabbr or (tname[:3].upper() if tname else "N/A"),
                        elo=self._team_elo_cache.get(sid, INITIAL_ELO),
                        attacking=self._team_att_def_cache.get(sid, {}).get("attacking", 1.0),
                        defensive=self._team_att_def_cache.get(sid, {}).get("defensive", 1.0),
                        squad_value=estimate_squad_value(None),
                        xg_per_match=xgf,
                    )

        def match_to_dict(m) -> dict:
            home_name = m.home_team_name or m.home_team_short or f"Team {m.home_team_id}"
            away_name = m.away_team_name or m.away_team_short or f"Team {m.away_team_id}"
            d: dict[str, Any] = {
                "id": m.id,
                "matchday": m.matchday,
                "status": m.status,
                "utc_date": m.utc_date,
                "stage": m.stage or "",
                "group": m.group or "",
                "home_team": {
                    "id": m.home_team_id,
                    "name": home_name,
                    "short": m.home_team_short or home_name,
                    "tla": m.home_team_tla or "",
                    "crest": m.home_team_crest or "",
                },
                "away_team": {
                    "id": m.away_team_id,
                    "name": away_name,
                    "short": m.away_team_short or away_name,
                    "tla": m.away_team_tla or "",
                    "crest": m.away_team_crest or "",
                },
                "score": {"home": m.score_home, "away": m.score_away, "winner": m.winner},
                "halftime_score": {"home": m.score_halftime_home, "away": m.score_halftime_away},
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
                        "model_agreement": pred.model_agreement,
                    }
                    # Build comprehensive markets from expected goals
                    import math
                    def _pois(k, lam):
                        if lam <= 0: return 1.0 if k == 0 else 0.0
                        return (math.exp(-lam) * (lam**k)) / math.factorial(k)

                    xg_h = pred.expected_goals_home
                    xg_a = pred.expected_goals_away

                    # Precompute Poisson grid
                    grid = {}
                    for i in range(10):
                        for j in range(10):
                            grid[(i, j)] = _pois(i, xg_h) * _pois(j, xg_a)
                            # Dixon-Coles correction for low scores
                            if i <= 1 and j <= 1:
                                if i == 0 and j == 0: grid[(i, j)] *= max(0, 1 - xg_h * xg_a * 0.15)
                                elif i == 1 and j == 0: grid[(i, j)] *= (1 + xg_a * 0.15)
                                elif i == 0 and j == 1: grid[(i, j)] *= (1 + xg_h * 0.15)
                                elif i == 1 and j == 1: grid[(i, j)] *= max(0, 1 - 0.15)

                    # Exact scores
                    exact_scores = []
                    for (i, j), prob in grid.items():
                        if prob > 0.005:
                            exact_scores.append({"score": f"{i}-{j}", "probability": round(prob, 4)})
                    exact_scores.sort(key=lambda x: x["probability"], reverse=True)

                    # Over/Under multi-line
                    over_under = {}
                    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
                        over = round(sum(p for (i, j), p in grid.items() if i + j > line), 4)
                        over_under[f"over_{str(line).replace('.','_')}"] = over
                        over_under[f"under_{str(line).replace('.','_')}"] = round(1 - over, 4)

                    # BTTS (Both Teams To Score)
                    btts_yes = round(sum(p for (i, j), p in grid.items() if i >= 1 and j >= 1), 4)
                    btts_no = round(1 - btts_yes, 4)

                    # Double chance
                    double_chance = {
                        "1X": round(pred.home_win + pred.draw, 4),
                        "12": round(pred.home_win + pred.away_win, 4),
                        "X2": round(pred.draw + pred.away_win, 4),
                    }

                    # Total goals distribution
                    exact_goals = {}
                    for total_g in range(7):
                        p = sum(pr for (i, j), pr in grid.items() if i + j == total_g)
                        if p > 0.005:
                            exact_goals[str(total_g)] = round(p, 4)

                    # Halftime predictions (45% of goals in 1st half)
                    ht_xg_h, ht_xg_a = xg_h * 0.45, xg_a * 0.45
                    ht_draw = sum(_pois(k, ht_xg_h) * _pois(k, ht_xg_a) for k in range(5))
                    ht_home = sum(
                        _pois(i, ht_xg_h) * _pois(j, ht_xg_a)
                        for i in range(6) for j in range(6) if i > j
                    )
                    ht_away = 1 - ht_draw - ht_home
                    ht_over_05 = 1 - (_pois(0, ht_xg_h) * _pois(0, ht_xg_a))

                    # Asian handicap
                    spread = xg_h - xg_a

                    # Corners estimate
                    corners_h = round(max(3.5, xg_h * 3.5), 1)
                    corners_a = round(max(3.5, xg_a * 3.5), 1)

                    # Clean sheet probabilities
                    cs_home = round(_pois(0, xg_a), 4)  # away scores 0
                    cs_away = round(_pois(0, xg_h), 4)  # home scores 0

                    d["special_markets"] = {
                        "exact_scores": exact_scores[:10],
                        "over_under": over_under,
                        "btts": {"yes": btts_yes, "no": btts_no},
                        "double_chance": double_chance,
                        "exact_goals": exact_goals,
                        "halftime": {
                            "home": round(ht_home, 4),
                            "draw": round(ht_draw, 4),
                            "away": round(max(ht_away, 0), 4),
                            "over_0_5": round(ht_over_05, 4),
                        },
                        "asian_handicap": {"line": round(spread * 2) / 2},
                        "expected_corners": {
                            "home": corners_h,
                            "away": corners_a,
                            "total": corners_h + corners_a,
                        },
                        "clean_sheet": {"home": cs_home, "away": cs_away},
                    }

            # Live stats enrichment for IN_PLAY matches
            if m.status == "IN_PLAY":
                try:
                    from clients.scores365_client import Scores365Client
                    s365 = Scores365Client()
                    s365_game_id = s365.find_game_by_teams(
                        m.home_team_name or "", m.away_team_name or ""
                    )
                    if s365_game_id:
                        stats = s365.get_game_stats(s365_game_id)
                        if stats:
                            d["live_stats"] = {
                                "home": stats.home_stats(),
                                "away": stats.away_stats(),
                                "game_time": stats.game_time_display,
                                "game_minute": int(stats.game_time) if stats.game_time >= 0 else None,
                            }
                            xg_data = []
                            for ce in stats.chart_events:
                                xg_data.append({
                                    "xg": round(ce.xg, 2),
                                    "xgot": round(ce.xgot, 2),
                                    "time": ce.time,
                                    "team": "home" if ce.competitor_num == 1 else "away",
                                    "is_goal": ce.outcome_id == 0,
                                })
                            if xg_data:
                                d["xg_chart"] = xg_data
                except Exception as exc:
                    logger.debug("live_stats_enrich_failed", error=str(exc))

            if m.status in ("IN_PLAY", "FINISHED") and (m.goals or m.bookings or m.substitutions):
                d["events"] = _events_to_dict(m.goals, m.bookings, m.substitutions)
            return d

        groups: dict[str, dict] = {}
        knockout: dict[str, list] = {}
        try:
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
        except Exception as e:
            logger.error("predict_worldcup_processing_error", error=str(e))
            if groups or knockout:
                result = {"groups": groups, "standings": standings, "last_updated": datetime.now().isoformat(), "partial": True}
                if knockout:
                    result["knockout"] = knockout
                return result
            return {"error": f"Failed to process World Cup data: {e}"}
