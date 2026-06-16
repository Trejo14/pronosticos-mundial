"""API views for World Cup predictions."""
from __future__ import annotations

from typing import Any

import structlog
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status, views
from rest_framework.request import Request
from rest_framework.response import Response

from apps.predictions.serializers import (
    MatchPredictionRequestSerializer,
    MatchPredictionResponseSerializer,
    TournamentPredictionRequestSerializer,
    TournamentPredictionResponseSerializer,
    UpcomingMatchesRequestSerializer,
    ValueBetsResponseSerializer,
    ValueBetSerializer,
)
from apps.predictions.services import PredictionService

logger = structlog.get_logger(__name__)


class MatchPredictionView(views.APIView):
    """Predict the outcome of a single match."""

    @extend_schema(
        parameters=[
            OpenApiParameter(name="league", description="League slug (e.g., fifa.world)", required=False, type=str),
            OpenApiParameter(name="event_id", description="ESPN event ID", required=False, type=str),
            OpenApiParameter(name="sport", description="Sport slug (e.g., soccer)", required=False, type=str),
        ],
        responses={200: MatchPredictionResponseSerializer},
    )
    def get(self, request: Request) -> Response:
        params = MatchPredictionRequestSerializer(data=request.query_params)
        if not params.is_valid():
            return Response(params.errors, status=status.HTTP_400_BAD_REQUEST)

        league = params.validated_data.get("league", "fifa.world")
        event_id = params.validated_data.get("event_id")
        sport = params.validated_data.get("sport", "soccer")

        from clients.espn_client import ESPNClient
        from clients.stats_api_client import StatsAPIClient

        espn_client = ESPNClient()
        stats_client = StatsAPIClient()
        service = PredictionService(espn_client, stats_client)

        if event_id:
            try:
                event_data = client.get_event(sport, league, event_id)
                if not event_data or not event_data.data:
                    return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
                events_list = event_data.data.get("events", [])
                if events_list:
                    event_data.data = events_list[0]
            except Exception as e:
                logger.error("Failed to fetch event", event_id=event_id, error=str(e))
                return Response({"error": f"Failed to fetch event: {e}"}, status=status.HTTP_502_BAD_GATEWAY)
        else:
            try:
                scoreboard = client.get_scoreboard(sport, league)
                events = scoreboard.data.get("events", [])
                if not events:
                    return Response({"error": "No upcoming events found"}, status=status.HTTP_404_NOT_FOUND)
                event_data = events[0]
                event_id = event_data.get("id", "")
            except Exception as e:
                logger.error("Failed to fetch scoreboard", error=str(e))
                return Response({"error": f"Failed to fetch scoreboard: {e}"}, status=status.HTTP_502_BAD_GATEWAY)

        result = service.predict_match(event_data)
        if "error" in result:
            return Response(result, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        result["event_id"] = event_id
        serializer = MatchPredictionResponseSerializer(data=result)
        if serializer.is_valid():
            return Response(serializer.validated_data)
        logger.warning("Prediction response validation failed", errors=serializer.errors)
        return Response(result)


class TournamentPredictionView(views.APIView):
    """Predict tournament outcomes (win, final, semis, quarters)."""

    @extend_schema(
        parameters=[
            OpenApiParameter(name="league", description="League slug (e.g., fifa.world)", required=False, type=str),
            OpenApiParameter(name="simulations", description="Number of Monte Carlo simulations", required=False, type=int),
        ],
        responses={200: TournamentPredictionResponseSerializer},
    )
    def get(self, request: Request) -> Response:
        params = TournamentPredictionRequestSerializer(data=request.query_params)
        if not params.is_valid():
            return Response(params.errors, status=status.HTTP_400_BAD_REQUEST)

        league = params.validated_data.get("league", "fifa.world")
        num_simulations = params.validated_data.get("simulations", 10000)

        from clients.espn_client import ESPNClient
        from clients.stats_api_client import StatsAPIClient

        espn_client = ESPNClient()
        stats_client = StatsAPIClient()
        service = PredictionService(espn_client, stats_client)

        try:
            teams_data = espn_client.get_teams("soccer", league)
            teams_list = teams_data.data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            if not teams_list:
                teams_list = teams_data.data.get("teams", []) or teams_data.data.get("items", [])

            enriched_teams = []
            for td in teams_list:
                team = td.get("team", td)
                enriched_teams.append({
                    "id": team.get("id", ""),
                    "name": team.get("displayName", team.get("name", "")),
                    "abbreviation": team.get("abbreviation", ""),
                    "displayName": team.get("displayName", team.get("name", "")),
                    "attacking": 1.0,
                    "defensive": 1.0,
                })
        except Exception as e:
            logger.error("Failed to fetch teams", error=str(e))
            return Response({"error": f"Failed to fetch teams: {e}"}, status=status.HTTP_502_BAD_GATEWAY)

        result = service.predict_tournament(enriched_teams, num_simulations=num_simulations)
        serializer = TournamentPredictionResponseSerializer(data=result)
        if serializer.is_valid():
            return Response(serializer.validated_data)
        return Response(result)


class UpcomingMatchesView(views.APIView):
    """Predict upcoming matches with full edge/value analysis."""

    @extend_schema(
        parameters=[
            OpenApiParameter(name="league", description="League slug", required=False, type=str),
            OpenApiParameter(name="days_ahead", description="Days ahead to look", required=False, type=int),
        ],
        responses={200: MatchPredictionResponseSerializer(many=True)},
    )
    def get(self, request: Request) -> Response:
        params = UpcomingMatchesRequestSerializer(data=request.query_params)
        if not params.is_valid():
            return Response(params.errors, status=status.HTTP_400_BAD_REQUEST)

        league = params.validated_data.get("league", "fifa.world")
        days_ahead = params.validated_data.get("days_ahead", 7)

        from clients.espn_client import ESPNClient
        from clients.stats_api_client import StatsAPIClient

        espn_client = ESPNClient()
        stats_client = StatsAPIClient()
        service = PredictionService(espn_client, stats_client)

        try:
            results = service.predict_upcoming_matches(league=league, days_ahead=days_ahead)
        except Exception as e:
            logger.error("Failed to predict upcoming matches", error=str(e))
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        if results and "error" in results[0]:
            return Response(results[0], status=status.HTTP_502_BAD_GATEWAY)

        return Response(results)


class ValueBetsView(views.APIView):
    """Find matches with the best betting value (highest edge)."""

    @extend_schema(
        parameters=[
            OpenApiParameter(name="league", description="League slug", required=False, type=str),
            OpenApiParameter(name="days_ahead", description="Days ahead", required=False, type=int),
            OpenApiParameter(name="min_edge", description="Minimum edge threshold (e.g., 0.05 = 5%)", required=False, type=float),
            OpenApiParameter(name="limit", description="Max results", required=False, type=int),
        ],
        responses={200: ValueBetsResponseSerializer},
    )
    def get(self, request: Request) -> Response:
        league = request.query_params.get("league", "fifa.world")
        days_ahead = int(request.query_params.get("days_ahead", 7))
        min_edge = float(request.query_params.get("min_edge", 0.05))
        limit = int(request.query_params.get("limit", 20))

        from clients.espn_client import ESPNClient
        from clients.stats_api_client import StatsAPIClient

        espn_client = ESPNClient()
        stats_client = StatsAPIClient()
        service = PredictionService(espn_client, stats_client)

        try:
            matches = service.predict_upcoming_matches(league=league, days_ahead=days_ahead)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        value_bets = []
        for match in matches:
            if "error" in match:
                continue
            for outcome_name, outcome_data in match.get("predictions", {}).items():
                edge = outcome_data.get("edge", 0) or 0
                if edge >= min_edge:
                    value_bets.append({
                        "match": match.get("match", ""),
                        "match_date": match.get("match_date", ""),
                        "event_id": match.get("event_id", ""),
                        "outcome": outcome_name.replace("_", " ").title(),
                        "our_probability": outcome_data.get("probability", 0),
                        "best_odds": outcome_data.get("best_odds", 0),
                        "implied_probability": outcome_data.get("implied_prob", 0),
                        "edge": edge,
                        "expected_value": outcome_data.get("expected_value", 0),
                        "kelly_fraction": outcome_data.get("kelly", 0),
                        "risk_label": outcome_data.get("risk", "high"),
                    })

        value_bets.sort(key=lambda x: x["edge"], reverse=True)
        value_bets = value_bets[:limit]

        return Response({
            "count": len(value_bets),
            "results": value_bets,
        })
