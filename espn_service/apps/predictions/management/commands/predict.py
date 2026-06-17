"""Management command to run predictions from the command line.

Usage:
    python manage.py predict match --event-id 12345
    python manage.py predict match --league fifa.world
    python manage.py predict tournament --simulations 5000
    python manage.py predict upcoming --days 3
    python manage.py predict value-bets --min-edge 0.05
"""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from clients.espn_client import ESPNClient


class Command(BaseCommand):
    help = "Run World Cup predictions"

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest="subcommand", required=True)

        match_parser = sub.add_parser("match", help="Predict a single match")
        match_parser.add_argument("--event-id", type=str, help="ESPN event ID")
        match_parser.add_argument("--league", type=str, default="fifa.world")

        tourney_parser = sub.add_parser("tournament", help="Predict tournament outcomes")
        tourney_parser.add_argument("--simulations", type=int, default=5000)
        tourney_parser.add_argument("--league", type=str, default="fifa.world")

        upcoming_parser = sub.add_parser("upcoming", help="Predict upcoming matches")
        upcoming_parser.add_argument("--days", type=int, default=7)
        upcoming_parser.add_argument("--league", type=str, default="fifa.world")

        vb_parser = sub.add_parser("value-bets", help="Find value bets")
        vb_parser.add_argument("--min-edge", type=float, default=0.05)
        vb_parser.add_argument("--days", type=int, default=7)
        vb_parser.add_argument("--league", type=str, default="fifa.world")

    def handle(self, *args, **options):
        from apps.predictions.services import PredictionService

        client = ESPNClient()
        service = PredictionService(client)

        sub = options["subcommand"]

        if sub == "match":
            self._handle_match(service, options)
        elif sub == "tournament":
            self._handle_tournament(service, options)
        elif sub == "upcoming":
            self._handle_upcoming(service, options)
        elif sub == "value-bets":
            self._handle_value_bets(service, options)

    def _handle_match(self, service, options):
        league = options["league"]
        event_id = options.get("event_id")

        from clients.espn_client import ESPNClient

        client = ESPNClient()

        if event_id:
            resp = client.get_event("soccer", league, event_id)
            event_data = resp.data
            events = event_data.get("events", [])
            if events:
                event_data = events[0]
        else:
            resp = client.get_scoreboard("soccer", league)
            events = resp.data.get("events", [])
            if not events:
                self.stderr.write("No upcoming events")
                return
            event_data = events[0]
            event_id = event_data.get("id", "")

        result = service.predict_match(event_data)
        result["event_id"] = event_id or event_data.get("id", "")
        self.stdout.write(json.dumps(result, indent=2, default=str))

    def _handle_tournament(self, service, options):
        league = options["league"]
        simulations = options["simulations"]

        from clients.espn_client import ESPNClient

        client = ESPNClient()
        teams_resp = client.get_teams("soccer", league)
        teams_data = teams_resp.data
        teams_list = (
            teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            or teams_data.get("teams", [])
            or teams_data.get("items", [])
        )

        enriched = []
        for td in teams_list:
            team = td.get("team", td)
            enriched.append({
                "id": team.get("id", ""),
                "name": team.get("displayName", team.get("name", "")),
                "abbreviation": team.get("abbreviation", ""),
                "displayName": team.get("displayName", team.get("name", "")),
                "attacking": 1.0,
                "defensive": 1.0,
            })

        result = service.predict_tournament(enriched, num_simulations=simulations)
        self.stdout.write(json.dumps(result, indent=2, default=str))

    def _handle_upcoming(self, service, options):
        league = options["league"]
        days = options["days"]
        results = service.predict_upcoming_matches(league=league, days_ahead=days)
        self.stdout.write(json.dumps(results, indent=2, default=str))

    def _handle_value_bets(self, service, options):
        league = options["league"]
        days = options["days"]
        min_edge = options["min_edge"]

        matches = service.predict_upcoming_matches(league=league, days_ahead=days)
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
                        "edge": edge,
                        "expected_value": outcome_data.get("expected_value", 0),
                        "kelly": outcome_data.get("kelly", 0),
                        "risk": outcome_data.get("risk", "high"),
                    })

        value_bets.sort(key=lambda x: x["edge"], reverse=True)

        self.stdout.write(json.dumps({
            "count": len(value_bets),
            "min_edge": min_edge,
            "results": value_bets[:20],
        }, indent=2, default=str))
