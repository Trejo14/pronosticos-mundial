"""Management command to test The Odds API connection.

Usage:
    python manage.py check_odds_api
    python manage.py check_odds_api --sport soccer_epl
    python manage.py check_odds_api --sport soccer_fifa_world_cup --regions us,uk
"""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Test The Odds API connection"

    def add_arguments(self, parser):
        parser.add_argument("--sport", type=str, default="soccer_fifa_world_cup")
        parser.add_argument("--regions", type=str, default="us,uk,eu")
        parser.add_argument("--markets", type=str, default="h2h")

    def handle(self, *args, **options):
        from clients.odds_api_client import OddsApiClient

        client = OddsApiClient()

        if options["sport"] == "list":
            self.stdout.write("Fetching supported sports...")
            try:
                resp = client.get_sports()
                for s in resp.data:
                    self.stdout.write(f"  {s['key']:40s} {s['title']}")
            except Exception as e:
                raise CommandError(f"Failed: {e}")
            return

        sport = options["sport"]
        regions = options["regions"]
        markets = options["markets"]

        self.stdout.write(f"Fetching odds for sport={sport} regions={regions} markets={markets}")
        try:
            resp = client.get_sport_odds(
                sport_key=sport,
                regions=regions,
                markets=markets,
            )
            remaining = resp.remaining_requests
            data = resp.data
            if isinstance(data, list):
                self.stdout.write(f"Found {len(data)} events, remaining requests: {remaining}")
                for event in data[:5]:
                    self.stdout.write(f"\n  {event.get('id', '?')}")
                    self.stdout.write(f"  {event.get('home_team', '?')} vs {event.get('away_team', '?')}")
                    self.stdout.write(f"  Commence: {event.get('commence_time', '?')}")
                    for bm in event.get("bookmakers", []):
                        self.stdout.write(f"    {bm['title']}:")
                        for market in bm.get("markets", []):
                            for outcome in market.get("outcomes", []):
                                self.stdout.write(f"      {outcome['name']}: {outcome['price']}")
                if len(data) > 5:
                    self.stdout.write(f"  ... and {len(data) - 5} more")
            else:
                self.stdout.write(json.dumps(data, indent=2, default=str))
        except Exception as e:
            raise CommandError(f"Failed: {e}")
