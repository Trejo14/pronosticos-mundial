"""Test TheStatsAPI connection and World Cup data availability."""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Test TheStatsAPI connection"

    def add_arguments(self, parser):
        parser.add_argument("--endpoint", type=str, default="competitions", choices=[
            "competitions", "matches", "world-cup", "odds"])
        parser.add_argument("--id", type=str, default="")

    def handle(self, *args, **options):
        from clients.stats_api_client import StatsAPIClient

        client = StatsAPIClient()
        endpoint = options["endpoint"]
        eid = options["id"]

        try:
            if endpoint == "competitions":
                resp = client.get_competitions(per_page=50)
                data = resp if isinstance(resp, list) else resp.get("data", [])

                # Find World Cup
                wc = [c for c in data if "world" in c.get("name", "").lower() or "world" in c.get("title", "").lower()]
                self.stdout.write(f"Total competitions: {len(data)}")
                self.stdout.write(f"World Cup related: {len(wc)}")
                for c in wc:
                    self.stdout.write(json.dumps(c, indent=2))

                self.stdout.write("\n--- First 10 competitions ---")
                for c in data[:10]:
                    self.stdout.write(f"  {c.get('id','?'):20s} {c.get('name', c.get('title','?'))}")

            elif endpoint == "world-cup":
                # Search for World Cup competition
                resp = client.get_competitions(per_page=200)
                data = resp if isinstance(resp, list) else resp.get("data", [])
                wc = [c for c in data if "world" in c.get("name", "").lower()]
                if not wc:
                    self.stdout.write("World Cup not found among competitions")
                    self.stdout.write(json.dumps(data[:5], indent=2))
                    return
                comp = wc[0]
                cid = comp.get("id", comp.get("competition_id", ""))
                self.stdout.write(f"World Cup competition: {comp.get('name')} (ID: {cid})")

                # Get seasons
                seasons = client.get_competition_seasons(cid)
                self.stdout.write(f"Seasons: {json.dumps(seasons, indent=2)[:1000]}")

            elif endpoint == "matches":
                params = {"per_page": 10}
                if eid:
                    params["competition_id"] = eid
                resp = client.get_matches(**params)
                self.stdout.write(json.dumps(resp, indent=2)[:2000])

            elif endpoint == "odds":
                if not eid:
                    raise CommandError("--id required (match_id)")
                odds = client.get_match_odds(eid)
                self.stdout.write(json.dumps(odds, indent=2)[:2000])

        except Exception as e:
            import traceback
            raise CommandError(f"Failed: {e}\n{traceback.format_exc()}")
