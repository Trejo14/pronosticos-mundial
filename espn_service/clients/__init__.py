"""Clients package for external API integrations."""

from clients.espn_client import ESPNClient
from clients.football_data_client import FootballDataClient
from clients.scores365_client import Scores365Client

__all__ = ["ESPNClient", "FootballDataClient", "Scores365Client"]
