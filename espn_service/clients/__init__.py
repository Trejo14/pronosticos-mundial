"""Clients package for external API integrations."""

from clients.espn_client import ESPNClient
from clients.football_data_client import FootballDataClient

__all__ = ["ESPNClient", "FootballDataClient"]
