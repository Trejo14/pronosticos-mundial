"""DRF Views for NHL models."""

from rest_framework import viewsets

from apps.nhl.models import (
    Game,
    GoalieSeasonStats,
    Player,
    SkaterSeasonStats,
    Standing,
    Team,
)
from apps.nhl.serializers import (
    GameSerializer,
    GoalieSeasonStatsSerializer,
    PlayerSerializer,
    SkaterSeasonStatsSerializer,
    StandingSerializer,
    TeamSerializer,
)


class TeamViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for NHL Teams."""
    queryset = Team.objects.filter(is_active=True)
    serializer_class = TeamSerializer
    search_fields = ["name", "full_name", "abbreviation"]
    ordering_fields = ["name", "abbreviation"]


class PlayerViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for NHL Players."""
    queryset = Player.objects.select_related("current_team").filter(is_active=True)
    serializer_class = PlayerSerializer
    search_fields = ["first_name", "last_name", "full_name"]
    ordering_fields = ["last_name", "first_name", "sweater_number"]


class GameViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for NHL Games."""
    queryset = Game.objects.select_related("home_team", "away_team")
    serializer_class = GameSerializer
    filterset_fields = ["season", "game_type", "status"]
    ordering_fields = ["date"]


class StandingViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for NHL Standings."""
    queryset = Standing.objects.select_related("team")
    serializer_class = StandingSerializer
    filterset_fields = ["date"]
    ordering_fields = ["points", "date", "point_pct"]


class SkaterSeasonStatsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Skater Season Stats."""
    queryset = SkaterSeasonStats.objects.select_related("player")
    serializer_class = SkaterSeasonStatsSerializer
    filterset_fields = ["season"]
    ordering_fields = ["points", "goals", "assists", "plus_minus"]


class GoalieSeasonStatsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Goalie Season Stats."""
    queryset = GoalieSeasonStats.objects.select_related("player")
    serializer_class = GoalieSeasonStatsSerializer
    filterset_fields = ["season"]
    ordering_fields = ["wins", "save_pct", "gaa", "shutouts"]
