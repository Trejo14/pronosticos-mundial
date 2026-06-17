"""DRF Serializers for NHL models."""

from rest_framework import serializers

from apps.nhl.models import (
    Game,
    GoalieSeasonStats,
    Player,
    SkaterSeasonStats,
    Standing,
    Team,
)


class TeamSerializer(serializers.ModelSerializer):
    class Meta:
        model = Team
        fields = ["id", "team_id", "abbreviation", "name", "full_name", "franchise_id", "is_active", "raw_data"]


class PlayerSerializer(serializers.ModelSerializer):
    current_team = TeamSerializer(read_only=True)

    class Meta:
        model = Player
        fields = [
            "id", "player_id", "first_name", "last_name", "full_name",
            "sweater_number", "position", "current_team", "is_active", "headshot_url"
        ]


class GameSerializer(serializers.ModelSerializer):
    home_team = TeamSerializer(read_only=True)
    away_team = TeamSerializer(read_only=True)

    class Meta:
        model = Game
        fields = [
            "id", "game_id", "season", "game_type", "date",
            "home_team", "away_team", "home_score", "away_score", "status"
        ]


class StandingSerializer(serializers.ModelSerializer):
    team = TeamSerializer(read_only=True)

    class Meta:
        model = Standing
        fields = [
            "id", "team", "date", "games_played", "wins", "losses", "ot_losses",
            "points", "point_pct", "division_name", "conference_name"
        ]


class SkaterSeasonStatsSerializer(serializers.ModelSerializer):
    player = PlayerSerializer(read_only=True)

    class Meta:
        model = SkaterSeasonStats
        fields = [
            "id", "player", "season", "games_played", "goals", "assists",
            "points", "plus_minus", "raw_data"
        ]


class GoalieSeasonStatsSerializer(serializers.ModelSerializer):
    player = PlayerSerializer(read_only=True)

    class Meta:
        model = GoalieSeasonStats
        fields = [
            "id", "player", "season", "games_played", "wins", "losses",
            "save_pct", "gaa", "shutouts", "raw_data"
        ]
