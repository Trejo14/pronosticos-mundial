"""Django admin configuration for NHL models."""

from django.contrib import admin

from apps.nhl.models import (
    Game,
    GoalieSeasonStats,
    Player,
    SkaterSeasonStats,
    Standing,
    Team,
)


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("full_name", "abbreviation", "team_id", "is_active")
    search_fields = ("full_name", "abbreviation")


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("full_name", "sweater_number", "position", "current_team", "is_active")
    search_fields = ("full_name", "last_name", "player_id")
    list_filter = ("position", "is_active", "current_team")


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = ("date", "away_team", "home_team", "status", "season")
    list_filter = ("status", "season", "game_type")
    search_fields = ("game_id",)


@admin.register(Standing)
class StandingAdmin(admin.ModelAdmin):
    list_display = ("team", "date", "points", "games_played")
    list_filter = ("date", "team")


@admin.register(SkaterSeasonStats)
class SkaterSeasonStatsAdmin(admin.ModelAdmin):
    list_display = ("player", "season", "points", "goals", "assists")
    list_filter = ("season",)
    search_fields = ("player__full_name",)


@admin.register(GoalieSeasonStats)
class GoalieSeasonStatsAdmin(admin.ModelAdmin):
    list_display = ("player", "season", "wins", "save_pct", "gaa")
    list_filter = ("season",)
    search_fields = ("player__full_name",)
