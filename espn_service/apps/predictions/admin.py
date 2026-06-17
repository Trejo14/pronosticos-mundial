"""Admin configuration for predictions models."""
from django.contrib import admin

from apps.predictions.models import MatchPrediction, TeamRating, TournamentPrediction


@admin.register(TeamRating)
class TeamRatingAdmin(admin.ModelAdmin):
    list_display = ["team_name", "league", "elo_rating", "fifa_ranking", "attacking_strength", "defensive_strength", "season"]
    list_filter = ["league", "season"]
    search_fields = ["team_name", "team_abbreviation"]
    ordering = ["-elo_rating"]


@admin.register(MatchPrediction)
class MatchPredictionAdmin(admin.ModelAdmin):
    list_display = ["home_team", "away_team", "match_date", "home_win_prob", "draw_prob", "away_win_prob", "risk_label", "recommendation"]
    list_filter = ["league", "risk_label", "match_date"]
    search_fields = ["home_team", "away_team"]
    ordering = ["-match_date"]


@admin.register(TournamentPrediction)
class TournamentPredictionAdmin(admin.ModelAdmin):
    list_display = ["team_name", "league", "season", "win_tournament_prob", "reach_final_prob", "reach_semis_prob", "simulations_run"]
    list_filter = ["league", "season"]
    search_fields = ["team_name"]
    ordering = ["-win_tournament_prob"]
