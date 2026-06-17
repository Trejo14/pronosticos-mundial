"""URL configuration for predictions app."""
from django.urls import path

from apps.predictions.views import (
    CalibrationView,
    MatchDetailView,
    MatchPredictionView,
    TournamentPredictionView,
    UpcomingMatchesView,
    ValueBetsView,
    WorldCupDashboardView,
    frontend,
)

app_name = "predictions"

urlpatterns = [
    path("match/", MatchPredictionView.as_view(), name="match-prediction"),
    path("match-detail/<int:match_id>/", MatchDetailView.as_view(), name="match-detail"),
    path("tournament/", TournamentPredictionView.as_view(), name="tournament-prediction"),
    path("upcoming/", UpcomingMatchesView.as_view(), name="upcoming-matches"),
    path("value-bets/", ValueBetsView.as_view(), name="value-bets"),
    path("worldcup/", WorldCupDashboardView.as_view(), name="worldcup-dashboard"),
    path("calibration/", CalibrationView.as_view(), name="calibration"),
]
