"""URL configuration for predictions app."""
from django.urls import path

from apps.predictions.views import (
    MatchPredictionView,
    TournamentPredictionView,
    UpcomingMatchesView,
    ValueBetsView,
)

app_name = "predictions"

urlpatterns = [
    path("match/", MatchPredictionView.as_view(), name="match-prediction"),
    path("tournament/", TournamentPredictionView.as_view(), name="tournament-prediction"),
    path("upcoming/", UpcomingMatchesView.as_view(), name="upcoming-matches"),
    path("value-bets/", ValueBetsView.as_view(), name="value-bets"),
]
