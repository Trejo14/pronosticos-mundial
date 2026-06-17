"""URL configuration for nhl_service."""

from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from rest_framework.routers import DefaultRouter

from apps.nhl import views as nhl_views

router = DefaultRouter()
router.register(r"teams", nhl_views.TeamViewSet, basename="team")
router.register(r"players", nhl_views.PlayerViewSet, basename="player")
router.register(r"games", nhl_views.GameViewSet, basename="game")
router.register(r"standings", nhl_views.StandingViewSet, basename="standing")
router.register(r"skater-stats", nhl_views.SkaterSeasonStatsViewSet, basename="skater-stats")
router.register(r"goalie-stats", nhl_views.GoalieSeasonStatsViewSet, basename="goalie-stats")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include(router.urls)),
    # OpenAPI schema + UI
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
]
