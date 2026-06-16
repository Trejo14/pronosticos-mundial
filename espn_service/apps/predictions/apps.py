"""Predictions app configuration."""
from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    """Predictions application configuration."""
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.predictions"
    verbose_name = "World Cup Predictions"
