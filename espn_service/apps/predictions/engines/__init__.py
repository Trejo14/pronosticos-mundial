"""Multi-sport prediction engine registry.

Factory that selects the correct prediction engine based on sport type.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.predictions.engines.base_engine import SportPredictionEngine

# Sport slug → engine class mapping (lazy imports)
_ENGINE_REGISTRY: dict[str, str] = {
    "soccer": "apps.predictions.engines.soccer_engine.SoccerEngine",
    "basketball": "apps.predictions.engines.basketball_engine.BasketballEngine",
    "football": "apps.predictions.engines.football_engine.FootballEngine",
    "baseball": "apps.predictions.engines.baseball_engine.BaseballEngine",
}

# Cache instantiated engines
_engine_cache: dict[str, "SportPredictionEngine"] = {}


def get_engine(sport: str) -> "SportPredictionEngine":
    """Get the prediction engine for a sport.

    Falls back to soccer engine for unsupported sports.
    """
    if sport in _engine_cache:
        return _engine_cache[sport]

    class_path = _ENGINE_REGISTRY.get(sport, _ENGINE_REGISTRY["soccer"])
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    engine_class = getattr(module, class_name)
    engine = engine_class()
    _engine_cache[sport] = engine
    return engine


def list_supported_sports() -> list[str]:
    """List all sports with dedicated prediction engines."""
    return list(_ENGINE_REGISTRY.keys())
