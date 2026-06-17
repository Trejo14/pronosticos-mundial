"""Abstract base class for all sport prediction engines.

Every sport-specific engine inherits from this and implements its own
statistical model while exposing a common interface.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenericPrediction:
    """Common prediction result across all sports."""

    home_team: str
    away_team: str
    home_win: float
    draw: float  # 0.0 for sports without draws
    away_win: float
    expected_score_home: float
    expected_score_away: float
    home_strength: float
    away_strength: float
    confidence: float
    model_agreement: float = 0.0
    sport: str = "unknown"
    models_used: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted_outcome(self) -> str:
        if self.home_win > self.draw and self.home_win > self.away_win:
            return "home"
        if self.away_win > self.draw and self.away_win > self.home_win:
            return "away"
        return "draw"

    @property
    def spread(self) -> float:
        """Point/goal spread (positive = home favored)."""
        return round(self.expected_score_home - self.expected_score_away, 2)

    @property
    def total(self) -> float:
        """Over/under total."""
        return round(self.expected_score_home + self.expected_score_away, 2)


@dataclass
class TeamContext:
    """Unified team context passed to engines."""

    espn_id: str
    name: str
    abbreviation: str = ""
    elo: float = 1500.0
    # Offensive / defensive ratings (sport-specific interpretation)
    offensive_rating: float = 1.0
    defensive_rating: float = 1.0
    form_pts: float = 0.5  # 0-1 normalized
    recent_scores_for: float = 0.0  # avg points/goals scored recently
    recent_scores_against: float = 0.0
    home_record_pct: float = 0.5
    away_record_pct: float = 0.5
    win_pct: float = 0.5
    # Sport-specific stats (engine reads what it needs)
    stats: dict[str, Any] = field(default_factory=dict)


class SportPredictionEngine(ABC):
    """Base class for sport-specific prediction engines."""

    sport: str = "unknown"
    has_draws: bool = True
    typical_score: float = 1.5  # avg score per team per game

    @abstractmethod
    def predict(
        self,
        home: TeamContext,
        away: TeamContext,
        market_probs: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ) -> GenericPrediction:
        """Run the full prediction pipeline."""
        ...

    @abstractmethod
    def expected_scores(
        self,
        home: TeamContext,
        away: TeamContext,
    ) -> tuple[float, float]:
        """Calculate expected scores for home and away."""
        ...

    def blend(
        self,
        model_probs: list[tuple[str, tuple[float, float, float], float]],
        market_probs: tuple[float, float, float] | None = None,
        market_weight: float = 0.30,
    ) -> tuple[float, float, float]:
        """Weighted blend of multiple model outputs.

        Args:
            model_probs: List of (model_name, (h, d, a), weight)
            market_probs: Optional market-derived probabilities
            market_weight: Weight for market probs if available
        """
        if market_probs:
            model_probs.append(("market", market_probs, market_weight))

        total_weight = sum(w for _, _, w in model_probs)
        if total_weight <= 0:
            return (0.33, 0.34, 0.33) if self.has_draws else (0.5, 0.0, 0.5)

        h = sum(p[0] * w for _, p, w in model_probs) / total_weight
        d = sum(p[1] * w for _, p, w in model_probs) / total_weight
        a = sum(p[2] * w for _, p, w in model_probs) / total_weight

        # Normalize
        total = h + d + a
        if total > 0:
            h, d, a = h / total, d / total, a / total

        return (h, d, a)

    def calculate_confidence(
        self,
        final_probs: tuple[float, float, float],
        model_outputs: list[tuple[float, float, float]],
        elo_diff: float,
        market_available: bool = False,
    ) -> float:
        """Compute confidence score based on model agreement and decisiveness."""
        max_prob = max(final_probs)

        if len(model_outputs) < 2:
            base = 0.5 + (max_prob - 0.33) * 0.8
            return round(min(max(base, 0.2), 0.95), 4)

        # Model agreement (lower std = higher agreement)
        h_vals = [p[0] for p in model_outputs]
        d_vals = [p[1] for p in model_outputs]
        a_vals = [p[2] for p in model_outputs]

        def _std(vals: list[float]) -> float:
            mean = sum(vals) / len(vals)
            return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))

        agreement = 1.0 - min((_std(h_vals) + _std(d_vals) + _std(a_vals)) / 2, 1.0)
        decisiveness = (max_prob - 0.33) / 0.67
        elo_certainty = min(abs(elo_diff) / 400.0, 1.0)

        confidence = (
            0.15 * agreement
            + 0.45 * decisiveness
            + 0.15 * elo_certainty
            + 0.25 * (0.7 if market_available else 0.4)
        )
        return round(min(max(confidence, 0.1), 0.98), 4)

    def model_agreement_score(
        self,
        model_outputs: list[tuple[float, float, float]],
    ) -> float:
        """How much the models agree (0 = total disagreement, 1 = perfect agreement)."""
        if len(model_outputs) < 2:
            return 1.0
        h_vals = [p[0] for p in model_outputs]
        return round(1.0 - (max(h_vals) - min(h_vals)), 4)
