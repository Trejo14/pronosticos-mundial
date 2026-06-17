"""Football (NFL/NCAAF) prediction engine — DVOA / Efficiency.
"""
from __future__ import annotations

import math
from typing import Any

from apps.predictions.engines.base_engine import (
    GenericPrediction,
    SportPredictionEngine,
    TeamContext,
)


class FootballEngine(SportPredictionEngine):
    """American Football prediction using DVOA and yardage efficiency."""

    sport = "football"
    has_draws = False # NFL ties exist but are rare enough to ignore for simple ML
    typical_score = 23.0

    def expected_scores(self, home: TeamContext, away: TeamContext) -> tuple[float, float]:
        # Basic NFL efficiency baseline
        h_off = home.offensive_rating * self.typical_score
        h_def = home.defensive_rating * self.typical_score
        a_off = away.offensive_rating * self.typical_score
        a_def = away.defensive_rating * self.typical_score
        
        home_adv = 2.0 # Historically NFL home field was ~3, now closer to 1.5-2.0

        exp_h = (h_off + a_def) / 2.0 + home_adv
        exp_a = (a_off + h_def) / 2.0

        return max(exp_h, 3.0), max(exp_a, 3.0)

    def _win_prob_from_spread(self, spread: float) -> float:
        # NFL standard deviation of spread is ~13.5
        z = spread / 13.5
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _elo_probs(self, elo_h: float, elo_a: float, home_adv: float = 55.0) -> tuple[float, float, float]:
        exp_h = 1 / (1 + 10 ** ((elo_a - (elo_h + home_adv)) / 400.0))
        return exp_h, 0.0, 1 - exp_h

    def predict(
        self,
        home: TeamContext,
        away: TeamContext,
        market_probs: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ) -> GenericPrediction:
        exp_h, exp_a = self.expected_scores(home, away)
        spread = exp_h - exp_a
        
        stat_prob_h = self._win_prob_from_spread(spread)
        stat_probs = (stat_prob_h, 0.0, 1 - stat_prob_h)
        
        elo_probs = self._elo_probs(home.elo, away.elo)

        models = [
            ("efficiency", stat_probs, 0.50),
            ("elo", elo_probs, 0.50),
        ]

        final = self.blend(
            models,
            market_probs=market_probs,
            market_weight=0.45 if market_probs else 0.0,
        )

        all_outputs = [stat_probs, elo_probs]
        if market_probs:
            all_outputs.append(market_probs)

        conf = self.calculate_confidence(
            final, all_outputs, home.elo - away.elo, market_available=market_probs is not None
        )
        agreement = self.model_agreement_score(all_outputs)

        return GenericPrediction(
            home_team=home.name,
            away_team=away.name,
            home_win=round(final[0], 4),
            draw=0.0,
            away_win=round(final[2], 4),
            expected_score_home=round(exp_h, 1),
            expected_score_away=round(exp_a, 1),
            home_strength=round(home.elo / 1500.0, 3),
            away_strength=round(away.elo / 1500.0, 3),
            confidence=conf,
            model_agreement=agreement,
            sport=self.sport,
            models_used=["efficiency", "elo"] + (["market"] if market_probs else []),
            extra={
                "spread": round(spread, 1),
                "total": round(exp_h + exp_a, 1),
            },
        )
