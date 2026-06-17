"""Baseball (MLB) prediction engine — Pythagorean Expectation.
"""
from __future__ import annotations

import math
from typing import Any

from apps.predictions.engines.base_engine import (
    GenericPrediction,
    SportPredictionEngine,
    TeamContext,
)


class BaseballEngine(SportPredictionEngine):
    """Baseball prediction using Pythagorean Expectation and Elo."""

    sport = "baseball"
    has_draws = False
    typical_score = 4.5

    def expected_scores(self, home: TeamContext, away: TeamContext) -> tuple[float, float]:
        # Simple Runs Scored / Runs Allowed estimation
        # If we had starting pitcher ERA/FIP, we'd adjust here
        h_rs = home.recent_scores_for if home.recent_scores_for > 0 else self.typical_score
        h_ra = home.recent_scores_against if home.recent_scores_against > 0 else self.typical_score
        a_rs = away.recent_scores_for if away.recent_scores_for > 0 else self.typical_score
        a_ra = away.recent_scores_against if away.recent_scores_against > 0 else self.typical_score

        # Average them
        exp_h = (h_rs + a_ra) / 2.0
        exp_a = (a_rs + h_ra) / 2.0

        # Home field advantage in baseball is about 53-54% win probability, ~0.2 runs
        exp_h += 0.2

        return max(exp_h, 1.0), max(exp_a, 1.0)

    def _pythagorean_win_prob(self, rs: float, ra: float) -> float:
        """Pythagorean expectation formula."""
        if rs == 0 and ra == 0:
            return 0.5
        # Exponent 1.83 is typical for MLB
        return (rs ** 1.83) / (rs ** 1.83 + ra ** 1.83)

    def _elo_probs(self, elo_h: float, elo_a: float, home_adv: float = 24.0) -> tuple[float, float, float]:
        # MLB home advantage is roughly 24 Elo points
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
        
        stat_prob_h = self._pythagorean_win_prob(exp_h, exp_a)
        stat_probs = (stat_prob_h, 0.0, 1 - stat_prob_h)
        
        elo_probs = self._elo_probs(home.elo, away.elo)

        models = [
            ("pythagorean", stat_probs, 0.60),
            ("elo", elo_probs, 0.40),
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
            models_used=["pythagorean", "elo"] + (["market"] if market_probs else []),
            extra={
                "spread": round(exp_h - exp_a, 1),
                "total": round(exp_h + exp_a, 1),
            },
        )
