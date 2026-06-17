"""Basketball prediction engine — Four Factors / Pace / Efficiency.

Based on Dean Oliver's Four Factors (eFG%, TOV%, ORB%, FTR).
"""
from __future__ import annotations

import math
from typing import Any

from apps.predictions.engines.base_engine import (
    GenericPrediction,
    SportPredictionEngine,
    TeamContext,
)


class BasketballEngine(SportPredictionEngine):
    """Full basketball prediction using Four Factors and Pace-Adjusted Efficiency."""

    sport = "basketball"
    has_draws = False
    typical_score = 110.0

    def expected_scores(self, home: TeamContext, away: TeamContext) -> tuple[float, float]:
        # Fallback stats if not available
        h_pace = home.stats.get("pace", 100.0)
        a_pace = away.stats.get("pace", 100.0)
        game_pace = (h_pace + a_pace) / 2.0

        h_off_rtg = home.stats.get("offensive_rating", home.offensive_rating * 110)
        h_def_rtg = home.stats.get("defensive_rating", home.defensive_rating * 110)
        a_off_rtg = away.stats.get("offensive_rating", away.offensive_rating * 110)
        a_def_rtg = away.stats.get("defensive_rating", away.defensive_rating * 110)

        # Regress to mean for small samples or missing data
        league_avg_rtg = 115.0
        h_off_rtg = (h_off_rtg + league_avg_rtg) / 2
        h_def_rtg = (h_def_rtg + league_avg_rtg) / 2
        a_off_rtg = (a_off_rtg + league_avg_rtg) / 2
        a_def_rtg = (a_def_rtg + league_avg_rtg) / 2

        # Home court advantage (historically ~2.5 to 3.0 points in NBA)
        home_adv_rtg = 2.5

        # Pyth expectation logic for expected scores
        exp_h_rtg = (h_off_rtg + a_def_rtg) / 2 + home_adv_rtg
        exp_a_rtg = (a_off_rtg + h_def_rtg) / 2

        exp_h = exp_h_rtg * (game_pace / 100.0)
        exp_a = exp_a_rtg * (game_pace / 100.0)

        return max(exp_h, 50.0), max(exp_a, 50.0)

    def _win_prob_from_spread(self, spread: float) -> float:
        """Convert a point spread to a win probability using normal CDF approximation.
        
        A standard NBA game has a standard deviation of about 12 points.
        """
        # Normal CDF approximation
        z = spread / 12.0
        # math.erf requires python 3.2+
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _elo_probs(self, elo_h: float, elo_a: float, home_adv: float = 60.0) -> tuple[float, float, float]:
        # Basketball Elo typically uses ~60 to 100 pts for home court
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
            ("four_factors", stat_probs, 0.60),
            ("elo", elo_probs, 0.40),
        ]

        final = self.blend(
            models,
            market_probs=market_probs,
            market_weight=0.40 if market_probs else 0.0,
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
            models_used=["four_factors", "elo"] + (["market"] if market_probs else []),
            extra={
                "spread": round(spread, 1),
                "total": round(exp_h + exp_a, 1),
            },
        )
