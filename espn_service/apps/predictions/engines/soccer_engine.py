"""Soccer prediction engine — Poisson / Dixon-Coles / Elo / Form.

Wraps the existing prediction_engine.py logic into the common engine interface.
"""
from __future__ import annotations

import math
from typing import Any

from apps.predictions.engines.base_engine import (
    GenericPrediction,
    SportPredictionEngine,
    TeamContext,
)
from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    compute_team_strength,
    expected_score,
)
from apps.predictions.odds_analyzer import remove_vig


# ──────────────── Poisson / Dixon-Coles constants ────────────────

MAX_GOALS = 12
LEAGUE_AVG_GOALS = 2.5
DC_TAU = 0.15


def _poisson_prob(goals: int, expected: float) -> float:
    if expected <= 0:
        return 1.0 if goals == 0 else 0.0
    return (math.exp(-expected) * (expected ** goals)) / math.factorial(goals)


def _dixon_coles_tau(x: int, y: int, lam: float, mu: float, tau: float = DC_TAU) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * tau
    elif x == 0 and y == 1:
        return 1 + lam * tau
    elif x == 1 and y == 0:
        return 1 + mu * tau
    elif x == 1 and y == 1:
        return 1 - tau
    return 1.0


def _dixon_coles_prob(x: int, y: int, lam: float, mu: float, tau: float = DC_TAU) -> float:
    return _dixon_coles_tau(x, y, lam, mu, tau) * _poisson_prob(x, lam) * _poisson_prob(y, mu)


def _match_probs_dc(lam: float, mu: float, max_goals: int = MAX_GOALS, tau: float = DC_TAU) -> tuple[float, float, float]:
    hw, dr, aw = 0.0, 0.0, 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = _dixon_coles_prob(i, j, lam, mu, tau)
            if i > j:
                hw += p
            elif i == j:
                dr += p
            else:
                aw += p
    total = hw + dr + aw
    if total > 0:
        hw, dr, aw = hw / total, dr / total, aw / total
    return hw, dr, aw


def _match_probs_poisson(lam: float, mu: float, max_goals: int = MAX_GOALS) -> tuple[float, float, float]:
    hw, dr, aw = 0.0, 0.0, 0.0
    for i in range(max_goals + 1):
        pi = _poisson_prob(i, lam)
        for j in range(max_goals + 1):
            pj = _poisson_prob(j, mu)
            p = pi * pj
            if i > j:
                hw += p
            elif i == j:
                dr += p
            else:
                aw += p
    total = hw + dr + aw
    if total > 0:
        hw, dr, aw = hw / total, dr / total, aw / total
    return hw, dr, aw


def _elo_probs(elo_h: float, elo_a: float, ha: float = 0.06, draw_implied: float = 0.25) -> tuple[float, float, float]:
    exp_h = expected_score(elo_h + ha * 400, elo_a)
    h = exp_h * (1 - draw_implied)
    a = (1 - exp_h) * (1 - draw_implied)
    return h, draw_implied, a


def _form_probs(h_form: float, a_form: float, draw_prob: float = 0.26) -> tuple[float, float, float]:
    total = h_form + a_form
    if total <= 0:
        return 0.37, draw_prob, 0.37
    h = (h_form / total) * (1 - draw_prob)
    a = (a_form / total) * (1 - draw_prob)
    return h, draw_prob, a


class SoccerEngine(SportPredictionEngine):
    """Full soccer prediction: Poisson + Dixon-Coles + Elo + Form + Market blend."""

    sport = "soccer"
    has_draws = True
    typical_score = 1.25

    def expected_scores(self, home: TeamContext, away: TeamContext) -> tuple[float, float]:
        h_att = home.offensive_rating
        h_def = home.defensive_rating
        a_att = away.offensive_rating
        a_def = away.defensive_rating

        # Adjust by form
        if home.recent_scores_for > 0:
            h_att = (h_att + home.recent_scores_for / (LEAGUE_AVG_GOALS / 2)) / 2
        if away.recent_scores_for > 0:
            a_att = (a_att + away.recent_scores_for / (LEAGUE_AVG_GOALS / 2)) / 2

        exp_h = LEAGUE_AVG_GOALS * h_att * a_def * 1.06  # home advantage
        exp_a = LEAGUE_AVG_GOALS * a_att * h_def

        # Squad value adjustment
        sv_h = home.stats.get("squad_value", 50.0)
        sv_a = away.stats.get("squad_value", 50.0)
        sv_ratio = (sv_h / max(sv_a, 1)) ** 0.08
        exp_h *= sv_ratio
        exp_a /= sv_ratio

        return max(exp_h, 0.05), max(exp_a, 0.05)

    def predict(
        self,
        home: TeamContext,
        away: TeamContext,
        market_probs: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ) -> GenericPrediction:
        exp_h, exp_a = self.expected_scores(home, away)

        # Model 1: Poisson
        poisson = _match_probs_poisson(exp_h, exp_a)
        # Model 2: Dixon-Coles
        dc = _match_probs_dc(exp_h, exp_a)
        # Model 3: Elo
        elo = _elo_probs(home.elo, away.elo)
        # Model 4: Form
        form = _form_probs(home.form_pts, away.form_pts)

        models = [
            ("dixon_coles", dc, 0.30),
            ("elo", elo, 0.25),
            ("form", form, 0.15),
            ("poisson", poisson, 0.10),
        ]

        final = self.blend(
            models,
            market_probs=market_probs,
            market_weight=0.35 if market_probs else 0.0,
        )

        all_outputs = [poisson, dc, elo, form]
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
            draw=round(final[1], 4),
            away_win=round(final[2], 4),
            expected_score_home=round(exp_h, 2),
            expected_score_away=round(exp_a, 2),
            home_strength=round(compute_team_strength(home.elo, home.offensive_rating, home.defensive_rating, True, home.form_pts), 3),
            away_strength=round(compute_team_strength(away.elo, away.offensive_rating, away.defensive_rating, False, away.form_pts), 3),
            confidence=conf,
            model_agreement=agreement,
            sport="soccer",
            models_used=["poisson", "dixon_coles", "elo", "form"] + (["market"] if market_probs else []),
            extra={
                "xg_home": round(exp_h, 2),
                "xg_away": round(exp_a, 2),
                "over_2_5": round(sum(
                    _poisson_prob(i, exp_h) * _poisson_prob(j, exp_a)
                    for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)
                    if i + j > 2.5
                ), 4),
                "btts_yes": round(sum(
                    _poisson_prob(i, exp_h) * _poisson_prob(j, exp_a)
                    for i in range(1, MAX_GOALS + 1) for j in range(1, MAX_GOALS + 1)
                ), 4),
            },
        )
