"""Motor de predicción de clase mundial.

Modelos:
- Poisson bivariado con corrección Dixon-Coles (correlación en partidos de pocos goles)
- Elo rating con margen de gol
- Forma reciente ponderada
- Calibración con cuotas de mercado cuando están disponibles
- Blending adaptativo según confianza de cada modelo
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    compute_team_strength,
    expected_score,
    extract_team_stats_from_event,
)
from apps.predictions.odds_analyzer import remove_vig

# ──────────────────────── constantes ────────────────────────

MAX_GOALS = 12
LEAGUE_AVG_GOALS = 2.5
# Dixon-Coles: correlación en partidos de <2 goles
DC_TAU = 0.15


@dataclass
class MatchPrediction:
    home_win: float
    draw: float
    away_win: float
    expected_goals_home: float
    expected_goals_away: float
    home_strength: float
    away_strength: float
    confidence: float
    model_agreement: float = 0.0  # qué tanto coinciden los modelos
    home_xg: float = 0.0
    away_xg: float = 0.0


@dataclass
class TeamInfo:
    espn_id: str
    name: str
    abbreviation: str
    elo: float = INITIAL_ELO
    attacking: float = 1.0
    defensive: float = 1.0
    form_pts: float = 0.5
    recent_gf: float = 1.0
    recent_ga: float = 1.0
    squad_value: float = 50.0
    xg_per_match: float | None = None


# ──────────────────────── distribuciones ────────────────────────

def poisson_prob(goals: int, expected: float) -> float:
    if expected <= 0:
        return 1.0 if goals == 0 else 0.0
    return (math.exp(-expected) * (expected ** goals)) / math.factorial(goals)


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, tau: float = DC_TAU) -> float:
    """Factor de corrección Dixon-Coles para partidos de pocos goles.

    Cuando lam o mu son pequeños, la correlación es significativa.
    """
    if x == 0 and y == 0:
        return 1 - lam * mu * tau
    elif x == 0 and y == 1:
        return 1 + lam * tau
    elif x == 1 and y == 0:
        return 1 + mu * tau
    elif x == 1 and y == 1:
        return 1 - tau
    else:
        return 1.0


def dixon_coles_prob(x: int, y: int, lam: float, mu: float, tau: float = DC_TAU) -> float:
    """Probabilidad conjunta ajustada por Dixon-Coles."""
    return dixon_coles_tau(x, y, lam, mu, tau) * poisson_prob(x, lam) * poisson_prob(y, mu)


def match_probabilities_dc(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = MAX_GOALS,
    tau: float = DC_TAU,
) -> tuple[float, float, float]:
    """Distribución conjunta con Dixon-Coles."""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = dixon_coles_prob(i, j, lambda_home, lambda_away, tau)
            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    return home_win, draw, away_win


def poisson_match_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = MAX_GOALS,
) -> tuple[float, float, float]:
    """Poisson independiente (sin correlación)."""
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for i in range(max_goals + 1):
        p_i = poisson_prob(i, lambda_home)
        for j in range(max_goals + 1):
            p_j = poisson_prob(j, lambda_away)
            prob = p_i * p_j
            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    return home_win, draw, away_win


# ──────────────────────── modelo Elo ────────────────────────

def elo_match_probabilities(
    elo_home: float,
    elo_away: float,
    home_advantage: float = 0.0,
    draw_implied: float = 0.25,
) -> tuple[float, float, float]:
    """Probabilidades estilo Elo: la ventaja local desplaza el rating."""
    expected_home = expected_score(elo_home + home_advantage, elo_away)
    h = expected_home * (1 - draw_implied)
    a = (1 - expected_home) * (1 - draw_implied)
    d = draw_implied
    return h, d, a


# ──────────────────────── goles esperados ────────────────────────

def calculate_expected_goals(
    home_strength: TeamStrength,
    away_strength: TeamStrength,
    league_avg_goals: float = LEAGUE_AVG_GOALS,
    home_form_gf: float | None = None,
    away_form_gf: float | None = None,
    home_form_ga: float | None = None,
    away_form_ga: float | None = None,
) -> tuple[float, float]:
    home_attack = home_strength.attacking
    home_defense = home_strength.defensive
    away_attack = away_strength.attacking
    away_defense = away_strength.defensive

    # Ajuste por forma reciente (si está disponible)
    if home_form_gf:
        home_attack = (home_attack + home_form_gf / (league_avg_goals / 2)) / 2
    if away_form_gf:
        away_attack = (away_attack + away_form_gf / (league_avg_goals / 2)) / 2

    # Expected goals con ataque local x defensa visitante (y viceversa)
    expected_home = league_avg_goals * home_attack * away_defense * (1 + home_strength.home_advantage)
    expected_away = league_avg_goals * away_attack * home_defense

    # Ajuste por valor de plantilla: equipos con plantilla más cara
    # tienden a rendir mejor de lo que sus stats recientes indican
    sv_h = home_strength.squad_value or 50.0
    sv_a = away_strength.squad_value or 50.0
    sv_ratio = (sv_h / max(sv_a, 1)) ** 0.08  # factor suave (~1.12 para 4x diferencia)
    expected_home *= sv_ratio
    expected_away /= sv_ratio

    # Ajuste mínimo
    expected_home = max(expected_home, 0.05)
    expected_away = max(expected_away, 0.05)

    return expected_home, expected_away


# ──────────────────────── blending ────────────────────────

def blend_probabilities(
    poisson_probs: tuple[float, float, float],
    dc_probs: tuple[float, float, float],
    elo_probs: tuple[float, float, float],
    market_probs: tuple[float, float, float] | None = None,
    form_probs: tuple[float, float, float] | None = None,
    elo_weight: float = 0.25,
    market_weight: float = 0.35,
    dc_weight: float = 0.25,
    form_weight: float = 0.15,
    calibrated_weights: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """Blend adaptativo de múltiples modelos.

    Cuando hay cuotas de mercado, tienen más peso (los mercados son eficientes).
    Cuando no hay, pesa más el modelo Dixon-Coles + Poisson + Elo.
    Si calibrated_weights se proporciona, se usan en lugar de los defaults.
    """
    if calibrated_weights:
        dc_weight = calibrated_weights.get("dc", dc_weight)
        elo_weight = calibrated_weights.get("elo", elo_weight)
        form_weight = calibrated_weights.get("form", form_weight)
        market_weight = calibrated_weights.get("market", market_weight)
        poisson_weight = calibrated_weights.get("poisson", dc_weight)
        # poisson y dc comparten peso si no hay separado
        dc_weight = poisson_weight if "poisson" in calibrated_weights and "dc" not in calibrated_weights else dc_weight

    weights = {}

    weights["dc"] = dc_weight
    weights["elo"] = elo_weight
    weights["form"] = form_weight if form_probs else 0.0
    weights["market"] = market_weight if market_probs else 0.0

    total_w = sum(weights.values())
    if total_w <= 0:
        return remove_vig(*poisson_probs)

    h = 0.0
    d = 0.0
    a = 0.0

    h += dc_probs[0] * weights["dc"]
    d += dc_probs[1] * weights["dc"]
    a += dc_probs[2] * weights["dc"]

    h += elo_probs[0] * weights["elo"]
    d += elo_probs[1] * weights["elo"]
    a += elo_probs[2] * weights["elo"]

    if form_probs:
        h += form_probs[0] * weights["form"]
        d += form_probs[1] * weights["form"]
        a += form_probs[2] * weights["form"]

    if market_probs:
        h += market_probs[0] * weights["market"]
        d += market_probs[1] * weights["market"]
        a += market_probs[2] * weights["market"]

    return remove_vig(h / total_w, d / total_w, a / total_w)


def _form_to_probs(
    home_form_pts: float,
    away_form_pts: float,
    draw_prob: float = 0.26,
) -> tuple[float, float, float]:
    """Convierte forma reciente en probabilidades."""
    total = home_form_pts + away_form_pts
    if total <= 0:
        return 0.37, draw_prob, 0.37
    h = (home_form_pts / total) * (1 - draw_prob)
    a = (away_form_pts / total) * (1 - draw_prob)
    return h, draw_prob, a


def calculate_confidence(
    final_probs: tuple[float, float, float],
    model_probs_list: list[tuple[float, float, float]],
    elo_diff: float,
    market_available: bool = False,
) -> float:
    """Confianza basada en:
    - Qué tan decidida está la predicción (prob máxima)
    - Acuerdo entre modelos (desviación estándar de las prob)
    - Diferencia Elo (partidos parejos = menos confianza)
    """
    max_prob = max(final_probs)

    if len(model_probs_list) < 2:
        base = 0.5 + (max_prob - 0.33) * 0.8
        return min(max(base, 0.2), 0.95)

    # Agreement: qué tanto varían los modelos
    h_vals = [p[0] for p in model_probs_list]
    d_vals = [p[1] for p in model_probs_list]
    a_vals = [p[2] for p in model_probs_list]

    def std(vals: list[float]) -> float:
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))

    agreement = 1.0 - min((std(h_vals) + std(d_vals) + std(a_vals)) / 2, 1.0)

    # Decisión: qué tan clara es la predicción
    decisiveness = (max_prob - 0.33) / 0.67  # 0 = empate perfecto, 1 = total

    # Diferencia Elo
    elo_certainty = min(abs(elo_diff) / 400.0, 1.0)

    confidence = (
        0.15 * agreement +
        0.45 * decisiveness +
        0.15 * elo_certainty +
        0.25 * (0.7 if market_available else 0.4)
    )

    return round(min(max(confidence, 0.1), 0.98), 4)


# ──────────────────────── predict_match ────────────────────────

def _load_calibrated_weights() -> dict[str, float] | None:
    """Carga pesos calibrados desde model_calibration si hay suficientes datos."""
    try:
        from apps.predictions.model_calibration import get_calibration_summary
        cal = get_calibration_summary()
        if cal.get("count", 0) >= 10:
            return dict(cal["blend_weights"])
    except Exception:
        pass
    return None


def predict_match(
    home: TeamInfo,
    away: TeamInfo,
    espn_win_probs: tuple[float, float, float] | None = None,
    league_avg_goals: float = LEAGUE_AVG_GOALS,
    elo_weight: float = 0.20,
    market_probs: tuple[float, float, float] | None = None,
) -> MatchPrediction:
    """Predicción completa multi-modelo."""
    home_strength = TeamStrength(
        name=home.name, espn_id=home.espn_id,
        elo=home.elo, attacking=home.attacking, defensive=home.defensive,
        home_advantage=0.06,
        form_pts=home.form_pts,
        recent_gf=home.recent_gf, recent_ga=home.recent_ga,
        squad_value=home.squad_value,
        xg_per_match=home.xg_per_match,
    )
    away_strength = TeamStrength(
        name=away.name, espn_id=away.espn_id,
        elo=away.elo, attacking=away.attacking, defensive=away.defensive,
        home_advantage=0.0,
        form_pts=away.form_pts,
        recent_gf=away.recent_gf, recent_ga=away.recent_ga,
        squad_value=away.squad_value,
        xg_per_match=away.xg_per_match,
    )

    # Expected goals desde forma reciente y ataque/defensa
    exp_home, exp_away = calculate_expected_goals(
        home_strength, away_strength, league_avg_goals,
        home_form_gf=home.recent_gf, away_form_gf=away.recent_gf,
        home_form_ga=home.recent_ga, away_form_ga=away.recent_ga,
    )

    # 1. Poisson independiente
    poisson_probs = poisson_match_probabilities(exp_home, exp_away)

    # 2. Dixon-Coles (corregido por correlación)
    dc_probs = match_probabilities_dc(exp_home, exp_away)

    # 3. Elo
    elo_probs = elo_match_probabilities(home.elo, away.elo, home_advantage=0.06)

    # 4. Forma reciente
    form_probs = _form_to_probs(home.form_pts, away.form_pts)

    # 5. ESPN (si está disponible) - tratar como un modelo más
    espn_probs = None
    if espn_win_probs:
        espn_probs = remove_vig(*espn_win_probs)

    # Pesos calibrados del blending
    cal_weights = _load_calibrated_weights()

    # Blending
    market_available = market_probs is not None
    final_home, final_draw, final_away = blend_probabilities(
        poisson_probs, dc_probs, elo_probs,
        market_probs=market_probs,
        form_probs=form_probs,
        market_weight=0.35 if market_available else 0.0,
        calibrated_weights=cal_weights,
    )

    model_probs_list = [poisson_probs, dc_probs, elo_probs, form_probs]
    if market_probs:
        model_probs_list.append(market_probs)
    if espn_probs:
        model_probs_list.append(espn_probs)

    confidence = calculate_confidence(
        (final_home, final_draw, final_away),
        model_probs_list,
        home.elo - away.elo,
        market_available=market_available,
    )

    # Agreement entre modelos
    h_vals = [p[0] for p in model_probs_list]
    agreement = 1.0 - (max(h_vals) - min(h_vals))

    # xG esperados del modelo DC
    home_xg = round(exp_home, 2)
    away_xg = round(exp_away, 2)

    return MatchPrediction(
        home_win=round(final_home, 4),
        draw=round(final_draw, 4),
        away_win=round(final_away, 4),
        expected_goals_home=round(exp_home, 2),
        expected_goals_away=round(exp_away, 2),
        home_strength=round(compute_team_strength(home.elo, home.attacking, home.defensive, True, home.form_pts), 3),
        away_strength=round(compute_team_strength(away.elo, away.attacking, away.defensive, False, away.form_pts), 3),
        confidence=round(confidence, 4),
        model_agreement=round(agreement, 4),
        home_xg=home_xg,
        away_xg=away_xg,
    )


# ──────────────────────── extractores ESPN ────────────────────────

def extract_espn_win_probs(event_data: dict[str, Any]) -> tuple[float, float, float] | None:
    competitions = event_data.get("competitions", [])
    if not competitions and "events" in event_data:
        events = event_data.get("events", [])
        if events:
            competitions = events[0].get("competitions", [])
    if not competitions:
        return None
    for comp in competitions:
        probs = comp.get("probabilities") or comp.get("predictor") or {}
        home_team_data = probs.get("homeTeam") or {}
        away_team_data = probs.get("awayTeam") or {}
        home_win = home_team_data.get("gameProjection") or home_team_data.get("homeWinPercentage")
        away_win = away_team_data.get("teamChanceLoss") or away_team_data.get("awayWinPercentage")
        if home_win is not None:
            home_val = float(home_win) / 100 if float(home_win) > 1 else float(home_win)
            if away_win is not None:
                away_val = float(away_win) / 100 if float(away_win) > 1 else float(away_win)
            else:
                away_val = 1.0 - home_val
            draw_val = 1.0 - home_val - away_val
            return (home_val, draw_val, away_val)
    return None


def extract_odds(event_data: dict[str, Any]) -> list[dict[str, Any]] | None:
    competitions = event_data.get("competitions", [])
    if not competitions and "events" in event_data:
        events = event_data.get("events", [])
        if events:
            competitions = events[0].get("competitions", [])
    if not competitions:
        return None
    for comp in competitions:
        odds = comp.get("odds") or comp.get("oddsData")
        if odds:
            return odds if isinstance(odds, list) else [odds]
        competitors = comp.get("competitors", [])
        if len(competitors) >= 2:
            return None
    return None


def parse_odds_into_probs(odds_list: list[dict[str, Any]]) -> tuple[float, float, float] | None:
    for provider in odds_list:
        home_ml = provider.get("homeTeamOdds", {}).get("moneyLine") or provider.get("homeOdds")
        away_ml = provider.get("awayTeamOdds", {}).get("moneyLine") or provider.get("awayOdds")
        draw_odds = provider.get("drawOdds") or provider.get("drawOdds")
        if home_ml and away_ml:
            from apps.predictions.odds_analyzer import american_to_decimal, decimal_to_implied_prob
            home_dec = american_to_decimal(home_ml) if isinstance(home_ml, int) and abs(home_ml) > 100 else float(home_ml)
            away_dec = american_to_decimal(away_ml) if isinstance(away_ml, int) and abs(away_ml) > 100 else float(away_ml)
            draw_dec = None
            if draw_odds:
                draw_dec = american_to_decimal(draw_odds) if isinstance(draw_odds, int) and abs(draw_odds) > 100 else float(draw_odds)
            h_prob = decimal_to_implied_prob(home_dec)
            a_prob = decimal_to_implied_prob(away_dec)
            d_prob = decimal_to_implied_prob(draw_dec) if draw_dec else (1.0 - h_prob - a_prob)
            if draw_dec:
                return remove_vig(h_prob, d_prob, a_prob)
            return h_prob, d_prob, a_prob
    return None
