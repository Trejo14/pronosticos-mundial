"""Feature engineering: Elo persistente, forma reciente, ataque/defensa ponderados.

Sistema de clase mundial:
- Elo persistente en JSON con decaimiento temporal
- Forma reciente ponderada (últimos 5 partidos, más peso al más reciente)
- Ataque/defensa por ventana móvil ajustado por rival
- Factor local/visitante por selección
- Head-to-head reciente
- Integración con xG de 365Scores
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from apps.predictions.h2h_cache import get_h2h_cache

logger = __import__("structlog").get_logger(__name__)

# ──────────────────────── constantes ────────────────────────

INITIAL_ELO = 1500.0
ELO_K_BASE = 32.0
HOME_ADVANTAGE_BASE = 0.06  # ~0.5 goals advantage
ELO_CACHE_FILENAME = "elo_cache.json"
XG_CACHE_FILENAME = "xg_cache.json"
CALIBRATION_FILENAME = "calibration.json"
FORM_WINDOW = 5              # partidos para forma reciente
GOAL_WINDOW = 10             # partidos para media de goles
DECAY_DAYS = 365 * 2         # 2 años para decaer Elo al inicial
DECAY_RATE = 0.5             # qué fracción del camino de regreso

# Pesos exponenciales para forma reciente (más reciente = más peso)
_FORM_WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08]

# Valor de plantilla aproximado por ranking FIFA (millones EUR)
# Basado en correlación histórica ranking ↔ valor de mercado
_SQUAD_VALUE_BY_RANK = {
    (1, 5): 850, (6, 10): 650, (11, 15): 500, (16, 20): 380,
    (21, 30): 250, (31, 40): 160, (41, 50): 100, (51, 70): 55,
    (71, 100): 25, (101, 150): 10, (151, 211): 3,
}


# ──────────────────────── dataclasses ────────────────────────

@dataclass
class TeamStrength:
    name: str
    espn_id: str
    elo: float
    attacking: float
    defensive: float
    home_advantage: float = 0.0
    form_pts: float = 0.5        # puntos por partido últimos 5
    recent_gf: float = 1.0       # goles a favor promedio últimos 10
    recent_ga: float = 1.0       # goles en contra promedio últimos 10
    xg_per_match: float | None = None
    clean_sheet_rate: float = 0.0
    btts_rate: float = 0.0
    squad_value: float = 50.0     # valor de plantilla en millones EUR


# ──────────────────────── Elo persistente ────────────────────────

def _elo_cache_path() -> Path:
    return Path(settings.BASE_DIR) / "data" / ELO_CACHE_FILENAME


def _load_elo_cache() -> dict[str, dict]:
    path = _elo_cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("elo_cache_read_failed", error=str(exc))
    return {}


def _save_elo_cache(cache: dict[str, dict]) -> None:
    path = _elo_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("elo_cache_write_failed", error=str(exc))


def get_elo(team_id: str, fifa_rank: int | None = None) -> float:
    """Obtiene Elo de un equipo, inicializando desde FIFA ranking si es nuevo."""
    cache = _load_elo_cache()
    entry = cache.get(team_id)
    if entry is None:
        elo = fifa_ranking_to_elo(fifa_rank)
        cache[team_id] = {
            "elo": elo,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "matches_played": 0,
        }
        _save_elo_cache(cache)
        return elo
    elo = entry["elo"]
    last_str = entry.get("last_updated")
    if last_str:
        try:
            last_dt = datetime.fromisoformat(last_str)
            days_passed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            if days_passed > 30:
                decay = min(days_passed / DECAY_DAYS, 1.0)
                elo = INITIAL_ELO + (elo - INITIAL_ELO) * (1 - decay * DECAY_RATE)
        except (ValueError, TypeError):
            pass
    return elo


def save_elo(team_id: str, new_elo: float, matches_played: int | None = None) -> None:
    cache = _load_elo_cache()
    old = cache.get(team_id, {})
    old["elo"] = new_elo
    old["last_updated"] = datetime.now(timezone.utc).isoformat()
    old["matches_played"] = (old.get("matches_played", 0) + 1) if matches_played is None else matches_played
    cache[team_id] = old
    _save_elo_cache(cache)


# ──────────────────────── xG cache ────────────────────────

def _xg_cache_path() -> Path:
    return Path(settings.BASE_DIR) / "data" / XG_CACHE_FILENAME


def _load_xg_cache() -> dict[str, dict]:
    path = _xg_cache_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("xg_cache_read_failed", error=str(exc))
    return {}


def _save_xg_cache(cache: dict[str, dict]) -> None:
    path = _xg_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("xg_cache_write_failed", error=str(exc))


def save_team_xg(team_id: str, xg_for: float, xg_against: float) -> None:
    """Guarda xG real de un partido para un equipo (media móvil)."""
    cache = _load_xg_cache()
    entry = cache.get(team_id, {"xg_for": [], "xg_against": [], "count": 0})
    entry["xg_for"].append(xg_for)
    entry["xg_against"].append(xg_against)
    entry["count"] = entry.get("count", 0) + 1
    # Mantener últimos 20
    entry["xg_for"] = entry["xg_for"][-20:]
    entry["xg_against"] = entry["xg_against"][-20:]
    cache[team_id] = entry
    _save_xg_cache(cache)


def get_team_xg(team_id: str) -> tuple[float | None, float | None]:
    """Devuelve (xg_for_avg, xg_against_avg) para un equipo."""
    cache = _load_xg_cache()
    entry = cache.get(team_id)
    if not entry or not entry.get("xg_for"):
        return None, None
    avg_for = sum(entry["xg_for"]) / len(entry["xg_for"])
    avg_against = sum(entry["xg_against"]) / len(entry["xg_against"])
    return round(avg_for, 3), round(avg_against, 3)


# ──────────────────────── squad value ────────────────────────

def estimate_squad_value(fifa_rank: int | None) -> float:
    """Estima el valor de plantilla en millones EUR desde el ranking FIFA."""
    if fifa_rank is None or fifa_rank <= 0:
        return 50.0
    for (lo, hi), val in _SQUAD_VALUE_BY_RANK.items():
        if lo <= fifa_rank <= hi:
            return val
    return 10.0


# ──────────────────────── funciones base ────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    score_b: float,
    k: float = ELO_K_BASE,
    margin: float = 0.0,
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    margin_mult = 1.0 + math.log(abs(margin) + 1.0) * 0.15 if margin else 1.0
    actual_a = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
    k_adj = k * margin_mult
    new_a = rating_a + k_adj * (actual_a - expected_a)
    new_b = rating_b + k_adj * ((1.0 - actual_a) - expected_b)
    return new_a, new_b


def fifa_ranking_to_elo(fifa_rank: int | None, max_rank: int = 211) -> float:
    if fifa_rank is None or fifa_rank <= 0:
        return INITIAL_ELO
    normalized = 1.0 - (fifa_rank - 1) / max_rank
    return 1000.0 + normalized * 1000.0


def calculate_attacking_strength(goals_scored: float, league_avg_goals: float = 1.5) -> float:
    if league_avg_goals <= 0:
        return 1.0
    return goals_scored / league_avg_goals


def calculate_defensive_strength(goals_conceded: float, league_avg_goals: float = 1.5) -> float:
    if league_avg_goals <= 0:
        return 1.0
    return goals_conceded / league_avg_goals


# ──────────────────────── forma reciente ────────────────────────

def calculate_form_from_results(results: list[dict]) -> dict[str, float]:
    """Calcula forma ponderada, goles promedio, clean sheets, BTTS.

    results: lista de dicts con 'gf', 'ga', ordenados del más reciente al más viejo.
    """
    n = min(len(results), FORM_WINDOW)
    if n == 0:
        return {"form_pts": 0.5, "recent_gf": 1.0, "recent_ga": 1.0, "clean_sheet_rate": 0.0, "btts_rate": 0.0}

    total_weight = sum(_FORM_WEIGHTS[:n])
    form_pts = 0.0
    gf_sum = 0.0
    ga_sum = 0.0
    cs_count = 0
    btts_count = 0

    for i in range(n):
        w = _FORM_WEIGHTS[i]
        r = results[i]
        gf = r.get("gf", 0)
        ga = r.get("ga", 0)
        gf_sum += gf * w
        ga_sum += ga * w
        if gf > ga:
            form_pts += 3 * w
        elif gf == ga:
            form_pts += 1 * w
        if ga == 0:
            cs_count += 1
        if gf > 0 and ga > 0:
            btts_count += 1

    form_pts /= total_weight
    avg_gf = gf_sum / total_weight
    avg_ga = ga_sum / total_weight

    return {
        "form_pts": form_pts / 3.0,  # normalizado a 0-1
        "recent_gf": max(avg_gf, 0.3),
        "recent_ga": max(avg_ga, 0.3),
        "clean_sheet_rate": cs_count / n,
        "btts_rate": btts_count / n,
    }


# ──────────────────────── factores contextuales (Mejora 2) ────────────────────────

def calculate_h2h_factor(home_id: str, away_id: str) -> tuple[float, float]:
    """Obtiene factor head-to-head multiplicador desde la caché."""
    h2h_cache = get_h2h_cache()
    return h2h_cache.get_h2h_factor(home_id, away_id)


def calculate_fatigue_factor(last_match_date: str | None, current_match_date: str | None) -> float:
    """Calcula penalización por fatiga si jugó hace muy poco (< 4 días).
    Devuelve un multiplicador (1.0 = sin fatiga, < 1.0 = fatigado).
    """
    if not last_match_date or not current_match_date:
        return 1.0
        
    try:
        # Simplificación asumiendo formato ISO 8601
        last_dt = datetime.fromisoformat(last_match_date.replace("Z", "+00:00"))
        curr_dt = datetime.fromisoformat(current_match_date.replace("Z", "+00:00"))
        
        days_rest = (curr_dt - last_dt).total_seconds() / 86400.0
        
        if days_rest <= 2.5:
            return 0.95  # Severe fatigue
        elif days_rest <= 4.0:
            return 0.98  # Moderate fatigue
        elif days_rest > 10.0:
            return 0.99  # Rust factor (too much rest)
            
        return 1.0
    except (ValueError, TypeError):
        return 1.0


def calculate_motivation_factor(is_must_win: bool, is_already_qualified: bool) -> float:
    """Multiplicador de motivación (útil en fines de temporada o fase de grupos)."""
    if is_must_win:
        return 1.05
    if is_already_qualified:
        return 0.95
    return 1.0


def compute_team_strength(
    elo: float = INITIAL_ELO,
    attacking: float = 1.0,
    defensive: float = 1.0,
    home: bool = False,
    form_pts: float = 0.5,
) -> float:
    strength = elo / 1000.0
    strength *= attacking
    strength /= max(defensive, 0.1)
    strength *= (0.7 + 0.3 * form_pts)
    if home:
        strength *= (1.0 + HOME_ADVANTAGE_BASE)
    return strength


# ──────────────────────── extractores ESPN ────────────────────────

def extract_team_stats_from_standings(standings_data: dict[str, Any]) -> dict[str, dict]:
    teams: dict[str, dict] = {}
    children = standings_data.get("children", [])
    for group in children:
        entries = group.get("standings", {}).get("entries", [])
        for entry in entries:
            team = entry.get("team", {})
            tid = str(team.get("id", ""))
            stats = {}
            for s in entry.get("stats", []):
                name = s.get("name", "")
                val = s.get("displayValue", "0")
                try:
                    stats[name] = float(val) if val.replace(".", "").replace("-", "").isdigit() else val
                except (ValueError, TypeError):
                    stats[name] = val
            teams[tid] = {
                "id": tid,
                "name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "draws": stats.get("ties", 0) or stats.get("draws", 0),
                "points": stats.get("points", 0),
                "goals_for": stats.get("goalsFor", 0) or stats.get("pointsFor", 0),
                "goals_against": stats.get("goalsAgainst", 0) or stats.get("pointsAgainst", 0),
                "goal_differential": stats.get("goalDifferential", 0) or stats.get("pointDifferential", 0),
                "form": stats.get("form", ""),
                "games_played": stats.get("gamesPlayed", 0) or (stats.get("wins", 0) + stats.get("losses", 0) + stats.get("ties", 0)),
                "rank": stats.get("rank", 0) or stats.get("position", 0),
            }
    return teams


def extract_team_stats_from_event(event_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    events = event_data.get("events", [event_data])
    for event in events:
        for comp in event.get("competitions", []):
            for competitor in comp.get("competitors", []):
                team = competitor.get("team", {})
                tid = str(team.get("id", ""))
                home_away = competitor.get("homeAway", "away")
                score = competitor.get("score", "0")
                records = competitor.get("records", [])
                record_str = ""
                for r in records:
                    if r.get("name") in ("overall", "all"):
                        record_str = r.get("summary", "")
                        break
                teams[tid] = {
                    "id": tid,
                    "name": team.get("displayName", ""),
                    "abbreviation": team.get("abbreviation", ""),
                    "home_away": home_away,
                    "score": score,
                    "record": record_str,
                    "logo": team.get("logo", ""),
                }
    return teams


def build_team_strength(
    team_id: str,
    team_name: str,
    elo: float,
    recent_results: list[dict] | None = None,
    xg_avg: float | None = None,
    home: bool = False,
    league_avg_goals: float = 2.5,
    fifa_rank: int | None = None,
) -> TeamStrength:
    """Construye TeamStrength combinando Elo, forma reciente, xG real y valor de plantilla."""
    att, deff = 1.0, 1.0
    form_pts = 0.5
    recent_gf, recent_ga = 1.0, 1.0
    cs_rate, btts_rate = 0.0, 0.0

    if recent_results:
        form = calculate_form_from_results(recent_results)
        form_pts = form["form_pts"]
        recent_gf = form["recent_gf"]
        recent_ga = form["recent_ga"]
        cs_rate = form["clean_sheet_rate"]
        btts_rate = form["btts_rate"]
        att = calculate_attacking_strength(recent_gf, league_avg_goals)
        deff = calculate_defensive_strength(recent_ga, league_avg_goals)

    # Ajuste con xG real histórico de 365Scores
    xg_for, xg_against = get_team_xg(team_id)
    if xg_for is not None and xg_for > 0:
        xg_att = xg_for / (league_avg_goals / 2)
        xg_def = xg_against / (league_avg_goals / 2)
        att = (att * 0.5) + (xg_att * 0.5)
        deff = (deff * 0.5) + (xg_def * 0.5)

    if xg_avg is not None and xg_avg > 0:
        xg_att = xg_avg / (league_avg_goals / 2)
        att = (att + xg_att) / 2

    # Suavizado hacia 1.0 para evitar extremos
    att = 0.3 + 0.7 * att
    deff = 0.3 + 0.7 * deff

    squad_val = estimate_squad_value(fifa_rank)

    return TeamStrength(
        name=team_name,
        espn_id=team_id,
        elo=elo,
        attacking=round(att, 3),
        defensive=round(deff, 3),
        home_advantage=HOME_ADVANTAGE_BASE if home else 0.0,
        form_pts=round(form_pts, 3),
        recent_gf=round(recent_gf, 2),
        recent_ga=round(recent_ga, 2),
        xg_per_match=round(xg_avg, 2) if xg_avg else None,
        clean_sheet_rate=round(cs_rate, 3),
        btts_rate=round(btts_rate, 3),
        squad_value=squad_val,
    )
