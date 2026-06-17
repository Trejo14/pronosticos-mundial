"""Calibración y blending adaptativo de modelos de predicción.

- Guarda el historial de predicciones vs resultados reales
- Ajusta pesos óptimos de blending mediante optimización
- Calcula Brier score, log-loss, accuracy por umbral
- Persiste el modelo aprendido en JSON
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings

logger = __import__("structlog").get_logger(__name__)

CALIBRATION_FILENAME = "calibration.json"
BRIER_REFERENCE = 0.25
LOG_LOSS_REFERENCE = 0.70


@dataclass
class PredictionRecord:
    match_id: str
    home_team: str
    away_team: str
    home_proba: float
    draw_proba: float
    away_proba: float
    actual_home: int | None
    actual_away: int | None
    actual_outcome: str | None  # H / D / A
    brier_score: float = 0.0
    correct: bool = False
    correct_exact: bool = False
    timestamp: str = ""


def _outcome_from_score(h: int, a: int) -> str:
    if h > a:
        return "H"
    if h == a:
        return "D"
    return "A"


def _brier(h: float, d: float, a: float, actual: str) -> float:
    """Brier score: menor es mejor (0 = perfecto, 2 = peor)."""
    return (h - (1.0 if actual == "H" else 0.0)) ** 2 + \
           (d - (1.0 if actual == "D" else 0.0)) ** 2 + \
           (a - (1.0 if actual == "A" else 0.0)) ** 2


def _log_loss(h: float, d: float, a: float, actual: str) -> float:
    probs = {"H": h, "D": d, "A": a}
    p = probs.get(actual, 0.0)
    p = max(min(p, 1 - 1e-15), 1e-15)
    return -math.log(p)


@dataclass
class CalibrationState:
    records: list[dict] = field(default_factory=list)
    blend_weights: dict[str, float] = field(default_factory=lambda: {
        "poisson": 0.30,
        "elo": 0.25,
        "market": 0.20,
        "form": 0.15,
        "dc": 0.10,
    })
    brier_total: float = 0.0
    log_loss_total: float = 0.0
    count: int = 0
    correct_outcome: int = 0
    correct_exact: int = 0
    last_updated: str = ""

    @property
    def brier_avg(self) -> float:
        return self.brier_total / max(self.count, 1)

    @property
    def log_loss_avg(self) -> float:
        return self.log_loss_total / max(self.count, 1)

    @property
    def accuracy(self) -> float:
        return self.correct_outcome / max(self.count, 1)

    @property
    def exact_accuracy(self) -> float:
        return self.correct_exact / max(self.count, 1)

    @property
    def brier_skill_score(self) -> float:
        """Brier Skill Score: 1 = perfecto, 0 = referencia, negativo = peor."""
        return 1.0 - (self.brier_avg / BRIER_REFERENCE)


# ──────────────────────── path ────────────────────────

def _calibration_path() -> Path:
    return Path(settings.BASE_DIR) / "data" / CALIBRATION_FILENAME


# ──────────────────────── load / save ────────────────────────

def _load_calibration() -> CalibrationState:
    path = _calibration_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CalibrationState(**{k: data.get(k, v) for k, v in CalibrationState().__dict__.items() if not k.startswith("_")})
        except Exception as exc:
            logger.warning("calibration_load_failed", error=str(exc))
    return CalibrationState()


def _save_calibration(state: CalibrationState) -> None:
    path = _calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = datetime.now(timezone.utc).isoformat()
    try:
        path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("calibration_save_failed", error=str(exc))


# ──────────────────────── registro ────────────────────────

def record_prediction(
    match_id: str,
    home_team: str,
    away_team: str,
    home_proba: float,
    draw_proba: float,
    away_proba: float,
    actual_home: int,
    actual_away: int,
) -> None:
    """Registra una predicción y su resultado real para calibración."""
    actual_outcome = _outcome_from_score(actual_home, actual_away)
    bs = _brier(home_proba, draw_proba, away_proba, actual_outcome)
    ll = _log_loss(home_proba, draw_proba, away_proba, actual_outcome)

    predicted_outcome = max(
        [("H", home_proba), ("D", draw_proba), ("A", away_proba)],
        key=lambda x: x[1],
    )[0]

    state = _load_calibration()
    state.records.append({
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_proba": round(home_proba, 4),
        "draw_proba": round(draw_proba, 4),
        "away_proba": round(away_proba, 4),
        "actual_home": actual_home,
        "actual_away": actual_away,
        "actual_outcome": actual_outcome,
        "predicted_outcome": predicted_outcome,
        "brier": round(bs, 4),
        "log_loss": round(ll, 4),
        "correct": predicted_outcome == actual_outcome,
        "correct_exact": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    state.brier_total += bs
    state.log_loss_total += ll
    state.count += 1
    if predicted_outcome == actual_outcome:
        state.correct_outcome += 1

    # Mantener últimos 500 registros
    if len(state.records) > 500:
        removed = state.records.pop(0)
        state.brier_total -= removed.get("brier", 0)
        state.log_loss_total -= removed.get("log_loss", 0)
        state.count -= 1
        if removed.get("correct"):
            state.correct_outcome -= 1

    _save_calibration(state)
    logger.info(
        "prediction_recorded",
        match_id=match_id,
        accuracy=state.accuracy,
        brier=state.brier_avg,
        count=state.count,
    )


# ──────────────────────── optimización de blending ────────────────────────

def _simple_opt(
    records: list[dict],
    initial_weights: dict[str, float],
    steps: int = 200,
    step_size: float = 0.02,
) -> dict[str, float]:
    """Optimización greed: busca pesos que minimizan Brier score."""
    if not records:
        return dict(initial_weights)

    names = list(initial_weights.keys())
    w = [initial_weights[n] for n in names]

    def brier_for_weights(weights: list[float]) -> float:
        total = 0.0
        count = 0
        for r in records:
            blended = sum(weights[i] * r.get(f"proba_{names[i]}", 0) for i in range(len(names)))
            blended = max(min(blended, 0.999), 0.001)
            actual = r.get("actual_outcome", "D")
            hs = r.get("proba_H", 0.33)
            ds = r.get("proba_D", 0.33)
            aas = r.get("proba_A", 0.33)
            target = 1.0 if actual == "H" else 0.0
            total += (blended - target) ** 2
            count += 1
        return total / max(count, 1)

    best_w = list(w)
    best_brier = brier_for_weights(w)

    for _ in range(steps):
        i = _ % len(names)
        old = w[i]
        for delta in [step_size, -step_size]:
            w[i] = max(0.01, min(0.90, old + delta))
            total_w = sum(w)
            w_norm = [x / total_w for x in w]
            b = brier_for_weights(w_norm)
            if b < best_brier:
                best_brier = b
                best_w = list(w_norm)
                w[i] = w_norm[i] * total_w
            else:
                w[i] = old
    return {names[i]: round(best_w[i], 4) for i in range(len(names))}


def optimize_blend_weights() -> dict[str, float]:
    """Optimiza los pesos de blending basado en el historial de predicciones."""
    state = _load_calibration()
    if state.count < 10:
        logger.info("skip_opt_not_enough_data", count=state.count)
        return state.blend_weights

    records = state.records
    new_weights = _simple_opt(records, state.blend_weights)
    state.blend_weights = new_weights
    _save_calibration(state)
    logger.info("blend_weights_optimized", weights=new_weights, brier=state.brier_avg)
    return new_weights


# ──────────────────────── consulta ────────────────────────

def get_calibration_summary() -> dict[str, Any]:
    """Resumen del estado actual de calibración."""
    state = _load_calibration()
    return {
        "count": state.count,
        "accuracy": round(state.accuracy, 4),
        "exact_accuracy": round(state.exact_accuracy, 4),
        "brier_avg": round(state.brier_avg, 4),
        "log_loss_avg": round(state.log_loss_avg, 4),
        "brier_skill_score": round(state.brier_skill_score, 4),
        "blend_weights": state.blend_weights,
        "last_updated": state.last_updated,
    }
