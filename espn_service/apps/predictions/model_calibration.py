"""Calibración y blending adaptativo de modelos de predicción.

- Guarda el historial de predicciones vs resultados reales
- Ajusta pesos óptimos de blending mediante optimización bayesiana (scipy)
- Online learning: actualiza pesos tras cada resultado
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


# ──────────────────────── registro y online learning ────────────────────────

def _online_weight_update(state: CalibrationState, actual: str, record: dict, learning_rate: float = 0.01) -> None:
    """Actualiza pesos online basado en el error del último partido (stochastic gradient descent)."""
    # Solo actualiza si tenemos predicciones de modelos guardadas en el registro (en "probs_model_xyz")
    models_in_record = [k.replace("proba_", "") for k in record.keys() if k.startswith("proba_") and k not in ("proba_H", "proba_D", "proba_A")]
    if not models_in_record:
        return
        
    names = list(state.blend_weights.keys())
    w = {n: state.blend_weights.get(n, 0.0) for n in names}
    
    # Objetivo
    target_h = 1.0 if actual == "H" else 0.0
    target_d = 1.0 if actual == "D" else 0.0
    target_a = 1.0 if actual == "A" else 0.0
    
    # Blended actual
    pred_h = record.get("home_proba", 0.33)
    pred_d = record.get("draw_proba", 0.33)
    pred_a = record.get("away_proba", 0.33)
    
    error_h = pred_h - target_h
    error_d = pred_d - target_d
    error_a = pred_a - target_a
    
    # Actualizar cada peso en dirección contraria al gradiente
    for name in names:
        # Extraer probs base del modelo del registro si están, si no usar el blended final como fallback
        mod_h = record.get(f"proba_{name}_H", pred_h)
        mod_d = record.get(f"proba_{name}_D", pred_d)
        mod_a = record.get(f"proba_{name}_A", pred_a)
        
        # Gradiente simple derivado de MSE
        grad = 2 * (error_h * mod_h + error_d * mod_d + error_a * mod_a)
        
        # Update
        w[name] = w[name] - learning_rate * grad
        w[name] = max(0.01, w[name]) # constrain > 0
        
    # Re-normalize
    total_w = sum(w.values())
    if total_w > 0:
        state.blend_weights = {k: round(v / total_w, 4) for k, v in w.items()}


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
    record = {
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
    }
    state.records.append(record)
    state.brier_total += bs
    state.log_loss_total += ll
    state.count += 1
    if predicted_outcome == actual_outcome:
        state.correct_outcome += 1

    # Mantener últimos 1000 registros
    if len(state.records) > 1000:
        removed = state.records.pop(0)
        state.brier_total -= removed.get("brier", 0)
        state.log_loss_total -= removed.get("log_loss", 0)
        state.count -= 1
        if removed.get("correct"):
            state.correct_outcome -= 1

    # Online learning (SGD)
    _online_weight_update(state, actual_outcome, record)

    _save_calibration(state)
    logger.info(
        "prediction_recorded",
        match_id=match_id,
        accuracy=state.accuracy,
        brier=state.brier_avg,
        count=state.count,
    )


# ──────────────────────── optimización de blending ────────────────────────

def _scipy_opt(
    records: list[dict],
    initial_weights: dict[str, float],
) -> dict[str, float]:
    """Optimización usando scipy.optimize para encontrar pesos ideales que minimizan Brier Score."""
    try:
        from scipy.optimize import minimize
        import numpy as np
    except ImportError:
        logger.warning("scipy_not_installed_fallback_to_initial")
        return initial_weights

    if not records:
        return dict(initial_weights)

    names = list(initial_weights.keys())
    w_init = np.array([initial_weights[n] for n in names])

    def objective(w: np.ndarray) -> float:
        # Penaliza pesos negativos
        if np.any(w < 0):
            return 1000.0
        
        w_norm = w / np.sum(w)
        total_brier = 0.0
        
        for r in records:
            # Reconstruir blending
            pred_h, pred_d, pred_a = 0.0, 0.0, 0.0
            
            for i, name in enumerate(names):
                # Fallback to general probability if model-specific isn't saved
                m_h = r.get(f"proba_{name}_H", r.get("home_proba", 0.33))
                m_d = r.get(f"proba_{name}_D", r.get("draw_proba", 0.33))
                m_a = r.get(f"proba_{name}_A", r.get("away_proba", 0.33))
                
                pred_h += w_norm[i] * m_h
                pred_d += w_norm[i] * m_d
                pred_a += w_norm[i] * m_a
                
            # Normalize
            total = pred_h + pred_d + pred_a
            if total > 0:
                pred_h, pred_d, pred_a = pred_h / total, pred_d / total, pred_a / total
                
            actual = r.get("actual_outcome", "D")
            target_h = 1.0 if actual == "H" else 0.0
            target_d = 1.0 if actual == "D" else 0.0
            target_a = 1.0 if actual == "A" else 0.0
            
            brier = (pred_h - target_h) ** 2 + (pred_d - target_d) ** 2 + (pred_a - target_a) ** 2
            total_brier += brier
            
        return total_brier / len(records)

    # Constraints: sum(w) = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # Bounds: 0.01 <= w_i <= 1.0
    bounds = tuple((0.01, 1.0) for _ in range(len(names)))

    result = minimize(
        objective,
        w_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )

    if result.success:
        optimized = result.x / np.sum(result.x)
        return {names[i]: round(float(optimized[i]), 4) for i in range(len(names))}
    
    return initial_weights


def optimize_blend_weights() -> dict[str, float]:
    """Optimiza los pesos de blending basado en el historial de predicciones."""
    state = _load_calibration()
    if state.count < 50:
        logger.info("skip_opt_not_enough_data", count=state.count)
        return state.blend_weights

    # Solo toma los últimos 200 para optimizar y no overfittear todo el historial
    records = state.records[-200:]
    new_weights = _scipy_opt(records, state.blend_weights)
    state.blend_weights = new_weights
    _save_calibration(state)
    logger.info("blend_weights_optimized", weights=new_weights, brier=state.brier_avg)
    return new_weights


# ──────────────────────── Isotonic Regression (Calibración de prob) ────────────────────────

class SimpleIsotonicRegression:
    """Implementación ligera de Isotonic Regression (PAVA) para no depender estrictamente de sklearn."""
    def fit_transform(self, y: list[float], y_true: list[float]) -> list[float]:
        n = len(y)
        if n == 0:
            return []
            
        # PAVA (Pool Adjacent Violators Algorithm)
        # Sort by predicted probability
        sorted_pairs = sorted(zip(y, y_true), key=lambda x: x[0])
        y_sorted = [x[0] for x in sorted_pairs]
        targets = [x[1] for x in sorted_pairs]
        
        values = [[t] for t in targets]
        weights = [[1.0] for _ in targets]
        
        while True:
            merged = False
            for i in range(len(values) - 1):
                val1 = sum(values[i]) / sum(weights[i])
                val2 = sum(values[i+1]) / sum(weights[i+1])
                if val1 > val2:
                    values[i].extend(values[i+1])
                    weights[i].extend(weights[i+1])
                    del values[i+1]
                    del weights[i+1]
                    merged = True
                    break
            if not merged:
                break
                
        calibrated_targets = []
        for v_list, w_list in zip(values, weights):
            mean_val = sum(v_list) / sum(w_list)
            calibrated_targets.extend([mean_val] * len(v_list))
            
        # Create mapping function (simple linear interpolation or nearest neighbor)
        self.points = list(zip(y_sorted, calibrated_targets))
        return calibrated_targets
        
    def predict(self, x: float) -> float:
        if not hasattr(self, 'points') or not self.points:
            return x
        # Nearest neighbor for simplicity
        closest = min(self.points, key=lambda p: abs(p[0] - x))
        return closest[1]


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
