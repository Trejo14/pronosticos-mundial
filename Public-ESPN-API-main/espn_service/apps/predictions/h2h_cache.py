"""Head-to-head records cache.

Stores and analyzes recent direct matchups between teams to adjust baseline probabilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings

logger = __import__("structlog").get_logger(__name__)

H2H_FILENAME = "h2h_cache.json"


@dataclass
class H2HMatchup:
    team_a_id: str
    team_b_id: str
    matches: list[dict[str, Any]] = field(default_factory=list)
    last_updated: str = ""

    def add_match(self, date: str, a_score: int, b_score: int, a_was_home: bool) -> None:
        """Add a match if it's not already in the history."""
        # Simple deduplication by date
        for m in self.matches:
            if m["date"][:10] == date[:10]:
                return
                
        self.matches.append({
            "date": date,
            "a_score": a_score,
            "b_score": b_score,
            "a_was_home": a_was_home
        })
        # Keep only the last 10 matchups
        self.matches.sort(key=lambda x: x["date"], reverse=True)
        if len(self.matches) > 10:
            self.matches = self.matches[:10]
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def get_factor(self, is_a_home: bool) -> float:
        """Calculate the H2H factor for Team A.
        
        Returns a multiplier centered around 1.0. 
        > 1.0 means Team A has historical advantage.
        < 1.0 means Team B has historical advantage.
        """
        if not self.matches:
            return 1.0

        a_pts = 0.0
        total_weight = 0.0

        for i, m in enumerate(self.matches):
            # Recency weighting (most recent = weight 1.0, 10th = weight 0.1)
            weight = 1.0 - (i * 0.09)
            
            a_score = m["a_score"]
            b_score = m["b_score"]
            
            if a_score > b_score:
                pts = 3.0
            elif a_score == b_score:
                pts = 1.0
            else:
                pts = 0.0
                
            a_pts += pts * weight
            total_weight += 3.0 * weight

        if total_weight == 0:
            return 1.0
            
        win_pct = a_pts / total_weight
        
        # Scale to max ±5% adjustment
        # If win_pct is 1.0 (Team A always wins), factor = 1.05
        # If win_pct is 0.0 (Team A always loses), factor = 0.95
        # If win_pct is 0.33 (even), factor = 1.0
        # (win_pct - 0.33) goes from -0.33 to 0.67
        # We'll map (0, 1) -> (0.95, 1.05)
        # linear mapping: y = 0.1 * x + 0.95
        factor = 0.95 + (win_pct * 0.1)
        
        # Minor bump if team A is playing at home and historically won at home
        if is_a_home:
            home_wins = sum(1 for m in self.matches if m["a_was_home"] and m["a_score"] > m["b_score"])
            if home_wins >= 2:
                factor += 0.01

        return round(factor, 3)


class H2HCache:
    def __init__(self):
        self.path = Path(settings.BASE_DIR) / "data" / H2H_FILENAME
        self._data: dict[str, H2HMatchup] = {}
        self._load()

    def _get_key(self, id1: str, id2: str) -> tuple[str, str, str]:
        # Sort so key is independent of home/away
        sorted_ids = sorted([id1, id2])
        return f"{sorted_ids[0]}_vs_{sorted_ids[1]}", sorted_ids[0], sorted_ids[1]

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    self._data[k] = H2HMatchup(
                        team_a_id=v["team_a_id"],
                        team_b_id=v["team_b_id"],
                        matches=v.get("matches", []),
                        last_updated=v.get("last_updated", "")
                    )
            except Exception as e:
                logger.warning("h2h_cache_load_failed", error=str(e))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dump = {k: {"team_a_id": v.team_a_id, "team_b_id": v.team_b_id, "matches": v.matches, "last_updated": v.last_updated} for k, v in self._data.items()}
            self.path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("h2h_cache_save_failed", error=str(e))

    def record_matchup(self, id1: str, id2: str, date: str, score1: int, score2: int, is_id1_home: bool) -> None:
        key, team_a, team_b = self._get_key(id1, id2)
        if key not in self._data:
            self._data[key] = H2HMatchup(team_a_id=team_a, team_b_id=team_b)
        
        matchup = self._data[key]
        
        # Ensure we map scores to team A and team B correctly based on the sorted keys
        if id1 == team_a:
            a_score = score1
            b_score = score2
            a_was_home = is_id1_home
        else:
            a_score = score2
            b_score = score1
            a_was_home = not is_id1_home
            
        matchup.add_match(date, a_score, b_score, a_was_home)
        self.save()

    def get_h2h_factor(self, home_id: str, away_id: str) -> tuple[float, float]:
        """Returns (home_factor, away_factor).
        
        Product of factors should be ~1.0 (if home factor is 1.05, away is ~0.95).
        """
        if not home_id or not away_id:
            return 1.0, 1.0
            
        key, team_a, team_b = self._get_key(home_id, away_id)
        if key not in self._data:
            return 1.0, 1.0
            
        matchup = self._data[key]
        
        if home_id == team_a:
            factor_a = matchup.get_factor(is_a_home=True)
            factor_b = 2.0 - factor_a
            return factor_a, factor_b
        else:
            # Home is team B
            factor_a = matchup.get_factor(is_a_home=False)
            factor_b = 2.0 - factor_a
            return factor_b, factor_a

_global_h2h_cache = None

def get_h2h_cache() -> H2HCache:
    global _global_h2h_cache
    if _global_h2h_cache is None:
        _global_h2h_cache = H2HCache()
    return _global_h2h_cache
