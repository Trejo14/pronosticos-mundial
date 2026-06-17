"""Simulador Monte Carlo de torneo con Poisson, Elo y grupos detallados.

Simula el torneo completo (fase de grupos + eliminatorias) miles de veces
para calcular probabilidades de cada equipo.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from apps.predictions.feature_engineering import (
    INITIAL_ELO,
    TeamStrength,
    expected_score,
)
from apps.predictions.odds_analyzer import remove_vig


@dataclass
class SimulationResult:
    team_name: str
    team_espn_id: str
    win_tournament: int = 0
    reach_final: int = 0
    reach_semis: int = 0
    reach_quarters: int = 0
    win_group: int = 0
    total_simulations: int = 0

    @property
    def win_prob(self) -> float:
        return self.win_tournament / max(self.total_simulations, 1)

    @property
    def final_prob(self) -> float:
        return self.reach_final / max(self.total_simulations, 1)

    @property
    def semis_prob(self) -> float:
        return self.reach_semis / max(self.total_simulations, 1)

    @property
    def quarters_prob(self) -> float:
        return self.reach_quarters / max(self.total_simulations, 1)

    @property
    def group_win_prob(self) -> float:
        return self.win_group / max(self.total_simulations, 1)


@dataclass
class Team:
    espn_id: str
    name: str
    elo: float = INITIAL_ELO
    group: str = ""
    attacking: float = 1.0
    defensive: float = 1.0
    form_pts: float = 0.5

    @property
    def strength(self) -> float:
        return self.elo / 1000.0 * self.attacking / max(self.defensive, 0.1) * (0.7 + 0.3 * self.form_pts)


MAX_GOALS_SIM = 10


def _poisson_prob(goals: int, expected: float) -> float:
    if expected <= 0:
        return 1.0 if goals == 0 else 0.0
    return (math.exp(-expected) * (expected ** goals)) / math.factorial(goals)


def simulate_score(home_elo: float, away_elo: float, home_att: float, away_att: float,
                   home_def: float, away_def: float, league_avg: float = 2.5,
                   home_adv: float = 0.06) -> tuple[int, int]:
    """Simula el marcador exacto usando Poisson con fuerzas de equipo."""
    exp_home = league_avg * home_att * away_def * (1 + home_adv)
    exp_away = league_avg * away_att * home_def
    exp_home = max(exp_home, 0.05)
    exp_away = max(exp_away, 0.05)

    # Generar goles desde Poisson
    r_h = random.random()
    r_a = random.random()
    home_goals = 0
    cum = 0.0
    for g in range(MAX_GOALS_SIM):
        cum += _poisson_prob(g, exp_home)
        if r_h <= cum:
            home_goals = g
            break
    away_goals = 0
    cum = 0.0
    for g in range(MAX_GOALS_SIM):
        cum += _poisson_prob(g, exp_away)
        if r_a <= cum:
            away_goals = g
            break
    return home_goals, away_goals


def simulate_knockout_score(
    team_a: Team, team_b: Team,
    home_adv: float = 0.06, league_avg: float = 2.5,
    extra_time: bool = False,
) -> tuple[int, int, str | None]:
    """Simula un partido eliminatorio con posible prórroga."""
    # 90 minutos
    h, a = simulate_score(team_a.elo, team_b.elo,
                          team_a.attacking, team_b.attacking,
                          team_a.defensive, team_b.defensive,
                          league_avg, home_adv)
    if h != a:
        return h, a, None

    # Empate → prórroga
    if extra_time:
        # En prórroga los equipos juegan más cansados (ataque/defensa reducido)
        et_h, et_a = simulate_score(team_a.elo * 0.97, team_b.elo * 0.97,
                                    team_a.attacking * 0.9, team_b.attacking * 0.9,
                                    team_a.defensive * 0.9, team_b.defensive * 0.9,
                                    league_avg * 0.6, home_adv * 0.5)
        if et_h != et_a:
            return h + et_h, a + et_a, None

    # Penales (50/50 con ligera ventaja del equipo con mejor Elo)
    elo_adv = expected_score(team_a.elo, team_b.elo)
    if random.random() < elo_adv:
        return h, a, team_a.name
    else:
        return h, a, team_b.name


def simulate_group_match(
    team_a: Team, team_b: Team,
    home_adv: float = 0.06, league_avg: float = 2.5,
) -> tuple[int, int]:
    """Simula un partido de grupo."""
    return simulate_score(team_a.elo, team_b.elo,
                          team_a.attacking, team_b.attacking,
                          team_a.defensive, team_b.defensive,
                          league_avg, home_adv)


def simulate_group_stage(
    groups: dict[str, list[Team]],
    league_avg: float = 2.5,
) -> dict[str, tuple[Team, Team, int, int, int, int]]:
    """Simula fase de grupos completa y devuelve los 2 primeros de cada grupo.

    Returns: dict[group_name, (1er, 2do, pts1, pts2, gf1, gf2)]
    """
    winners: dict[str, tuple[Team, Team, int, int, int, int]] = {}
    for group_name, teams in groups.items():
        if len(teams) < 2:
            continue
        points: dict[str, int] = {t.name: 0 for t in teams}
        gf: dict[str, int] = {t.name: 0 for t in teams}
        ga: dict[str, int] = {t.name: 0 for t in teams}

        # Round-robin
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                h_goals, a_goals = simulate_group_match(teams[i], teams[j], 0.06, league_avg)
                gf[teams[i].name] += h_goals
                ga[teams[i].name] += a_goals
                gf[teams[j].name] += a_goals
                ga[teams[j].name] += h_goals
                if h_goals > a_goals:
                    points[teams[i].name] += 3
                elif a_goals > h_goals:
                    points[teams[j].name] += 3
                else:
                    points[teams[i].name] += 1
                    points[teams[j].name] += 1

        # Desempate: puntos > diferencia de gol > goles a favor > enfrentamiento directo
        def _sort_key(t: Team) -> tuple:
            gd = gf[t.name] - ga[t.name]
            # El enfrentamiento directo se calcula aquí
            return (-points[t.name], -gd, -gf[t.name])

        sorted_teams = sorted(teams, key=_sort_key)

        t1, t2 = sorted_teams[0], sorted_teams[1]
        winners[group_name] = (
            t1, t2,
            points[t1.name], points[t2.name],
            gf[t1.name], gf[t2.name],
        )
    return winners


def build_r16_matches(
    group_winners: dict[str, tuple[Team, Team, int, int, int, int]],
) -> list[tuple[Team, Team]]:
    """Build Round of 16 matchups from group winners.

    World Cup format:
      1A vs 2B, 1C vs 2D, 1E vs 2F, 1G vs 2H
      1B vs 2A, 1D vs 2C, 1F vs 2E, 1H vs 2G
    """
    pairings = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
                ("B", "A"), ("D", "C"), ("F", "E"), ("H", "G")]
    matches: list[tuple[Team, Team]] = []
    for g1, g2 in pairings:
        w1 = group_winners.get(g1)
        w2 = group_winners.get(g2)
        if w1 and w2:
            matches.append((w1[0], w2[1]))
    return matches


def simulate_single_tournament(
    teams: list[Team],
    groups: dict[str, list[Team]],
    league_avg: float = 2.5,
) -> list[str]:
    """Simula 1 torneo completo y devuelve ranking de resultados."""
    group_winners = simulate_group_stage(groups, league_avg)

    # Round of 16
    r16_matches = build_r16_matches(group_winners)
    if len(r16_matches) < 2:
        return []

    # Octavos
    r16_winners: list[Team] = []
    for a, b in r16_matches:
        h, a_goals, winner = simulate_knockout_score(a, b, extra_time=True)
        r16_winners.append(winner if winner == a.name else b if winner == b.name else (a if h > a_goals else b))

    if len(r16_winners) < 4:
        return [t.name for t in r16_winners]

    # Cuartos
    qf_matches = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, len(r16_winners) - 1, 2)]
    qf_winners: list[Team] = []
    for a, b in qf_matches:
        h, a_goals, winner = simulate_knockout_score(a, b, extra_time=True)
        qf_winners.append(winner if winner == a.name else b if winner == b.name else (a if h > a_goals else b))

    if len(qf_winners) < 2:
        return [t.name for t in qf_winners]

    # Semis
    sf_matches = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, len(qf_winners) - 1, 2)]
    sf_winners: list[Team] = []
    for a, b in sf_matches:
        h, a_goals, winner = simulate_knockout_score(a, b, extra_time=True)
        sf_winners.append(winner if winner == a.name else b if winner == b.name else (a if h > a_goals else b))

    if len(sf_winners) < 2:
        return [t.name for t in sf_winners]

    # Final
    h, a_goals, winner = simulate_knockout_score(sf_winners[0], sf_winners[1], extra_time=True)
    champion = winner if winner == sf_winners[0].name else sf_winners[1] if winner == sf_winners[1].name else (sf_winners[0] if h > a_goals else sf_winners[1])
    runner_up = sf_winners[1] if champion.name == sf_winners[0].name else sf_winners[0]

    # Ranking
    results = [champion.name, runner_up.name]
    for t in sf_winners:
        if t.name not in results:
            results.append(t.name)
    for t in qf_winners:
        if t.name not in results:
            results.append(t.name)
    for t in r16_winners:
        if t.name not in results:
            results.append(t.name)

    return results


def simulate_tournament(
    teams: list[Team],
    groups: dict[str, list[Team]],
    num_simulations: int = 10000,
    league_avg: float = 2.5,
) -> dict[str, SimulationResult]:
    """Simula el torneo N veces y devuelve resultados agregados."""
    results_map: dict[str, SimulationResult] = {}
    for t in teams:
        results_map[t.name] = SimulationResult(
            team_name=t.name,
            team_espn_id=t.espn_id,
        )

    for _ in range(num_simulations):
        standings = simulate_single_tournament(teams, groups, league_avg)
        for team in teams:
            results_map[team.name].total_simulations += 1

        if standings:
            champion = standings[0]
            if champion in results_map:
                results_map[champion].win_tournament += 1
            for name in standings[:2]:
                if name in results_map:
                    results_map[name].reach_final += 1
            for name in standings[:4]:
                if name in results_map:
                    results_map[name].reach_semis += 1
            for name in standings[:8]:
                if name in results_map:
                    results_map[name].reach_quarters += 1

    return results_map


def build_world_cup_2026_group_stage(
    teams_data: list[dict[str, Any]],
    get_elo: Callable | None = None,
    max_teams: int = 32,
) -> tuple[list[Team], dict[str, list[Team]]]:
    """Build groups for a 32-team World Cup."""
    team_objects: list[Team] = []
    groups_dict: dict[str, list[Team]] = {}
    group_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

    limited = teams_data[:max_teams]
    for i, td in enumerate(limited):
        name = td.get("name", td.get("displayName", ""))
        espn_id = str(td.get("id", td.get("espnId", "")))
        raw_group = td.get("group", td.get("groupName", ""))
        if raw_group:
            group_name = raw_group
        else:
            group_name = group_letters[i // 4] if i // 4 < len(group_letters) else "A"

        elo_val = INITIAL_ELO
        if get_elo and espn_id:
            elo_val = get_elo(espn_id)
            if elo_val is None:
                elo_val = INITIAL_ELO

        t = Team(
            espn_id=espn_id,
            name=name,
            elo=elo_val,
            group=group_name,
            attacking=td.get("attacking", 1.0),
            defensive=td.get("defensive", 1.0),
            form_pts=td.get("form_pts", 0.5),
        )
        team_objects.append(t)
        if group_name not in groups_dict:
            groups_dict[group_name] = []
        groups_dict[group_name].append(t)

    return team_objects, groups_dict
