"""Monte Carlo tournament simulator for World Cup predictions.

Simulates the entire tournament bracket thousands of times to compute
probabilities for each team: win tournament, reach final, reach semis, etc.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

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


@dataclass
class Team:
    espn_id: str
    name: str
    elo: float = INITIAL_ELO
    group: str = ""
    attacking: float = 1.0
    defensive: float = 1.0

    @property
    def strength(self) -> float:
        return self.elo / 1000.0 * self.attacking / max(self.defensive, 0.1)


def match_winner(team_a: Team, team_b: Team, home_advantage: float = 0.0) -> Team:
    expected_a = expected_score(team_a.elo + home_advantage, team_b.elo)
    random_draw = random.random()
    if random_draw < expected_a:
        return team_a
    return team_b


def match_winner_with_draw(
    team_a: Team,
    team_b: Team,
    draw_probability: float = 0.25,
    home_advantage: float = 0.0,
) -> Team | None:
    expected_a = expected_score(team_a.elo + home_advantage, team_b.elo)
    adjusted_draw = draw_probability
    p_a = expected_a * (1.0 - adjusted_draw)
    p_b = (1.0 - expected_a) * (1.0 - adjusted_draw)
    p_draw = adjusted_draw
    total = p_a + p_draw + p_b
    p_a /= total
    p_draw /= total
    p_b /= total
    r = random.random()
    if r < p_a:
        return team_a
    if r < p_a + p_draw:
        return None
    return team_b


def simulate_group_stage(
    groups: dict[str, list[Team]],
    group_draw_prob: float = 0.28,
) -> dict[str, tuple[Team, Team]]:
    winners: dict[str, tuple[Team, Team]] = {}
    for group_name, teams in groups.items():
        if len(teams) < 2:
            continue
        points = {t.name: 0 for t in teams}
        played = set()
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                key = tuple(sorted([teams[i].name, teams[j].name]))
                if key in played:
                    continue
                played.add(key)
                winner = match_winner_with_draw(teams[i], teams[j], group_draw_prob)
                if winner is None:
                    points[teams[i].name] += 1
                    points[teams[j].name] += 1
                else:
                    points[winner.name] += 3
        sorted_teams = sorted(teams, key=lambda t: points[t.name], reverse=True)
        if len(sorted_teams) >= 2:
            winners[group_name] = (sorted_teams[0], sorted_teams[1])
        elif len(sorted_teams) == 1:
            winners[group_name] = (sorted_teams[0], sorted_teams[0])
    return winners


def simulate_knockout_round(
    matches: list[tuple[Team, Team]],
    draw_possible: bool = False,
    draw_prob: float = 0.25,
) -> list[Team]:
    winners: list[Team] = []
    for a, b in matches:
        if draw_possible:
            winner = match_winner_with_draw(a, b, draw_prob)
            if winner is None:
                winner = match_winner(a, b)
        else:
            winner = match_winner(a, b)
        winners.append(winner)
    return winners


def build_r16_matches(
    group_winners: dict[str, tuple[Team, Team]],
) -> list[tuple[Team, Team]]:
    """Build Round of 16 matchups from group winners.

    Standard World Cup format:
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
    group_draw_prob: float = 0.28,
    knockout_draw_prob: float = 0.25,
    quarter_draw_prob: float = 0.20,
    semi_draw_prob: float = 0.20,
) -> list[str]:
    group_winners = simulate_group_stage(groups, group_draw_prob)
    r16_matches = build_r16_matches(group_winners)
    if len(r16_matches) < 2:
        return []
    r16_winners = simulate_knockout_round(r16_matches, draw_possible=True, draw_prob=knockout_draw_prob)
    if len(r16_winners) < 4:
        return [t.name for t in r16_winners]
    qf_matches = [(r16_winners[i], r16_winners[i + 1]) for i in range(0, len(r16_winners) - 1, 2)]
    qf_winners = simulate_knockout_round(qf_matches, draw_possible=True, draw_prob=quarter_draw_prob)
    if len(qf_winners) < 2:
        return [t.name for t in qf_winners]
    sf_matches = [(qf_winners[i], qf_winners[i + 1]) for i in range(0, len(qf_winners) - 1, 2)]
    sf_winners = simulate_knockout_round(sf_matches, draw_possible=True, draw_prob=semi_draw_prob)
    if len(sf_winners) < 2:
        return [t.name for t in sf_winners]
    champion = match_winner(sf_winners[0], sf_winners[1])
    runner_up = sf_winners[1] if sf_winners[0].name == champion.name else sf_winners[0]
    results = [champion.name, runner_up.name]
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
    group_draw_prob: float = 0.28,
    knockout_draw_prob: float = 0.25,
    quarter_draw_prob: float = 0.20,
    semi_draw_prob: float = 0.20,
) -> dict[str, SimulationResult]:
    results_map: dict[str, SimulationResult] = {}
    for t in teams:
        results_map[t.name] = SimulationResult(
            team_name=t.name,
            team_espn_id=t.espn_id,
        )
    for _ in range(num_simulations):
        standings = simulate_single_tournament(
            teams, groups,
            group_draw_prob=group_draw_prob,
            knockout_draw_prob=knockout_draw_prob,
            quarter_draw_prob=quarter_draw_prob,
            semi_draw_prob=semi_draw_prob,
        )
        for team in teams:
            results_map[team.name].total_simulations += 1
        if standings:
            champion = standings[0]
            if champion in results_map:
                results_map[champion].win_tournament += 1
            for name in standings[:2]:
                if name in results_map:
                    results_map[name].reach_final += 1
            for name in standings[:min(4, len(standings))]:
                if name in results_map:
                    results_map[name].reach_semis += 1
            for name in standings[:min(8, len(standings))]:
                if name in results_map:
                    results_map[name].reach_quarters += 1
    return results_map


def build_world_cup_2026_group_stage(
    teams_data: list[dict[str, Any]],
    get_elo: callable | None = None,
    max_teams: int = 32,
) -> tuple[list[Team], dict[str, list[Team]], list[tuple[str, str]]]:
    """Build groups for a 32-team World Cup.
    
    Assigns teams to groups A-H (4 teams each) based on their order in the list,
    or by the 'group' field if provided.
    """
    team_objects: list[Team] = []
    groups_dict: dict[str, list[Team]] = {}
    group_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    teams_per_group = 4
    limited = teams_data[:max_teams]
    for i, td in enumerate(limited):
        name = td.get("name", td.get("displayName", ""))
        espn_id = str(td.get("id", td.get("espnId", "")))
        raw_group = td.get("group", td.get("groupName", ""))
        if raw_group:
            group_name = raw_group
        else:
            group_name = group_letters[i // teams_per_group] if i // teams_per_group < len(group_letters) else "A"
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
        )
        team_objects.append(t)
        if group_name not in groups_dict:
            groups_dict[group_name] = []
        groups_dict[group_name].append(t)
    return team_objects, groups_dict
