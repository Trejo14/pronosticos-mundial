"""Tests for predictions app."""
from __future__ import annotations

from django.test import TestCase

from apps.predictions.odds_analyzer import (
    american_to_decimal,
    calculate_edge,
    calculate_expected_value,
    calculate_kelly,
    calculate_risk_label,
    decimal_to_implied_prob,
    remove_vig,
)
from apps.predictions.feature_engineering import (
    expected_score,
    fifa_ranking_to_elo,
    update_elo,
)
from apps.predictions.prediction_engine import (
    TeamInfo,
    poisson_match_probabilities,
    predict_match,
)


class OddsAnalyzerTests(TestCase):
    def test_american_to_decimal_positive(self):
        self.assertAlmostEqual(american_to_decimal(150), 2.5)

    def test_american_to_decimal_negative(self):
        self.assertAlmostEqual(american_to_decimal(-150), 1.6667, places=4)

    def test_decimal_to_implied_prob(self):
        self.assertAlmostEqual(decimal_to_implied_prob(2.0), 0.5)

    def test_decimal_to_implied_prob_evens(self):
        self.assertAlmostEqual(decimal_to_implied_prob(1.91), 0.5236, places=3)

    def test_remove_vig_even_market(self):
        h, d, a = remove_vig(0.5, 0.0, 0.5)
        self.assertAlmostEqual(h, 0.5)
        self.assertAlmostEqual(a, 0.5)

    def test_remove_vig_with_vig(self):
        h, d, a = remove_vig(0.52, 0.0, 0.52)
        self.assertAlmostEqual(h, 0.5)
        self.assertAlmostEqual(a, 0.5)

    def test_remove_vig_three_way(self):
        h, d, a = remove_vig(0.35, 0.30, 0.40)
        total = h + d + a
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(h, 0.35 / 1.05, places=5)

    def test_calculate_edge_positive(self):
        edge = calculate_edge(0.6, 2.0)
        self.assertAlmostEqual(edge, 0.2)

    def test_calculate_edge_negative(self):
        edge = calculate_edge(0.4, 2.0)
        self.assertAlmostEqual(edge, -0.2)

    def test_calculate_edge_zero(self):
        edge = calculate_edge(0.5, 2.0)
        self.assertAlmostEqual(edge, 0.0)

    def test_expected_value_positive(self):
        ev = calculate_expected_value(0.6, 2.0)
        self.assertAlmostEqual(ev, 0.2)

    def test_expected_value_negative(self):
        ev = calculate_expected_value(0.4, 2.0)
        self.assertAlmostEqual(ev, -0.2)

    def test_expected_value_breakeven(self):
        ev = calculate_expected_value(0.5, 2.0)
        self.assertAlmostEqual(ev, 0.0)

    def test_kelly_positive(self):
        k = calculate_kelly(0.6, 2.0, bank_fraction=1.0)
        self.assertAlmostEqual(k, 0.2)
        k_capped = calculate_kelly(0.6, 2.0)
        self.assertAlmostEqual(k_capped, 0.05)

    def test_kelly_zero_edge(self):
        k = calculate_kelly(0.5, 2.0)
        self.assertEqual(k, 0.0)

    def test_kelly_negative_edge(self):
        k = calculate_kelly(0.4, 2.0)
        self.assertEqual(k, 0.0)

    def test_kelly_high_value(self):
        k = calculate_kelly(0.7, 2.0, bank_fraction=0.1)
        self.assertAlmostEqual(k, 0.1)

    def test_risk_label_low(self):
        label = calculate_risk_label(0.6, 0.2, 0.04)
        self.assertEqual(label, "low")

    def test_risk_label_medium(self):
        label = calculate_risk_label(0.5, 0.08, 0.02)
        self.assertEqual(label, "medium")

    def test_risk_label_high(self):
        label = calculate_risk_label(0.3, 0.02, 0.0)
        self.assertEqual(label, "high")

    def test_remove_vig_empty(self):
        h, d, a = remove_vig(0, 0, 0)
        self.assertEqual(h, 0)
        self.assertEqual(d, 0)
        self.assertEqual(a, 0)


class FeatureEngineeringTests(TestCase):
    def test_expected_score_equal(self):
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)

    def test_expected_score_stronger(self):
        self.assertGreater(expected_score(1600, 1400), 0.5)

    def test_expected_score_weaker(self):
        self.assertLess(expected_score(1400, 1600), 0.5)

    def test_expected_score_extreme(self):
        self.assertGreater(expected_score(2000, 1000), 0.99)

    def test_update_elo_winner(self):
        new_a, new_b = update_elo(1500, 1500, 1, 0)
        self.assertGreater(new_a, 1500)
        self.assertLess(new_b, 1500)

    def test_update_elo_draw(self):
        new_a, new_b = update_elo(1500, 1500, 0, 0)
        self.assertAlmostEqual(new_a, 1500)
        self.assertAlmostEqual(new_b, 1500)

    def test_update_elo_upset(self):
        new_a, new_b = update_elo(1400, 1600, 1, 0)
        self.assertGreater(new_a, 1400)
        self.assertGreater(new_a - 1400, new_b - 1600)

    def test_fifa_ranking_to_elo_top(self):
        elo = fifa_ranking_to_elo(1)
        self.assertAlmostEqual(elo, 2000.0)

    def test_fifa_ranking_to_elo_mid(self):
        elo = fifa_ranking_to_elo(106)
        self.assertAlmostEqual(elo, 1502.37, places=1)

    def test_fifa_ranking_to_elo_none(self):
        elo = fifa_ranking_to_elo(None)
        self.assertAlmostEqual(elo, 1500.0)

    def test_fifa_ranking_to_elo_invalid(self):
        elo = fifa_ranking_to_elo(-1)
        self.assertAlmostEqual(elo, 1500.0)

    def test_fifa_ranking_to_elo_last(self):
        elo = fifa_ranking_to_elo(211)
        self.assertAlmostEqual(elo, 1004.74, places=1)


class PredictionEngineTests(TestCase):
    def test_poisson_equal_teams(self):
        h, d, a = poisson_match_probabilities(1.25, 1.25)
        self.assertAlmostEqual(h + d + a, 1.0, places=5)
        self.assertAlmostEqual(h, a, places=2)

    def test_poisson_strong_home(self):
        h, d, a = poisson_match_probabilities(2.0, 0.8)
        self.assertGreater(h, a)
        self.assertGreater(h, d)

    def test_poisson_strong_away(self):
        h, d, a = poisson_match_probabilities(0.8, 2.0)
        self.assertGreater(a, h)

    def test_poisson_sum_to_one(self):
        for hg in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for ag in [0.5, 1.0, 1.5]:
                h, d, a = poisson_match_probabilities(hg, ag)
                self.assertAlmostEqual(h + d + a, 1.0, places=4)

    def test_predict_match_no_espn(self):
        home = TeamInfo(espn_id="1", name="Team A", abbreviation="A", elo=1600, attacking=1.2, defensive=0.8)
        away = TeamInfo(espn_id="2", name="Team B", abbreviation="B", elo=1400, attacking=0.9, defensive=1.1)
        result = predict_match(home, away)
        self.assertAlmostEqual(result.home_win + result.draw + result.away_win, 1.0, places=4)
        self.assertGreater(result.home_win, result.away_win)
        self.assertGreater(result.expected_goals_home, 0)
        self.assertGreater(result.expected_goals_away, 0)

    def test_predict_match_with_espn(self):
        home = TeamInfo(espn_id="1", name="Team A", abbreviation="A", elo=1500, attacking=1.0, defensive=1.0)
        away = TeamInfo(espn_id="2", name="Team B", abbreviation="B", elo=1500, attacking=1.0, defensive=1.0)
        result = predict_match(home, away, espn_win_probs=(0.4, 0.25, 0.35))
        self.assertAlmostEqual(result.home_win + result.draw + result.away_win, 1.0, places=4)

    def test_predict_match_confidence(self):
        home = TeamInfo(espn_id="1", name="Team A", abbreviation="A", elo=1500, attacking=1.0, defensive=1.0)
        away = TeamInfo(espn_id="2", name="Team B", abbreviation="B", elo=1500, attacking=1.0, defensive=1.0)
        result = predict_match(home, away, espn_win_probs=(0.4, 0.25, 0.35))
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_predict_match_extreme_diff(self):
        home = TeamInfo(espn_id="1", name="Strong", abbreviation="S", elo=2000, attacking=2.0, defensive=0.5)
        away = TeamInfo(espn_id="2", name="Weak", abbreviation="W", elo=1000, attacking=0.5, defensive=2.0)
        result = predict_match(home, away)
        self.assertGreater(result.home_win, 0.7)
        self.assertGreater(result.expected_goals_home, 2.0)


class TournamentSimulatorTests(TestCase):
    def test_tournament_simulation_runs(self):
        from apps.predictions.tournament_simulator import (
            Team,
            simulate_single_tournament,
        )

        teams_a = [
            Team(espn_id="1", name="Brazil", elo=1900, group="A"),
            Team(espn_id="2", name="Cameroon", elo=1400, group="A"),
            Team(espn_id="3", name="Serbia", elo=1500, group="A"),
            Team(espn_id="4", name="Switzerland", elo=1550, group="A"),
        ]
        teams_b = [
            Team(espn_id="5", name="Spain", elo=1850, group="B"),
            Team(espn_id="6", name="Japan", elo=1450, group="B"),
            Team(espn_id="7", name="Costa Rica", elo=1350, group="B"),
            Team(espn_id="8", name="Germany", elo=1800, group="B"),
        ]
        teams = teams_a + teams_b
        groups = {"A": teams_a, "B": teams_b}
        result = simulate_single_tournament(teams, groups)
        self.assertTrue(len(result) > 0)
        self.assertIn(result[0], ["Brazil", "Spain", "Germany"])

    def test_simulate_multiple(self):
        from apps.predictions.tournament_simulator import (
            Team,
            simulate_tournament,
        )

        teams = [
            Team(espn_id="1", name="Brazil", elo=1900, group="A"),
            Team(espn_id="2", name="Cameroon", elo=1400, group="A"),
            Team(espn_id="3", name="Serbia", elo=1500, group="A"),
            Team(espn_id="4", name="Switzerland", elo=1550, group="A"),
            Team(espn_id="5", name="Spain", elo=1850, group="B"),
            Team(espn_id="6", name="Japan", elo=1450, group="B"),
            Team(espn_id="7", name="Costa Rica", elo=1350, group="B"),
            Team(espn_id="8", name="Germany", elo=1800, group="B"),
        ]
        groups = {"A": teams[:4], "B": teams[4:]}
        results = simulate_tournament(teams, groups, num_simulations=100)
        brazil = next(r for r in results.values() if r.team_name == "Brazil")
        self.assertGreater(brazil.win_prob, 0)
        self.assertEqual(brazil.total_simulations, 100)

    def test_simulate_strongest_wins_most(self):
        from apps.predictions.tournament_simulator import (
            Team,
            simulate_tournament,
        )

        teams_a = [
            Team(espn_id="1", name="Brazil", elo=2000, group="A"),
            Team(espn_id="2", name="Japan", elo=1300, group="A"),
            Team(espn_id="3", name="Croatia", elo=1400, group="A"),
            Team(espn_id="4", name="Cameroon", elo=1250, group="A"),
        ]
        teams_b = [
            Team(espn_id="5", name="Spain", elo=1900, group="B"),
            Team(espn_id="6", name="Canada", elo=1200, group="B"),
            Team(espn_id="7", name="Panama", elo=1100, group="B"),
            Team(espn_id="8", name="Haiti", elo=1050, group="B"),
        ]
        teams = teams_a + teams_b
        groups = {"A": teams_a, "B": teams_b}
        results = simulate_tournament(teams, groups, num_simulations=100)
        brazil = results["Brazil"]
        self.assertGreater(brazil.win_tournament, 0)
        self.assertGreater(brazil.reach_final, 0)
