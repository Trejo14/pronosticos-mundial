"""Database models for predictions data."""
from django.db import models


class TimestampMixin(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class TeamRating(TimestampMixin):
    league = models.CharField(max_length=50, db_index=True)
    team_espn_id = models.CharField(max_length=50, db_index=True)
    team_name = models.CharField(max_length=100)
    team_abbreviation = models.CharField(max_length=10, blank=True)
    elo_rating = models.FloatField(default=1500.0)
    fifa_ranking = models.IntegerField(null=True, blank=True)
    attacking_strength = models.FloatField(default=1.0)
    defensive_strength = models.FloatField(default=1.0)
    home_advantage = models.FloatField(default=0.0)
    season = models.IntegerField()
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-elo_rating"]
        unique_together = [["league", "team_espn_id", "season"]]

    def __str__(self):
        return f"{self.team_name} ({self.elo_rating:.0f})"


class MatchPrediction(TimestampMixin):
    event_espn_id = models.CharField(max_length=50, unique=True, db_index=True)
    league = models.CharField(max_length=50, db_index=True)
    match_date = models.DateTimeField()
    home_team = models.CharField(max_length=100)
    home_team_id = models.CharField(max_length=50)
    away_team = models.CharField(max_length=100)
    away_team_id = models.CharField(max_length=50)

    home_win_prob = models.FloatField()
    draw_prob = models.FloatField()
    away_win_prob = models.FloatField()

    home_win_edge = models.FloatField(null=True, blank=True)
    draw_edge = models.FloatField(null=True, blank=True)
    away_win_edge = models.FloatField(null=True, blank=True)

    home_value = models.FloatField(null=True, blank=True)
    draw_value = models.FloatField(null=True, blank=True)
    away_value = models.FloatField(null=True, blank=True)

    home_kelly = models.FloatField(null=True, blank=True)
    draw_kelly = models.FloatField(null=True, blank=True)
    away_kelly = models.FloatField(null=True, blank=True)

    best_home_odds = models.FloatField(null=True, blank=True)
    best_draw_odds = models.FloatField(null=True, blank=True)
    best_away_odds = models.FloatField(null=True, blank=True)
    best_provider = models.CharField(max_length=50, blank=True)

    risk_score = models.FloatField(null=True, blank=True)
    risk_label = models.CharField(max_length=20, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    expected_goals_home = models.FloatField(null=True, blank=True)
    expected_goals_away = models.FloatField(null=True, blank=True)

    market_home_prob = models.FloatField(null=True, blank=True)
    market_draw_prob = models.FloatField(null=True, blank=True)
    market_away_prob = models.FloatField(null=True, blank=True)

    model_version = models.CharField(max_length=20, default="1.0")
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-match_date"]

    def __str__(self):
        return f"{self.home_team} vs {self.away_team}"

    @property
    def recommendation(self):
        outcomes = []
        if self.home_win_edge and self.home_win_edge > 0.05:
            outcomes.append(("Home Win", self.home_win_edge, self.home_kelly))
        if self.draw_edge and self.draw_edge > 0.05:
            outcomes.append(("Draw", self.draw_edge, self.draw_kelly))
        if self.away_win_edge and self.away_win_edge > 0.05:
            outcomes.append(("Away Win", self.away_win_edge, self.away_kelly))
        if not outcomes:
            return "No value bets detected"
        best = max(outcomes, key=lambda x: x[1])
        return f"Bet {best[0]} (edge: {best[1]:+.1%}, Kelly: {best[2]:.1%})"


class TournamentPrediction(TimestampMixin):
    league = models.CharField(max_length=50, db_index=True)
    season = models.IntegerField()
    team_espn_id = models.CharField(max_length=50, db_index=True)
    team_name = models.CharField(max_length=100)
    win_tournament_prob = models.FloatField()
    reach_final_prob = models.FloatField()
    reach_semis_prob = models.FloatField()
    reach_quarters_prob = models.FloatField(null=True, blank=True)
    win_tournament_edge = models.FloatField(null=True, blank=True)
    win_tournament_kelly = models.FloatField(null=True, blank=True)
    best_odds_to_win = models.FloatField(null=True, blank=True)
    simulations_run = models.IntegerField(default=0)
    model_version = models.CharField(max_length=20, default="1.0")
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-win_tournament_prob"]
        unique_together = [["league", "season", "team_espn_id"]]

    def __str__(self):
        return f"{self.team_name}: {self.win_tournament_prob:.1%} to win"
