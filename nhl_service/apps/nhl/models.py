"""Database models for NHL functionality."""

from django.db import models


class TimestampMixin(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Team(TimestampMixin):
    """NHL Team."""
    team_id = models.CharField(max_length=50, unique=True, db_index=True)
    abbreviation = models.CharField(max_length=10)
    name = models.CharField(max_length=100)
    full_name = models.CharField(max_length=100)
    franchise_id = models.CharField(max_length=50, blank=True)
    is_active = models.BooleanField(default=True)
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.full_name} ({self.abbreviation})"


class Player(TimestampMixin):
    """NHL Player / Skater / Goalie."""
    player_id = models.CharField(max_length=50, unique=True, db_index=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    full_name = models.CharField(max_length=100)
    sweater_number = models.CharField(max_length=10, blank=True)
    position = models.CharField(max_length=10, blank=True)
    current_team = models.ForeignKey(
        Team, on_delete=models.SET_NULL, null=True, blank=True, related_name="players"
    )
    is_active = models.BooleanField(default=True)
    headshot_url = models.URLField(max_length=500, blank=True)
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["last_name", "first_name"]

    def __str__(self) -> str:
        return f"{self.full_name} #{self.sweater_number}"


class Game(TimestampMixin):
    """NHL Game / Event."""
    game_id = models.CharField(max_length=50, unique=True, db_index=True)
    season = models.CharField(max_length=20, db_index=True)  # e.g. "20232024"
    game_type = models.PositiveSmallIntegerField(default=2)  # 1=Pre, 2=Reg, 3=Playoff
    date = models.DateTimeField(db_index=True)
    home_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="home_games")
    away_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="away_games")
    home_score = models.PositiveSmallIntegerField(null=True, blank=True)
    away_score = models.PositiveSmallIntegerField(null=True, blank=True)
    status = models.CharField(max_length=50, default="scheduled")
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-date"]

    def __str__(self) -> str:
        return f"{self.away_team.abbreviation} @ {self.home_team.abbreviation} ({self.date.date()})"


class Standing(TimestampMixin):
    """Team Season Standing."""
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="standings")
    date = models.DateField(db_index=True)
    games_played = models.PositiveSmallIntegerField(default=0)
    wins = models.PositiveSmallIntegerField(default=0)
    losses = models.PositiveSmallIntegerField(default=0)
    ot_losses = models.PositiveSmallIntegerField(default=0)
    points = models.PositiveSmallIntegerField(default=0)
    point_pct = models.FloatField(null=True, blank=True)
    division_name = models.CharField(max_length=100, blank=True)
    conference_name = models.CharField(max_length=100, blank=True)
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-date", "-points"]
        unique_together = [["team", "date"]]

    def __str__(self) -> str:
        return f"{self.team.abbreviation} - {self.points} pts ({self.date})"


class SkaterSeasonStats(TimestampMixin):
    """Skater season stats (from Stats REST API)."""
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="skater_stats")
    season = models.CharField(max_length=20, db_index=True)
    games_played = models.PositiveSmallIntegerField(default=0)
    goals = models.PositiveSmallIntegerField(default=0)
    assists = models.PositiveSmallIntegerField(default=0)
    points = models.PositiveSmallIntegerField(default=0)
    plus_minus = models.SmallIntegerField(default=0)
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-season", "-points"]
        unique_together = [["player", "season"]]

    def __str__(self) -> str:
        return f"{self.player.full_name} ({self.season}): {self.points} pts"


class GoalieSeasonStats(TimestampMixin):
    """Goalie season stats (from Stats REST API)."""
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="goalie_stats")
    season = models.CharField(max_length=20, db_index=True)
    games_played = models.PositiveSmallIntegerField(default=0)
    wins = models.PositiveSmallIntegerField(default=0)
    losses = models.PositiveSmallIntegerField(default=0)
    save_pct = models.FloatField(null=True, blank=True)
    gaa = models.FloatField(null=True, blank=True)
    shutouts = models.PositiveSmallIntegerField(default=0)
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-season", "-wins"]
        unique_together = [["player", "season"]]

    def __str__(self) -> str:
        return f"{self.player.full_name} ({self.season}): {self.wins} W"
