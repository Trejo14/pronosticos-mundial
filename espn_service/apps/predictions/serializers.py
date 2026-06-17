"""Serializers for predictions API endpoints."""
from rest_framework import serializers


class MatchPredictionRequestSerializer(serializers.Serializer):
    league = serializers.CharField(default="fifa.world", required=False)
    event_id = serializers.CharField(required=False)
    sport = serializers.CharField(default="soccer", required=False)


class TournamentPredictionRequestSerializer(serializers.Serializer):
    league = serializers.CharField(default="fifa.world", required=False)
    simulations = serializers.IntegerField(default=10000, min_value=100, max_value=100000, required=False)


class UpcomingMatchesRequestSerializer(serializers.Serializer):
    league = serializers.CharField(default="fifa.world", required=False)
    days_ahead = serializers.IntegerField(default=7, min_value=1, max_value=30, required=False)


class TeamStrengthSerializer(serializers.Serializer):
    home = serializers.FloatField()
    away = serializers.FloatField()


class ExpectedGoalsSerializer(serializers.Serializer):
    home = serializers.FloatField()
    away = serializers.FloatField()


class OutcomeDetailSerializer(serializers.Serializer):
    probability = serializers.FloatField()
    expected_odds = serializers.FloatField(allow_null=True)
    best_odds = serializers.FloatField(required=False, allow_null=True)
    implied_prob = serializers.FloatField(required=False, allow_null=True)
    edge = serializers.FloatField(required=False, allow_null=True)
    expected_value = serializers.FloatField(required=False, allow_null=True)
    kelly = serializers.FloatField(required=False, allow_null=True)
    risk = serializers.CharField(required=False, allow_null=True)


class PredictionsSerializer(serializers.Serializer):
    home_win = OutcomeDetailSerializer()
    draw = OutcomeDetailSerializer()
    away_win = OutcomeDetailSerializer()


class RecommendationSerializer(serializers.Serializer):
    action = serializers.CharField()
    outcome = serializers.CharField(required=False, allow_null=True)
    edge = serializers.FloatField(required=False, allow_null=True)
    kelly_fraction = serializers.FloatField(required=False, allow_null=True)
    message = serializers.CharField()


class RiskSerializer(serializers.Serializer):
    score = serializers.FloatField()
    label = serializers.CharField()


class MatchPredictionResponseSerializer(serializers.Serializer):
    match = serializers.CharField()
    home_team = serializers.CharField()
    away_team = serializers.CharField()
    home_team_id = serializers.CharField()
    away_team_id = serializers.CharField()
    predictions = PredictionsSerializer()
    expected_goals = ExpectedGoalsSerializer()
    team_strength = TeamStrengthSerializer()
    risk = RiskSerializer()
    confidence = serializers.FloatField()
    recommendation = RecommendationSerializer()
    event_id = serializers.CharField(required=False, allow_null=True)
    match_date = serializers.DateTimeField(required=False, allow_null=True)
    market_probabilities = serializers.DictField(required=False, allow_null=True)

    class Meta:
        ref_name = "MatchPrediction"


class TournamentPredictionItemSerializer(serializers.Serializer):
    team = serializers.CharField()
    team_id = serializers.CharField()
    win_probability = serializers.FloatField()
    reach_final_probability = serializers.FloatField()
    reach_semis_probability = serializers.FloatField()
    reach_quarters_probability = serializers.FloatField()


class TournamentPredictionResponseSerializer(serializers.Serializer):
    simulations = serializers.IntegerField()
    predictions = TournamentPredictionItemSerializer(many=True)


class ValueBetSerializer(serializers.Serializer):
    match = serializers.CharField()
    match_date = serializers.DateTimeField()
    event_id = serializers.CharField()
    outcome = serializers.CharField()
    our_probability = serializers.FloatField()
    best_odds = serializers.FloatField()
    implied_probability = serializers.FloatField()
    edge = serializers.FloatField()
    expected_value = serializers.FloatField()
    kelly_fraction = serializers.FloatField()
    risk_label = serializers.CharField()


class ValueBetsResponseSerializer(serializers.Serializer):
    count = serializers.IntegerField()
    results = ValueBetSerializer(many=True)
