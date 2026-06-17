"""Pytest configuration and fixtures."""

import pytest
from django.test import Client
from rest_framework.test import APIClient


@pytest.fixture
def client() -> Client:
    """Standard Django client."""
    return Client()


@pytest.fixture
def api_client() -> APIClient:
    """DRF API client."""
    return APIClient()
