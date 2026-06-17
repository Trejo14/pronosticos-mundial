# ruff: noqa: F401, F403
"""Test settings — isolated SQLite, no external services."""

from .base import *

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}

CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Speed up password hashing in tests
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

NHL_API_WEB_BASE_URL = "https://api-web.nhle.com"
NHL_API_STATS_BASE_URL = "https://api.nhle.com/stats/rest"
