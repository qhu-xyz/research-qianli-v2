"""Shared test fixtures."""
import pytest


@pytest.fixture
def sample_py():
    return "2025-06"


@pytest.fixture
def sample_quarter():
    return "aq1"


@pytest.fixture
def holdout_py():
    return "2025-06"
