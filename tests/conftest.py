"""
Shared pytest fixtures for ArithmosDB tests.
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_vectors(rng):
    return rng.random((500, 32), dtype=np.float32)


@pytest.fixture
def medium_vectors(rng):
    return rng.random((10_000, 128), dtype=np.float32)


@pytest.fixture
def queries(rng):
    return rng.random((100, 128), dtype=np.float32)
