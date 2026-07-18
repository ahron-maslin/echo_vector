"""Unit tests for echovector.utils (Config, logging) and evaluation.metrics."""

import logging

import numpy as np
import pytest

from echovector.evaluation.metrics import cosine_similarity, euclidean_distance
from echovector.utils.config import Config
from echovector.utils.logging import setup_logger


def test_config_get_set_default():
    config = Config()
    assert config.get("missing") is None
    assert config.get("missing", "fallback") == "fallback"

    config.set("key", "value")
    assert config.get("key") == "value"


def test_config_update():
    config = Config({"a": 1})
    config.update({"a": 2, "b": 3})
    assert config.get("a") == 2
    assert config.get("b") == 3


def test_config_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    config = Config({"a": 1, "b": "two"})
    config.to_json(path)

    loaded = Config.from_json(path)
    assert loaded.get("a") == 1
    assert loaded.get("b") == "two"


def test_setup_logger_returns_configured_logger():
    logger = setup_logger("echovector-test", level=logging.DEBUG)
    assert logger.name == "echovector-test"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1

    # Calling again must not add duplicate handlers.
    same_logger = setup_logger("echovector-test", level=logging.DEBUG)
    assert same_logger is logger
    assert len(same_logger.handlers) == 1


def test_cosine_similarity_identical_vectors():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    vec1 = np.array([1.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    vec1 = np.zeros(3, dtype=np.float32)
    vec2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(vec1, vec2) == 0.0


def test_cosine_similarity_shape_mismatch():
    vec1 = np.zeros(3, dtype=np.float32)
    vec2 = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        cosine_similarity(vec1, vec2)


def test_euclidean_distance():
    vec1 = np.array([0.0, 0.0], dtype=np.float32)
    vec2 = np.array([3.0, 4.0], dtype=np.float32)
    assert euclidean_distance(vec1, vec2) == pytest.approx(5.0)


def test_euclidean_distance_shape_mismatch():
    vec1 = np.zeros(3, dtype=np.float32)
    vec2 = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        euclidean_distance(vec1, vec2)
