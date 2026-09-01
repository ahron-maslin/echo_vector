"""The model stack is an optional extra, so its absence must be explained well.

These tests deliberately do not import torch: they simulate a base install by
blocking the import, which is the situation a user hits after a plain
`pip install echo_vector`.
"""

import builtins
import importlib
import sys

import pytest

from echovector.embeddings.factory import (
    MODEL_EXTRA,
    EmbeddingFactory,
    MissingBackendDependencyError,
    get_embedding_model,
)

MODEL_BACKENDS = ["clap", "wav2vec2", "hubert", "ast"]


@pytest.fixture
def without_model_stack(monkeypatch):
    """Make torch and transformers unimportable, as on a base install."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    # Backends are cached once resolved, and the modules may already be loaded
    # from another test; clear both so the import is genuinely re-attempted.
    monkeypatch.setattr(EmbeddingFactory, "_registry", {})
    for module in [m for m in list(sys.modules) if m.startswith(("torch", "transformers"))]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    for module in [m for m in list(sys.modules) if "echovector.embeddings.clap" in m]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    for module in [m for m in list(sys.modules) if "transformers_audio" in m]:
        monkeypatch.delitem(sys.modules, module, raising=False)


@pytest.mark.parametrize("backend", MODEL_BACKENDS)
def test_model_backends_explain_how_to_install_them(without_model_stack, backend: str) -> None:
    """A bare ModuleNotFoundError tells the user nothing about the fix."""
    with pytest.raises(MissingBackendDependencyError) as excinfo:
        get_embedding_model(backend)

    message = str(excinfo.value)
    assert f"pip install 'echo_vector[{MODEL_EXTRA}]'" in message
    assert backend in message
    # The CPU wheels are the difference between a 200MB and a 4GB download.
    assert "download.pytorch.org/whl/cpu" in message


def test_the_error_is_an_importerror(without_model_stack) -> None:
    """Callers catching ImportError should still catch this."""
    assert issubclass(MissingBackendDependencyError, ImportError)
    with pytest.raises(ImportError):
        get_embedding_model("clap")


def test_local_backend_works_without_the_model_stack(without_model_stack) -> None:
    """The base install has to be able to do something."""
    backend = get_embedding_model("local")
    assert backend.embedding_dim == 8
    assert "torch" not in sys.modules


def test_core_modules_import_without_the_model_stack(without_model_stack) -> None:
    """Indexing, search, and evaluation must not depend on the optional extra."""
    for name in (
        "echovector.core",
        "echovector.cli.main",
        "echovector.indexing.faiss_index",
        "echovector.search.spans",
        "echovector.evaluation.harness",
        "echovector.audio.extract",
    ):
        assert importlib.import_module(name) is not None


def test_unknown_backend_still_reports_as_a_value_error(without_model_stack) -> None:
    """A typo is a different problem from a missing dependency."""
    with pytest.raises(ValueError, match="Unknown embedding backend") as excinfo:
        get_embedding_model("nonexistent")
    assert not isinstance(excinfo.value, ImportError)
