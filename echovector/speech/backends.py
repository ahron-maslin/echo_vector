"""Real model backends for the speech path. Heavy imports stay inside the functions."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from echovector.speech.transcript import Word

DEFAULT_WHISPER = "small"
DEFAULT_EMBEDDER = "intfloat/multilingual-e5-large"

_IPA_STRIP = str.maketrans("", "", "ˈˌː͡ˑ'` ")


def transcribe(audio: Path, model: str = DEFAULT_WHISPER) -> list[Word]:
    """Word-level transcript via faster-whisper (int8, VAD-filtered)."""
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, compute_type="int8")
    segments, _ = whisper.transcribe(str(audio), word_timestamps=True, vad_filter=True)
    return [Word(w.word.strip(), w.start, w.end) for seg in segments for w in seg.words or []]


class FastEmbedder:
    """Multilingual text embedder returning L2-normalised rows; e5 models need role prefixes."""

    def __init__(self, model: str = DEFAULT_EMBEDDER) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self._prefixed = "e5" in model

    def __call__(self, texts: list[str], *, query: bool) -> npt.NDArray[np.float32]:
        """Embed texts as queries or passages."""
        prefix = ("query: " if query else "passage: ") if self._prefixed else ""
        embedded = self._model.embed([prefix + t for t in texts])
        vectors = np.asarray(list(embedded), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised: npt.NDArray[np.float32] = vectors / norms
        return normalised


def espeak_ipa(surfaces: list[str], language: str = "en-us") -> dict[str, str]:
    """Map surface forms to stress-free IPA using the bundled espeak-ng."""
    import espeakng_loader
    from phonemizer import phonemize
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    out = phonemize(
        surfaces,
        language=language,
        backend="espeak",
        strip=True,
        with_stress=False,
        language_switch="remove-flags",
        preserve_punctuation=False,
        njobs=4,
    )
    return {s: p.strip().translate(_IPA_STRIP) for s, p in zip(surfaces, out, strict=True)}
