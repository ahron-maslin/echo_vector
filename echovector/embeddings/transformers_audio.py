"""Audio-only embedding backends built on transformers audio encoders.

These models have no text tower, so they cannot answer a text query. What they
can do is place audio in a space where similar-sounding clips sit close
together, which is exactly what :meth:`echovector.EchoVector.search_by_audio`
needs. Reach for them when the question is "find more that sounds like this"
rather than "find the part where they discuss pricing".
"""

from typing import Any, ClassVar, cast

import librosa
import numpy as np
import numpy.typing as npt
import torch

from echovector.embeddings.base import EmbeddingBackend, TextSearchUnsupportedError

__all__ = [
    "ASTBackend",
    "HubertBackend",
    "TextSearchUnsupportedError",
    "TransformersAudioBackend",
    "Wav2Vec2Backend",
]


class TransformersAudioBackend(EmbeddingBackend):
    """Mean-pooled hidden states from a transformers audio encoder.

    Subclasses name the checkpoint and the feature-extractor/model classes to
    load. Pooling the encoder's final hidden states over time turns a
    variable-length clip into one fixed-width vector, which is what the index
    stores.
    """

    #: Short name recorded in the index manifest, e.g. "wav2vec2".
    family: ClassVar[str] = "transformers-audio"
    #: Checkpoint used when the caller does not name one.
    default_model: ClassVar[str] = ""
    #: Rate these encoders are trained on; input is resampled to it.
    target_sample_rate: ClassVar[int] = 16_000

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Initialize the backend.

        Args:
            model_name: Hugging Face checkpoint. Defaults to the family's.
            device: Device to run on (e.g. 'cpu', 'cuda'). Auto-detected if None.
        """
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = model_name or self.default_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor: Any = AutoFeatureExtractor.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_name
        )
        self.model: Any = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        """Return the encoder's hidden size."""
        return int(self.model.config.hidden_size)

    @property
    def fingerprint(self) -> str:
        """Return the identity of the exact weights backing this backend."""
        return f"{self.family}:{self.model_name}"

    def embed_audio(self, audio_paths: list[str]) -> npt.NDArray[np.float32]:
        """Embed a batch of audio files.

        Args:
            audio_paths: Paths to audio files.

        Returns:
            An array of shape (len(audio_paths), embedding_dim).
        """
        if not audio_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        waveforms = [
            cast(
                "npt.NDArray[np.float32]",
                librosa.load(path, sr=self.target_sample_rate, mono=True)[0],
            )
            for path in audio_paths
        ]
        return self._encode(waveforms)

    def embed_audio_arrays(
        self,
        waveforms: list[npt.NDArray[np.float32]],
        sample_rate: int,
    ) -> npt.NDArray[np.float32]:
        """Embed in-memory mono waveforms without a temp-file round trip.

        Args:
            waveforms: Mono float32 arrays.
            sample_rate: Sample rate shared by every waveform in the batch.

        Returns:
            An array of shape (len(waveforms), embedding_dim).
        """
        if not waveforms:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if sample_rate != self.target_sample_rate:
            waveforms = [
                cast(
                    "npt.NDArray[np.float32]",
                    librosa.resample(y=w, orig_sr=sample_rate, target_sr=self.target_sample_rate),
                )
                for w in waveforms
            ]
        return self._encode(waveforms)

    def embed_text(self, texts: list[str]) -> npt.NDArray[np.float32]:  # noqa: ARG002
        """Reject text queries.

        Args:
            texts: Ignored.

        Raises:
            TextSearchUnsupportedError: Always; this encoder has no text tower.
        """
        raise TextSearchUnsupportedError(
            f"The '{self.family}' backend embeds audio only and cannot answer a text query. "
            f"Use it with search_by_audio(), or index with the 'clap' backend for text search."
        )

    def _encode(self, waveforms: list[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
            pooled = self._pool(hidden, inputs.get("attention_mask"))
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return cast("npt.NDArray[np.float32]", pooled.cpu().numpy().astype(np.float32))

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Average encoder frames, ignoring padding.

        Padded frames carry no signal, so including them would pull short clips
        in a batch toward whatever the model emits for silence.
        """
        if attention_mask is None or attention_mask.dim() != 2:
            return hidden.mean(dim=1)

        # The encoder downsamples time, so the mask is rescaled to frame count.
        frames = hidden.shape[1]
        mask = torch.nn.functional.interpolate(
            attention_mask.unsqueeze(1).float(), size=frames, mode="nearest"
        ).squeeze(1)
        weights = mask.unsqueeze(-1)
        total = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / total


class Wav2Vec2Backend(TransformersAudioBackend):
    """Audio-only embeddings from a wav2vec2 encoder."""

    family: ClassVar[str] = "wav2vec2"
    default_model: ClassVar[str] = "facebook/wav2vec2-base"


class HubertBackend(TransformersAudioBackend):
    """Audio-only embeddings from a HuBERT encoder."""

    family: ClassVar[str] = "hubert"
    default_model: ClassVar[str] = "facebook/hubert-base-ls960"


class ASTBackend(TransformersAudioBackend):
    """Audio-only embeddings from an Audio Spectrogram Transformer."""

    family: ClassVar[str] = "ast"
    default_model: ClassVar[str] = "MIT/ast-finetuned-audioset-10-10-0.4593"

    @property
    def max_audio_seconds(self) -> float:
        """Return the fixed window AST accepts.

        AST pads or crops every input to a fixed spectrogram length, so a
        longer chunk would be truncated and the vector would not describe the
        timestamps stored beside it.
        """
        max_length: Any = getattr(self.feature_extractor, "max_length", None)
        try:
            # Spectrogram frames at a 10ms hop.
            return float(max_length) / 100.0
        except (TypeError, ValueError):
            return 10.24
