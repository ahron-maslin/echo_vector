import os

import numpy as np
import soundfile as sf

from echovector import EchoVector
from echovector.utils.logging import setup_logger

logger = setup_logger("demo")

def generate_synthetic_wav(
    filepath: str,
    duration: float,
    sr: int = 16000,
    freq: float = 440.0,
) -> None:
    """Generates a synthetic sine wave and saves it to a wav file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple tone
    data = 0.5 * np.sin(2 * np.pi * freq * t)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, data, sr)
    logger.info(f"Generated synthetic audio at {filepath}")

def main() -> None:
    """Run an end-to-end local EchoVector demo."""
    print("=== EchoVector End-to-End Demo ===")

    # 1. Setup temporary audio directory and build index
    audio_dir = "./demo_audio"
    index_dir = "./demo_index"

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # Generate a few synthetic audio clips
    # Clip 1: A short alert/sound (high frequency)
    generate_synthetic_wav(f"{audio_dir}/alert.wav", duration=5.0, freq=880.0)
    # Clip 2: A low tone (bass discussion)
    generate_synthetic_wav(f"{audio_dir}/bass_talk.wav", duration=10.0, freq=110.0)

    # 2. Instantiate EchoVector
    print("\n--- Initializing EchoVector ---")
    # Use the local backend so the demo runs without GPU access or model downloads.
    ev = EchoVector(store_dir=index_dir, backend="local")

    # 3. Index the synthetic audio directory
    print("\n--- Indexing Audio ---")
    ev.index(audio_dir)

    # 4. Search across indexed audio
    print("\n--- Searching the Index ---")
    # Since CLAP maps descriptive text queries to audio features, we query semantically
    query = "high frequency alarm tone"
    print(f"Searching for: '{query}'")
    results = ev.search(query, top_k=2)

    print("\nSearch Results:")
    for i, r in enumerate(results, 1):
        print(
            f"{i}. File: {r.filepath} | "
            f"Start: {r.timestamp_range.start:.2f}s | "
            f"End: {r.timestamp_range.end:.2f}s | "
            f"Score: {r.score:.4f}"
        )

    # 5. Index statistics
    print("\n--- Index Statistics ---")
    stats = ev.stats()
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
