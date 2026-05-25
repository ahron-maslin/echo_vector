"""
Core functionalities for EchoVector.
"""
import numpy as np
import numpy.typing as npt
from typing import Dict, Optional

from echovector.utils import logger, Config

class AudioProcessor:
    """Core class for processing audio signals into vectors."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the AudioProcessor.
        
        Args:
            config: Configuration object containing processing parameters.
        """
        self.config = config or Config({"sample_rate": 16000, "n_mels": 80})
        logger.info(f"Initialized AudioProcessor with sample rate {self.config.get('sample_rate')}")
        
    def process(self, audio_data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Process raw audio data into a feature vector (e.g., embeddings).
        
        Args:
            audio_data: 1D or 2D numpy array representing the audio waveform.
            
        Returns:
            Processed feature vector as a numpy array.
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("Audio data must be a numpy array.")
            
        if len(audio_data.shape) > 2:
            raise ValueError("Audio data must be 1D or 2D (mono or stereo).")
            
        logger.debug(f"Processing audio data of shape {audio_data.shape}")
        
        # Placeholder for actual processing logic
        # Generating a generic embedding by averaging across the time axis (for stereo, average per channel)
        if len(audio_data.shape) == 2:
            features = np.mean(audio_data, axis=0)
        else:
            features = np.array([np.mean(audio_data)])
            
        return features.astype(np.float32)

class VectorDatabase:
    """Core class for storing and retrieving audio vector embeddings."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the VectorDatabase.
        
        Args:
            config: Configuration object.
        """
        self.config = config or Config()
        self.vectors: Dict[str, npt.NDArray[np.float32]] = {}
        logger.info("Initialized VectorDatabase")
        
    def add(self, vector_id: str, vector: npt.NDArray[np.float32]) -> None:
        """
        Add a vector to the database.
        
        Args:
            vector_id: Unique identifier for the vector.
            vector: The vector data.
        """
        self.vectors[vector_id] = vector
        logger.debug(f"Added vector '{vector_id}' to database.")
        
    def get(self, vector_id: str) -> Optional[npt.NDArray[np.float32]]:
        """
        Retrieve a vector by its ID.
        
        Args:
            vector_id: Unique identifier for the vector.
            
        Returns:
            The vector data if found, else None.
        """
        return self.vectors.get(vector_id)
