"""
Configuration management for EchoVector.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

class Config:
    """Configuration class to manage settings for EchoVector."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the Config object.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
        """
        self._config = config_dict or {}
        
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            Config instance populated with data from the JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data)
        
    def to_json(self, file_path: Union[str, Path]) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            file_path: Path where the JSON configuration will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=4)
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if key is not found.
            
        Returns:
            The value for the specified key or the default value.
        """
        return self._config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        self._config[key] = value

    def update(self, other_config: Dict[str, Any]) -> None:
        """
        Update configuration with another dictionary.
        
        Args:
            other_config: Dictionary to update current configuration with.
        """
        self._config.update(other_config)
