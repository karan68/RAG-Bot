"""
Configuration Manager
Loads and provides access to centralized configuration.
"""

import json
from pathlib import Path
from typing import Any, Optional


class Config:
    """Centralized configuration manager."""
    
    _instance: Optional['Config'] = None
    _config: dict = {}
    
    def __new__(cls):
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from config.json."""
        # Find config file relative to this file or workspace root
        possible_paths = [
            Path(__file__).parent.parent / 'config.json',  # slm/config.json
            Path.cwd() / 'config.json',  # Current directory
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            print("Warning: config.json not found, using defaults")
            self._config = self._get_defaults()
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
        
        # Store base path for resolving relative paths
        self._base_path = config_path.parent
    
    def _get_defaults(self) -> dict:
        """Return default configuration."""
        return {
            "app": {
                "name": "Device Specification Chatbot",
                "version": "1.0.0",
                "server_host": "127.0.0.1",
                "server_port": 7860
            },
            "models": {
                "llm": "qwen2.5:1.5b",
                "embedding": "nomic-embed-text"
            },
            "llm_settings": {
                "temperature": 0.2,
                "max_tokens": 200,
                "top_p": 0.9,
                "stop_sequences": ["User Question:", "User:", "Question:"]
            },
            "paths": {
                "data_dir": "data",
                "chroma_db": "data/chroma_db",
                "processed_devices": "data/processed_devices.json",
                "inference_rules": "inference_rules.json",
                "test_devices": "test_devices"
            },
            "validation": {
                "max_query_length": 500,
                "min_query_length": 2,
                "blocked_patterns": []
            },
            "retrieval": {
                "top_k": 3,
                "min_similarity": 0.3
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        Example: config.get('models.llm') returns 'qwen2.5:1.5b'
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """Get a path configuration, resolved to absolute path."""
        relative_path = self.get(f'paths.{key}')
        if relative_path is None:
            raise KeyError(f"Path '{key}' not found in config")
        
        return self._base_path / relative_path
    
    @property
    def base_path(self) -> Path:
        """Get the base path (where config.json is located)."""
        return self._base_path
    
    @property
    def llm_model(self) -> str:
        """Get the LLM model name."""
        return self.get('models.llm', 'qwen2.5:1.5b')
    
    @property
    def embedding_model(self) -> str:
        """Get the embedding model name."""
        return self.get('models.embedding', 'nomic-embed-text')
    
    @property
    def server_host(self) -> str:
        """Get server host."""
        return self.get('app.server_host', '127.0.0.1')
    
    @property
    def server_port(self) -> int:
        """Get server port."""
        return self.get('app.server_port', 7860)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return config
