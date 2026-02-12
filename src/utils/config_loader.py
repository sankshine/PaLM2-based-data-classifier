"""
Configuration Loader Utility

Loads configuration from YAML file and environment variables.
"""

import yaml
import os
from typing import Any, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._merge_env_vars()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def _merge_env_vars(self):
        """Override config with environment variables"""
        # GCP settings
        if os.getenv('GCP_PROJECT_ID'):
            self.config.setdefault('gcp', {})['project_id'] = os.getenv('GCP_PROJECT_ID')
        if os.getenv('GCP_REGION'):
            self.config.setdefault('gcp', {})['region'] = os.getenv('GCP_REGION')
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            self.config.setdefault('gcp', {})['credentials_path'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # KMS key
        if os.getenv('KMS_KEY_NAME'):
            self.config.setdefault('dlp', {})['kms_key_name'] = os.getenv('KMS_KEY_NAME')
        
        logger.debug("Merged environment variables into config")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Args:
            key_path: Dot-separated path (e.g., 'gcp.project_id')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name
            
        Returns:
            Configuration dictionary
        """
        return self.config.get(section, {})
    
    def validate(self) -> bool:
        """
        Validate required configuration fields
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'gcp.project_id',
            'gcp.region',
            'vertex_ai.model_name'
        ]
        
        missing = []
        for field in required_fields:
            if self.get(field) is None:
                missing.append(field)
        
        if missing:
            logger.error(f"Missing required config fields: {missing}")
            return False
        
        logger.info("Configuration validation passed")
        return True


# Global config instance
_config = None

def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to config file
        
    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config
