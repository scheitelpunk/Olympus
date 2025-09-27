"""Configuration management system for MORPHEUS.

Handles loading, validation, and management of configuration files
with support for environment variables and multiple configuration sources.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import logging
from pydantic import BaseModel, Field, validator
from pydantic.env_settings import BaseSettings

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    database: str = Field(default="morpheus", env="DATABASE_NAME")
    user: str = Field(default="morpheus_user", env="DATABASE_USER")
    password: str = Field(default="morpheus_pass", env="DATABASE_PASSWORD")
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=50)
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v


class TactileConfig(BaseModel):
    """Tactile perception configuration."""
    enabled: bool = True
    sensitivity: float = Field(default=0.01, gt=0, le=100)  # Minimum force in Newtons
    sampling_rate: int = Field(default=1000, gt=0, le=10000)  # Hz
    vibration_window: float = Field(default=0.1, gt=0, le=10)  # seconds
    use_materials: bool = True
    max_contact_points: int = Field(default=50, gt=0, le=1000)
    embedding_dim: int = Field(default=64, gt=0, le=512)
    
    @validator('sampling_rate')
    def validate_sampling_rate(cls, v):
        if v not in [100, 500, 1000, 2000, 5000, 10000]:
            logger.warning(f"Non-standard sampling rate: {v}Hz")
        return v


class AudioConfig(BaseModel):
    """Audio perception configuration."""
    enabled: bool = True
    max_sources: int = Field(default=10, gt=0, le=100)
    frequency_range: List[float] = Field(default=[20, 20000])
    speed_of_sound: float = Field(default=343.0, gt=0)  # m/s
    doppler_enabled: bool = True
    echo_detection: bool = True
    embedding_dim: int = Field(default=32, gt=0, le=256)
    spectral_bins: int = Field(default=64, gt=0, le=1024)
    
    @validator('frequency_range')
    def validate_frequency_range(cls, v):
        if len(v) != 2 or v[0] >= v[1] or v[0] < 0:
            raise ValueError('Frequency range must be [min, max] with min < max and min >= 0')
        return v


class VisualConfig(BaseModel):
    """Visual perception configuration."""
    enabled: bool = True
    use_gasm: bool = True  # Use GASM for visual features
    feature_dim: int = Field(default=128, gt=0, le=1024)
    depth_enabled: bool = True
    color_analysis: bool = True
    texture_analysis: bool = True
    

class PerceptionConfig(BaseModel):
    """Perception subsystem configuration."""
    tactile: TactileConfig = Field(default_factory=TactileConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    fusion_enabled: bool = True
    prediction_enabled: bool = True


class FusionNetworkConfig(BaseModel):
    """Fusion network configuration."""
    input_dims: Dict[str, int] = Field(default={
        "tactile": 64,
        "audio": 32,
        "visual": 128
    })
    hidden_dim: int = Field(default=256, gt=0, le=2048)
    output_dim: int = Field(default=128, gt=0, le=1024)
    num_heads: int = Field(default=8, gt=0, le=32)
    dropout: float = Field(default=0.1, ge=0, le=0.9)
    num_layers: int = Field(default=3, gt=0, le=10)


class PredictorNetworkConfig(BaseModel):
    """Predictor network configuration."""
    state_dim: int = Field(default=128, gt=0, le=1024)
    action_dim: int = Field(default=7, gt=0)  # 6DOF + gripper
    hidden_dim: int = Field(default=256, gt=0, le=2048)
    output_dim: int = Field(default=128, gt=0, le=1024)
    uncertainty_estimation: bool = True
    num_layers: int = Field(default=4, gt=0, le=10)
    dropout: float = Field(default=0.1, ge=0, le=0.9)


class NetworksConfig(BaseModel):
    """Neural networks configuration."""
    fusion: FusionNetworkConfig = Field(default_factory=FusionNetworkConfig)
    predictor: PredictorNetworkConfig = Field(default_factory=PredictorNetworkConfig)
    device: str = Field(default="cpu")  # "cuda" or "cpu"
    checkpoint_dir: str = Field(default="checkpoints")
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ["cpu", "cuda", "mps"]:
            logger.warning(f"Unknown device: {v}. Falling back to CPU.")
            return "cpu"
        return v


class DreamConfig(BaseModel):
    """Dream subsystem configuration."""
    enabled: bool = True
    auto_dream: bool = False  # Automatically dream when idle
    auto_dream_threshold: int = Field(default=1000, gt=0)  # Experiences before auto-dream
    replay_speed: float = Field(default=10.0, gt=0, le=1000)
    variation_factor: float = Field(default=0.2, ge=0, le=1)
    exploration_rate: float = Field(default=0.3, ge=0, le=1)
    consolidation_threshold: float = Field(default=0.8, ge=0, le=1)
    min_improvement: float = Field(default=0.1, ge=0, le=1)
    max_iterations: int = Field(default=1000, gt=0, le=100000)
    parallel_dreams: int = Field(default=4, gt=0, le=32)
    time_range_hours: float = Field(default=24.0, gt=0, le=8760)  # Max 1 year
    max_experiences: int = Field(default=5000, gt=0, le=1000000)


class GASMConfig(BaseModel):
    """GASM integration configuration."""
    enabled: bool = True
    roboting_path: str = Field(default="../GASM-Roboting")
    config_file: str = Field(default="assets/configs/simulation_params.yaml")
    physics_enabled: bool = True
    visual_features_enabled: bool = True
    
    @validator('roboting_path')
    def validate_gasm_path(cls, v):
        path = Path(v)
        if not path.exists():
            logger.warning(f"GASM-Robotics path does not exist: {v}")
        return str(path)


class PerformanceConfig(BaseModel):
    """Performance and optimization configuration."""
    batch_size: int = Field(default=32, gt=0, le=1024)
    max_memory_gb: float = Field(default=4.0, gt=0, le=128)
    cache_size: int = Field(default=1000, gt=0, le=100000)
    cleanup_interval_hours: float = Field(default=24.0, gt=0, le=8760)
    retention_days: int = Field(default=30, gt=0, le=3650)  # Max 10 years
    num_workers: int = Field(default=4, gt=0, le=64)
    prefetch_factor: int = Field(default=2, gt=0, le=10)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[str] = None
    max_size: str = Field(default="10MB")
    backup_count: int = Field(default=5, ge=0, le=100)
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class SystemConfig(BaseModel):
    """System-wide configuration."""
    mode: str = Field(default="full")  # "full", "perception_only", "dream_only"
    device: str = Field(default="cpu")  # "cuda" or "cpu"
    debug: bool = Field(default=False)
    profile: bool = Field(default=False)
    
    @validator('mode')
    def validate_mode(cls, v):
        valid_modes = ['full', 'perception_only', 'dream_only', 'evaluation']
        if v not in valid_modes:
            raise ValueError(f'System mode must be one of {valid_modes}')
        return v


class MorpheusConfig(BaseSettings):
    """Main MORPHEUS configuration class."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    networks: NetworksConfig = Field(default_factory=NetworksConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    gasm: GASMConfig = Field(default_factory=GASMConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        env_nested_delimiter = '__'
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        import logging.handlers
        
        # Create logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.logging.level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.logging.file:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    self.logging.file,
                    maxBytes=self._parse_size(self.logging.max_size),
                    backupCount=self.logging.backup_count
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []
        
        # Check GASM path
        gasm_path = Path(self.gasm.roboting_path)
        if not gasm_path.exists():
            issues.append(f"GASM-Robotics path not found: {gasm_path}")
        else:
            config_path = gasm_path / self.gasm.config_file
            if not config_path.exists():
                issues.append(f"GASM config file not found: {config_path}")
        
        # Check device availability
        if self.system.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("CUDA requested but not available")
            except ImportError:
                issues.append("PyTorch not installed for CUDA support")
        
        # Check memory limits
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if self.performance.max_memory_gb > available_memory * 0.8:
            issues.append(f"Requested memory ({self.performance.max_memory_gb}GB) exceeds 80% of available ({available_memory:.1f}GB)")
        
        # Check database connection (optional)
        if self.database.host != "localhost":
            import socket
            try:
                socket.create_connection((self.database.host, self.database.port), timeout=5)
            except Exception as e:
                issues.append(f"Cannot connect to database: {e}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        # Convert pydantic models to dictionaries
        def convert_to_serializable(obj):
            if isinstance(obj, BaseModel):
                return obj.dict()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_config = convert_to_serializable(config_dict)
        
        with open(file_path, 'w') as f:
            yaml.dump(serializable_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")


class ConfigManager:
    """Configuration manager for MORPHEUS system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[MorpheusConfig] = None
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, 
                   config_path: Optional[Union[str, Path]] = None,
                   validate: bool = True) -> MorpheusConfig:
        """Load configuration from file.
        
        Args:
            config_path: Optional path to configuration file
            validate: Whether to validate configuration
            
        Returns:
            Loaded configuration object
        """
        path = Path(config_path) if config_path else self.config_path
        
        if path and path.exists():
            logger.info(f"Loading configuration from {path}")
            
            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Merge with environment variables
            self._config = MorpheusConfig(**yaml_config)
        else:
            logger.info("Using default configuration")
            self._config = MorpheusConfig()
        
        # Setup logging
        self._config.setup_logging()
        
        # Validate if requested
        if validate:
            issues = self._config.validate_config()
            if issues:
                logger.warning(f"Configuration issues found: {issues}")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        
        logger.info(f"Configuration loaded successfully (mode: {self._config.system.mode})")
        return self._config
    
    def get_config(self) -> MorpheusConfig:
        """Get current configuration.
        
        Returns:
            Current configuration object
            
        Raises:
            RuntimeError: If no configuration is loaded
        """
        if self._config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        return self._config
    
    def create_default_config(self, 
                             config_path: Union[str, Path],
                             overrides: Optional[Dict[str, Any]] = None) -> MorpheusConfig:
        """Create default configuration file.
        
        Args:
            config_path: Path where to save the configuration
            overrides: Optional configuration overrides
            
        Returns:
            Created configuration object
        """
        # Start with default config
        config = MorpheusConfig()
        
        # Apply overrides if provided
        if overrides:
            config_dict = config.dict()
            config_dict.update(overrides)
            config = MorpheusConfig(**config_dict)
        
        # Save to file
        config.save_to_file(config_path)
        
        return config
    
    def reload_config(self) -> MorpheusConfig:
        """Reload configuration from file.
        
        Returns:
            Reloaded configuration object
        """
        logger.info("Reloading configuration")
        return self.load_config(self.config_path)
    
    def get_database_url(self) -> str:
        """Get database URL from configuration.
        
        Returns:
            PostgreSQL database URL
        """
        config = self.get_config()
        db = config.database
        return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.database}"
    
    def get_gasm_config_path(self) -> Path:
        """Get path to GASM configuration file.
        
        Returns:
            Path to GASM simulation_params.yaml
        """
        config = self.get_config()
        return Path(config.gasm.roboting_path) / config.gasm.config_file


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None) -> MorpheusConfig:
    """Load configuration using global manager."""
    return get_config_manager().load_config(config_path)


def get_config() -> MorpheusConfig:
    """Get current configuration using global manager."""
    return get_config_manager().get_config()