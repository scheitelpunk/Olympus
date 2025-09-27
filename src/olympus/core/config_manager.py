"""
Configuration Manager - System Configuration Management
======================================================

Manages all configuration for the OLYMPUS system including:
- System-wide settings
- Module-specific configurations
- Security policies
- Runtime parameters
- Environment-specific configurations

Key Responsibilities:
- Configuration loading and validation
- Hot-reload capabilities
- Configuration versioning and rollback
- Secure configuration storage
- Environment variable integration
- Configuration change notification
"""

import asyncio
import logging
import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading
from collections import defaultdict
import hashlib
import copy


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"


class ConfigScope(Enum):
    """Configuration scope levels"""
    SYSTEM = "system"
    MODULE = "module"
    USER = "user"
    RUNTIME = "runtime"


@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    scope: ConfigScope
    required_fields: List[str] = field(default_factory=list)
    optional_fields: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigChange:
    """Represents a configuration change"""
    path: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str
    applied: bool = False


@dataclass
class ConfigVersion:
    """Configuration version information"""
    version: str
    timestamp: datetime
    changes: List[ConfigChange]
    checksum: str
    description: Optional[str] = None


class ConfigManager:
    """
    Manages system configuration for OLYMPUS
    
    Provides centralized configuration management with validation,
    versioning, hot-reload, and secure storage capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.versions: List[ConfigVersion] = []
        self.change_listeners: Dict[str, List[Callable]] = defaultdict(list)
        
        # File paths
        self.config_path = Path(config_path) if config_path else Path("config/olympus.yaml")
        self.config_dir = self.config_path.parent
        self.schema_dir = self.config_dir / "schemas"
        self.backup_dir = self.config_dir / "backups"
        
        # Configuration state
        self.current_version = "1.0.0"
        self.last_loaded = None
        self.last_modified = None
        self.auto_reload = True
        self.reload_interval = 5.0  # seconds
        
        # Validation settings
        self.strict_validation = True
        self.allow_unknown_keys = False
        self.require_schema = False
        
        # Security settings
        self.encrypt_sensitive = True
        self.sensitive_keys = {
            'password', 'secret', 'token', 'key', 'credential',
            'auth', 'cert', 'private', 'api_key'
        }
        
        # Internal state
        self._file_watchers: List[asyncio.Task] = []
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        self.logger.info("Configuration Manager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the Configuration Manager
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Configuration Manager...")
            
            # Create directories
            await self._ensure_directories()
            
            # Load configuration schemas
            await self._load_schemas()
            
            # Load main configuration
            await self._load_configuration()
            
            # Start file watching if auto-reload enabled
            if self.auto_reload:
                await self._start_file_watching()
            
            self.logger.info("Configuration Manager initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration Manager initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the Configuration Manager"""
        self.logger.info("Shutting down Configuration Manager...")
        
        self._shutdown_event.set()
        
        # Stop file watchers
        for watcher in self._file_watchers:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        
        # Save current configuration
        await self._save_configuration()
        
        self.logger.info("Configuration Manager shutdown complete")
    
    async def get_config(self, path: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """
        Get configuration value(s)
        
        Args:
            path: Dot-separated path to specific config (e.g., 'system.logging.level')
                 If None, returns entire configuration
        
        Returns:
            Configuration value or dictionary
        """
        with self._lock:
            if path is None:
                return copy.deepcopy(self.config)
            
            keys = path.split('.')
            current = self.config
            
            try:
                for key in keys:
                    current = current[key]
                return copy.deepcopy(current) if isinstance(current, (dict, list)) else current
            except KeyError:
                self.logger.warning(f"Configuration path not found: {path}")
                return None
    
    async def set_config(self, path: str, value: Any, source: str = "api") -> bool:
        """
        Set configuration value
        
        Args:
            path: Dot-separated path to config key
            value: Value to set
            source: Source of the change
            
        Returns:
            True if set successfully
        """
        try:
            with self._lock:
                keys = path.split('.')
                old_value = await self.get_config(path)
                
                # Navigate to parent of target key
                current = self.config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value
                current[keys[-1]] = value
                
                # Create change record
                change = ConfigChange(
                    path=path,
                    old_value=old_value,
                    new_value=value,
                    timestamp=datetime.now(),
                    source=source
                )
                
                # Validate if schema exists
                if await self._validate_change(change):
                    change.applied = True
                    
                    # Notify listeners
                    await self._notify_change_listeners(change)
                    
                    # Update version
                    await self._create_version([change])
                    
                    self.logger.info(f"Configuration updated: {path} = {value}")
                    return True
                else:
                    # Revert change
                    if old_value is not None:
                        current[keys[-1]] = old_value
                    else:
                        del current[keys[-1]]
                    return False
        
        except Exception as e:
            self.logger.error(f"Failed to set configuration {path}: {e}")
            return False
    
    async def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific module
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module configuration dictionary
        """
        module_config = await self.get_config(f"modules.{module_name}")
        if module_config is None:
            # Return default module config
            module_config = {
                'enabled': True,
                'log_level': 'INFO',
                'auto_start': True
            }
            
        # Merge with system defaults
        system_defaults = await self.get_config("system.module_defaults") or {}
        merged_config = {**system_defaults, **module_config}
        
        return merged_config
    
    async def reload_configuration(self) -> bool:
        """
        Reload configuration from files
        
        Returns:
            True if reload successful
        """
        try:
            self.logger.info("Reloading configuration...")
            
            old_config = copy.deepcopy(self.config)
            await self._load_configuration()
            
            # Find changes
            changes = await self._detect_changes(old_config, self.config)
            
            if changes:
                self.logger.info(f"Configuration reloaded with {len(changes)} changes")
                
                # Notify listeners
                for change in changes:
                    await self._notify_change_listeners(change)
                
                # Create version
                await self._create_version(changes)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            return False
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration
        
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'validated_schemas': []
        }
        
        try:
            # Validate against all schemas
            for schema_name, schema in self.schemas.items():
                schema_results = await self._validate_against_schema(schema)
                results['validated_schemas'].append({
                    'schema': schema_name,
                    'valid': len(schema_results['errors']) == 0,
                    'errors': schema_results['errors'],
                    'warnings': schema_results['warnings']
                })
                
                results['errors'].extend(schema_results['errors'])
                results['warnings'].extend(schema_results['warnings'])
            
            results['valid'] = len(results['errors']) == 0
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation error: {e}")
        
        return results
    
    async def register_schema(self, schema: ConfigSchema) -> bool:
        """
        Register a configuration schema
        
        Args:
            schema: Schema to register
            
        Returns:
            True if registered successfully
        """
        try:
            self.schemas[schema.name] = schema
            self.logger.info(f"Registered configuration schema: {schema.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register schema {schema.name}: {e}")
            return False
    
    async def add_change_listener(self, path_pattern: str, listener: Callable) -> None:
        """
        Add a configuration change listener
        
        Args:
            path_pattern: Configuration path pattern to listen for
            listener: Callback function for changes
        """
        self.change_listeners[path_pattern].append(listener)
        self.logger.debug(f"Added change listener for pattern: {path_pattern}")
    
    async def remove_change_listener(self, path_pattern: str, listener: Callable) -> None:
        """
        Remove a configuration change listener
        
        Args:
            path_pattern: Configuration path pattern
            listener: Callback function to remove
        """
        if path_pattern in self.change_listeners:
            try:
                self.change_listeners[path_pattern].remove(listener)
                self.logger.debug(f"Removed change listener for pattern: {path_pattern}")
            except ValueError:
                pass
    
    async def create_backup(self) -> str:
        """
        Create a configuration backup
        
        Returns:
            Backup file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"olympus_config_{timestamp}.yaml"
            
            backup_data = {
                'version': self.current_version,
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
            
            with open(backup_path, 'w') as f:
                yaml.dump(backup_data, f, default_flow_style=False)
            
            self.logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    async def restore_backup(self, backup_path: str) -> bool:
        """
        Restore configuration from backup
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful
        """
        try:
            with open(backup_path, 'r') as f:
                backup_data = yaml.safe_load(f)
            
            old_config = copy.deepcopy(self.config)
            self.config = backup_data['config']
            self.current_version = backup_data.get('version', '1.0.0')
            
            # Validate restored configuration
            validation_results = await self.validate_configuration()
            if not validation_results['valid']:
                # Revert if validation fails
                self.config = old_config
                self.logger.error(f"Backup restore failed validation: {validation_results['errors']}")
                return False
            
            # Save restored configuration
            await self._save_configuration()
            
            self.logger.info(f"Configuration restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    async def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get configuration version history
        
        Returns:
            List of version information
        """
        return [
            {
                'version': v.version,
                'timestamp': v.timestamp.isoformat(),
                'changes_count': len(v.changes),
                'checksum': v.checksum,
                'description': v.description
            }
            for v in self.versions
        ]
    
    # Private methods
    
    async def _ensure_directories(self) -> None:
        """Ensure configuration directories exist"""
        for directory in [self.config_dir, self.schema_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_schemas(self) -> None:
        """Load configuration schemas"""
        try:
            # Load built-in schemas
            await self._load_builtin_schemas()
            
            # Load custom schemas from schema directory
            if self.schema_dir.exists():
                for schema_file in self.schema_dir.glob("*.yaml"):
                    try:
                        with open(schema_file, 'r') as f:
                            schema_data = yaml.safe_load(f)
                        
                        schema = ConfigSchema(
                            name=schema_data['name'],
                            scope=ConfigScope(schema_data['scope']),
                            required_fields=schema_data.get('required_fields', []),
                            optional_fields=schema_data.get('optional_fields', {}),
                            default_values=schema_data.get('defaults', {})
                        )
                        
                        self.schemas[schema.name] = schema
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load schema {schema_file}: {e}")
            
            self.logger.info(f"Loaded {len(self.schemas)} configuration schemas")
            
        except Exception as e:
            self.logger.error(f"Failed to load schemas: {e}")
    
    async def _load_builtin_schemas(self) -> None:
        """Load built-in configuration schemas"""
        # System schema
        system_schema = ConfigSchema(
            name="system",
            scope=ConfigScope.SYSTEM,
            required_fields=["name", "version"],
            default_values={
                "name": "OLYMPUS",
                "version": "1.0.0",
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "module_defaults": {
                    "enabled": True,
                    "auto_start": True,
                    "log_level": "INFO"
                }
            }
        )
        self.schemas["system"] = system_schema
        
        # Module schema
        module_schema = ConfigSchema(
            name="module",
            scope=ConfigScope.MODULE,
            optional_fields={
                "enabled": bool,
                "auto_start": bool,
                "log_level": str,
                "config": dict
            },
            default_values={
                "enabled": True,
                "auto_start": True,
                "log_level": "INFO"
            }
        )
        self.schemas["module"] = module_schema
    
    async def _load_configuration(self) -> None:
        """Load configuration from files"""
        try:
            # Load main configuration file
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.json':
                        file_config = json.load(f)
                    else:
                        file_config = yaml.safe_load(f)
                
                self.config.update(file_config)
                self.last_loaded = datetime.now()
                self.last_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
            
            # Load environment variables
            await self._load_env_overrides()
            
            # Apply schema defaults
            await self._apply_schema_defaults()
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Use minimal default configuration
            self.config = {
                "system": {
                    "name": "OLYMPUS",
                    "version": "1.0.0"
                }
            }
    
    async def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables"""
        env_prefix = "OLYMPUS_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_path = key[len(env_prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                await self.set_config(config_path, parsed_value, source="environment")
    
    async def _apply_schema_defaults(self) -> None:
        """Apply default values from schemas"""
        for schema in self.schemas.values():
            for key, default_value in schema.default_values.items():
                if await self.get_config(key) is None:
                    await self.set_config(key, default_value, source="schema_default")
    
    async def _save_configuration(self) -> None:
        """Save configuration to file"""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix(f".backup.{int(datetime.now().timestamp())}")
                self.config_path.rename(backup_path)
            
            # Save configuration
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2, default=str)
                else:
                    yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.debug("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    async def _start_file_watching(self) -> None:
        """Start watching configuration files for changes"""
        try:
            task = asyncio.create_task(self._file_watcher())
            self._file_watchers.append(task)
            self.logger.debug("Configuration file watching started")
        except Exception as e:
            self.logger.error(f"Failed to start file watching: {e}")
    
    async def _file_watcher(self) -> None:
        """Watch configuration files for changes"""
        while not self._shutdown_event.is_set():
            try:
                if self.config_path.exists():
                    current_mtime = datetime.fromtimestamp(self.config_path.stat().st_mtime)
                    
                    if self.last_modified and current_mtime > self.last_modified:
                        self.logger.info("Configuration file changed, reloading...")
                        await self.reload_configuration()
                
                await asyncio.sleep(self.reload_interval)
                
            except Exception as e:
                self.logger.error(f"File watcher error: {e}")
                await asyncio.sleep(self.reload_interval)
    
    async def _validate_change(self, change: ConfigChange) -> bool:
        """Validate a configuration change"""
        if not self.strict_validation:
            return True
        
        try:
            # Find applicable schemas
            applicable_schemas = []
            for schema in self.schemas.values():
                if self._path_matches_schema(change.path, schema):
                    applicable_schemas.append(schema)
            
            # If no schemas and require_schema is True, reject
            if not applicable_schemas and self.require_schema:
                self.logger.warning(f"No schema found for path: {change.path}")
                return False
            
            # Validate against applicable schemas
            for schema in applicable_schemas:
                if not await self._validate_value_against_schema(change.path, change.new_value, schema):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error for change {change.path}: {e}")
            return False
    
    async def _validate_against_schema(self, schema: ConfigSchema) -> Dict[str, Any]:
        """Validate configuration against a schema"""
        results = {
            'errors': [],
            'warnings': []
        }
        
        try:
            # Get relevant config section
            config_section = await self.get_config(schema.scope.value)
            if not config_section:
                return results
            
            # Check required fields
            for field in schema.required_fields:
                if field not in config_section:
                    results['errors'].append(f"Missing required field: {field}")
            
            # Check unknown fields if not allowed
            if not self.allow_unknown_keys:
                known_fields = set(schema.required_fields) | set(schema.optional_fields.keys())
                for field in config_section.keys():
                    if field not in known_fields:
                        results['warnings'].append(f"Unknown field: {field}")
            
        except Exception as e:
            results['errors'].append(f"Schema validation error: {e}")
        
        return results
    
    def _path_matches_schema(self, path: str, schema: ConfigSchema) -> bool:
        """Check if a configuration path matches a schema"""
        return path.startswith(schema.scope.value)
    
    async def _validate_value_against_schema(self, path: str, value: Any, schema: ConfigSchema) -> bool:
        """Validate a value against a schema"""
        try:
            # Extract field name from path
            field_name = path.split('.')[-1]
            
            # Check if field is allowed
            if (field_name not in schema.required_fields and 
                field_name not in schema.optional_fields and 
                not self.allow_unknown_keys):
                return False
            
            # Check type if specified in optional fields
            if field_name in schema.optional_fields:
                expected_type = schema.optional_fields[field_name]
                if expected_type and not isinstance(value, expected_type):
                    return False
            
            # Run validation rules
            if field_name in schema.validation_rules:
                validator = schema.validation_rules[field_name]
                if not validator(value):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Value validation error: {e}")
            return False
    
    async def _notify_change_listeners(self, change: ConfigChange) -> None:
        """Notify configuration change listeners"""
        try:
            for pattern, listeners in self.change_listeners.items():
                if self._path_matches_pattern(change.path, pattern):
                    for listener in listeners:
                        try:
                            if asyncio.iscoroutinefunction(listener):
                                await listener(change)
                            else:
                                listener(change)
                        except Exception as e:
                            self.logger.error(f"Change listener error: {e}")
        except Exception as e:
            self.logger.error(f"Failed to notify change listeners: {e}")
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a pattern"""
        # Simple pattern matching (could be extended with regex)
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        return path == pattern
    
    async def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """Detect changes between configurations"""
        changes = []
        
        def compare_dict(old_dict: Dict[str, Any], new_dict: Dict[str, Any], path: str = ""):
            # Check for changed/added keys
            for key, new_value in new_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in old_dict:
                    # New key
                    changes.append(ConfigChange(
                        path=current_path,
                        old_value=None,
                        new_value=new_value,
                        timestamp=datetime.now(),
                        source="file_reload"
                    ))
                elif old_dict[key] != new_value:
                    if isinstance(new_value, dict) and isinstance(old_dict[key], dict):
                        # Recursively compare nested dictionaries
                        compare_dict(old_dict[key], new_value, current_path)
                    else:
                        # Value changed
                        changes.append(ConfigChange(
                            path=current_path,
                            old_value=old_dict[key],
                            new_value=new_value,
                            timestamp=datetime.now(),
                            source="file_reload"
                        ))
            
            # Check for removed keys
            for key in old_dict:
                if key not in new_dict:
                    current_path = f"{path}.{key}" if path else key
                    changes.append(ConfigChange(
                        path=current_path,
                        old_value=old_dict[key],
                        new_value=None,
                        timestamp=datetime.now(),
                        source="file_reload"
                    ))
        
        compare_dict(old_config, new_config)
        return changes
    
    async def _create_version(self, changes: List[ConfigChange]) -> None:
        """Create a new configuration version"""
        try:
            # Generate version number
            version_parts = self.current_version.split('.')
            patch_version = int(version_parts[2]) + 1
            new_version = f"{version_parts[0]}.{version_parts[1]}.{patch_version}"
            
            # Calculate checksum
            config_str = json.dumps(self.config, sort_keys=True)
            checksum = hashlib.md5(config_str.encode()).hexdigest()
            
            # Create version record
            version = ConfigVersion(
                version=new_version,
                timestamp=datetime.now(),
                changes=changes,
                checksum=checksum,
                description=f"Configuration update with {len(changes)} changes"
            )
            
            self.versions.append(version)
            self.current_version = new_version
            
            # Keep only last 100 versions
            if len(self.versions) > 100:
                self.versions = self.versions[-100:]
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {e}")