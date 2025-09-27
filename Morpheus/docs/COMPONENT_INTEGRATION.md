# MORPHEUS Component Integration Patterns

## Overview

The MORPHEUS system uses well-defined integration patterns to ensure loose coupling, high cohesion, and maintainable system architecture. This document outlines the integration strategies, data flow patterns, and communication protocols between major components.

## Integration Architecture

### System-Level Integration Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    MORPHEUS ORCHESTRATOR                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │   Configuration │  │   Session       │  │   Metrics    │  │
│  │   Manager       │  │   Manager       │  │   Collector  │  │
│  └─────────────────┘  └─────────────────┘  └──────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼─────┐   ┌────▼────┐
    │Perception│    │Dream      │   │Storage  │
    │Layer     │    │Engine     │   │Layer    │
    └─────┬───┘    └─────┬─────┘   └────┬────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
                ┌────────▼────────┐
                │Integration      │
                │Bridges          │
                │(GASM/PyBullet)  │
                └─────────────────┘
```

## Core Integration Patterns

### 1. Dependency Injection Pattern

**Purpose**: Manage component dependencies and enable testing

```python
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any

# Define interfaces
class MaterialBridgeProtocol(Protocol):
    def get_material(self, name: str) -> MaterialProperties: ...
    def compute_interaction(self, mat1: str, mat2: str) -> Dict[str, Any]: ...

class DatabaseProtocol(Protocol):
    def store_experience(self, exp: Dict[str, Any]) -> int: ...
    def get_recent_experiences(self, hours: float) -> List[Dict]: ...

# Component with injected dependencies
class MorpheusOrchestrator:
    def __init__(self, 
                 config: Dict[str, Any],
                 database: DatabaseProtocol,
                 material_bridge: MaterialBridgeProtocol):
        self.config = config
        self.db = database
        self.material_bridge = material_bridge
        
        # Initialize sub-components with injected dependencies
        self.tactile = TactileProcessor(
            config['perception']['tactile'],
            material_bridge
        )
        
        self.dream_engine = DreamOrchestrator(
            database,
            material_bridge,
            DreamConfig(**config['dream'])
        )
```

### 2. Observer Pattern for Event-Driven Updates

**Purpose**: Decouple components through event notifications

```python
from typing import List, Callable
from enum import Enum

class EventType(Enum):
    EXPERIENCE_STORED = "experience_stored"
    DREAM_COMPLETED = "dream_completed"
    STRATEGY_LEARNED = "strategy_learned"
    SENSOR_CALIBRATED = "sensor_calibrated"

class EventBus:
    """Centralized event bus for component communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        
    def subscribe(self, event_type: EventType, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        
    def publish(self, event_type: EventType, data: Dict[str, Any]):
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

# Usage in components
class PerceptionManager:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
    def process_observation(self, obs: Dict[str, Any]):
        # Process observation
        result = self._process_internal(obs)
        
        # Notify other components
        self.event_bus.publish(EventType.EXPERIENCE_STORED, {
            'experience_id': result['id'],
            'material': obs.get('material'),
            'success': obs.get('success')
        })
```

### 3. Factory Pattern for Component Creation

**Purpose**: Centralize component creation and configuration

```python
class ComponentFactory:
    """Factory for creating MORPHEUS components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_database(self) -> MorpheusDatabase:
        db_config = self.config['database']
        return MorpheusDatabase(db_config)
        
    def create_material_bridge(self, gasm_path: str) -> MaterialBridge:
        return MaterialBridge(gasm_path)
        
    def create_tactile_processor(self, 
                               material_bridge: MaterialBridge) -> TactileProcessor:
        return TactileProcessor(
            self.config['perception']['tactile'],
            material_bridge
        )
        
    def create_dream_engine(self,
                          database: MorpheusDatabase,
                          material_bridge: MaterialBridge) -> DreamOrchestrator:
        dream_config = DreamConfig(**self.config['dream'])
        return DreamOrchestrator(database, material_bridge, dream_config)
        
    def create_orchestrator(self, 
                          gasm_roboting_path: str) -> MorpheusOrchestrator:
        # Create dependencies
        database = self.create_database()
        material_bridge = self.create_material_bridge(gasm_roboting_path)
        
        # Create main orchestrator
        return MorpheusOrchestrator(
            config=self.config,
            database=database,
            material_bridge=material_bridge
        )
```

## Data Flow Integration Patterns

### 1. Pipeline Pattern for Perception Processing

```python
from typing import Any, Optional, Callable, List

class ProcessingPipeline:
    """Sequential processing pipeline with error handling"""
    
    def __init__(self):
        self._stages: List[Callable[[Any], Any]] = []
        
    def add_stage(self, processor: Callable[[Any], Any]):
        self._stages.append(processor)
        return self
        
    def process(self, input_data: Any) -> Any:
        current_data = input_data
        
        for i, stage in enumerate(self._stages):
            try:
                current_data = stage(current_data)
                if current_data is None:
                    logger.warning(f"Stage {i} returned None, stopping pipeline")
                    break
            except Exception as e:
                logger.error(f"Pipeline stage {i} failed: {e}")
                raise
                
        return current_data

# Usage for perception processing
def create_perception_pipeline(tactile_processor, audio_processor, fusion_network):
    pipeline = ProcessingPipeline()
    
    return (pipeline
        .add_stage(lambda obs: extract_sensor_data(obs))
        .add_stage(lambda data: process_tactile(data, tactile_processor))
        .add_stage(lambda data: process_audio(data, audio_processor))
        .add_stage(lambda data: fuse_modalities(data, fusion_network))
        .add_stage(lambda data: generate_embedding(data))
    )
```

### 2. Command Pattern for Action Execution

```python
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def undo(self) -> None: ...

class PerceptionCommand(Command):
    def __init__(self, orchestrator: MorpheusOrchestrator, observation: Dict):
        self.orchestrator = orchestrator
        self.observation = observation
        self.result = None
        
    def execute(self) -> Dict[str, Any]:
        self.result = self.orchestrator.perceive(self.observation)
        return self.result
        
    def undo(self) -> None:
        if self.result and 'experience_id' in self.result:
            # Could implement experience deletion if needed
            pass

class DreamCommand(Command):
    def __init__(self, orchestrator: MorpheusOrchestrator, duration: float):
        self.orchestrator = orchestrator
        self.duration = duration
        self.result = None
        
    def execute(self) -> Dict[str, Any]:
        self.result = self.orchestrator.dream(self.duration)
        return self.result
        
    def undo(self) -> None:
        # Dream sessions are generally not undoable
        pass

class CommandInvoker:
    def __init__(self):
        self.history: List[Command] = []
        
    def execute_command(self, command: Command) -> Dict[str, Any]:
        result = command.execute()
        self.history.append(command)
        return result
```

## Interface Definitions

### 1. Sensor Interface Protocol

```python
class SensorProtocol(Protocol):
    """Common interface for all sensor processors"""
    
    def process(self, sensor_data: Any) -> Optional[Dict[str, Any]]:
        """Process raw sensor data and return structured result"""
        ...
        
    def calibrate(self, calibration_data: Any) -> bool:
        """Calibrate sensor with given data"""
        ...
        
    def get_status(self) -> Dict[str, Any]:
        """Get current sensor status"""
        ...

# Implementations
class TactileProcessor:
    def process(self, sensor_data: Any) -> Optional[TactileSignature]:
        # Implementation
        pass
        
    def calibrate(self, calibration_data: Any) -> bool:
        # Implementation  
        pass
        
    def get_status(self) -> Dict[str, Any]:
        return {
            'type': 'tactile',
            'sensitivity': self.sensitivity,
            'sampling_rate': self.sampling_rate,
            'last_contact': self.last_contact_time
        }
```

### 2. Storage Interface Protocol

```python
class StorageProtocol(Protocol):
    """Common interface for storage backends"""
    
    def store_experience(self, experience: Dict[str, Any]) -> int: ...
    def get_experiences(self, filters: Dict[str, Any]) -> List[Dict]: ...
    def store_strategy(self, strategy: Dict[str, Any]) -> int: ...
    def get_strategies(self, filters: Dict[str, Any]) -> List[Dict]: ...
    def cleanup_old_data(self, retention_days: int) -> int: ...
```

## Configuration Integration

### 1. Hierarchical Configuration System

```python
from pathlib import Path
import yaml
from typing import Any, Dict

class ConfigurationManager:
    """Manages hierarchical configuration with environment overrides"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_environment_overrides()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        import os
        
        # Database overrides
        if 'DATABASE_HOST' in os.environ:
            self.config['database']['host'] = os.environ['DATABASE_HOST']
        if 'DATABASE_PORT' in os.environ:
            self.config['database']['port'] = int(os.environ['DATABASE_PORT'])
            
        # Add more overrides as needed
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current
        
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Required keys
        required_keys = [
            'database.host',
            'database.database', 
            'database.user',
            'perception.tactile.sensitivity',
            'dream.parallel_dreams'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"Missing required configuration: {key}")
                
        return errors
```

## Error Handling Integration

### 1. Centralized Error Handling

```python
from enum import Enum
from typing import Optional, Callable

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MorpheusError(Exception):
    def __init__(self, message: str, severity: ErrorSeverity, component: str):
        super().__init__(message)
        self.severity = severity
        self.component = component

class ErrorHandler:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.error_callbacks: Dict[ErrorSeverity, List[Callable]] = {}
        
    def register_callback(self, severity: ErrorSeverity, callback: Callable):
        if severity not in self.error_callbacks:
            self.error_callbacks[severity] = []
        self.error_callbacks[severity].append(callback)
        
    def handle_error(self, error: MorpheusError):
        logger.error(f"[{error.component}] {error.severity.value}: {error}")
        
        # Execute registered callbacks
        if error.severity in self.error_callbacks:
            for callback in self.error_callbacks[error.severity]:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")
                    
        # Publish error event
        self.event_bus.publish("error_occurred", {
            'error': str(error),
            'severity': error.severity.value,
            'component': error.component
        })
```

## Integration Testing Patterns

### 1. Integration Test Framework

```python
class IntegrationTestSuite:
    """Framework for testing component integration"""
    
    def __init__(self):
        self.test_config = self._create_test_config()
        self.mock_components = {}
        
    def _create_test_config(self) -> Dict[str, Any]:
        return {
            'database': {
                'host': 'localhost',
                'database': 'morpheus_test',
                'user': 'test_user',
                'password': 'test_pass'
            },
            'perception': {
                'tactile': {'sensitivity': 0.01, 'sampling_rate': 100}
            },
            'dream': {'parallel_dreams': 2}
        }
        
    def test_perception_integration(self):
        """Test perception components working together"""
        # Setup
        factory = ComponentFactory(self.test_config)
        orchestrator = factory.create_orchestrator("./test_gasm")
        
        # Test data
        observation = {
            'material': 'steel',
            'body_id': 1,
            'robot_position': [0, 0, 0.5],
            'success': True
        }
        
        # Execute
        result = orchestrator.perceive(observation)
        
        # Verify
        assert result is not None
        assert 'experience_id' in result
        assert 'fused_embedding' in result
        
    def test_dream_integration(self):
        """Test dream engine integration"""
        factory = ComponentFactory(self.test_config)
        orchestrator = factory.create_orchestrator("./test_gasm")
        
        # Add some test experiences
        for i in range(10):
            observation = {
                'material': 'rubber',
                'success': i % 2 == 0,
                'reward': 1.0 if i % 2 == 0 else -0.5
            }
            orchestrator.perceive(observation)
            
        # Test dream session
        result = orchestrator.dream(duration=5)
        
        # Verify
        assert result is not None
        assert 'strategies_found' in result
        assert result['experiences_processed'] > 0
```

## Performance Integration Patterns

### 1. Async Processing Integration

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMorpheusOrchestrator:
    """Async version of MORPHEUS for high-throughput scenarios"""
    
    def __init__(self, sync_orchestrator: MorpheusOrchestrator):
        self.sync_orchestrator = sync_orchestrator
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def perceive_async(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Async perception processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_orchestrator.perceive,
            observation
        )
        
    async def dream_async(self, duration: float) -> Dict[str, Any]:
        """Async dream processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_orchestrator.dream,
            duration
        )
        
    async def batch_perceive(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple observations concurrently"""
        tasks = [self.perceive_async(obs) for obs in observations]
        return await asyncio.gather(*tasks)
```

These integration patterns provide a robust foundation for building and maintaining the MORPHEUS system, ensuring clean component interactions, testability, and scalability.