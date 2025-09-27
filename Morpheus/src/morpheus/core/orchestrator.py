"""Main orchestrator for MORPHEUS system.

This module provides the central coordination system that integrates all MORPHEUS
components including perception, dream cycles, material learning, and session management.

Features:
- UUID-based session tracking
- Multi-modal perception processing 
- Dream cycle orchestration
- Strategy learning management
- Comprehensive error handling and cleanup
- Performance monitoring and metrics
"""

import time
import uuid
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

# Import MORPHEUS components
from ..storage.postgres_storage import PostgreSQLStorage
from ..integration.material_bridge import MaterialBridge
from ..integration.gasm_bridge import GASMBridge
from ..perception.tactile_processor import TactileProcessor
from ..perception.audio_spatial import AudioProcessor
from ..perception.sensory_fusion import SensoryFusionNetwork
from ..dream_sim.dream_orchestrator import DreamOrchestrator, DreamConfig
from ..predictive.forward_model import SensoryPredictor
from ..core.config import MorpheusConfig, ConfigManager
from ..core.types import (
    SensoryExperience, TactileSignature, AudioSignature, VisualSignature,
    MaterialProperties, ActionType, SystemMetrics, PredictionResult, Vector3D
)

logger = logging.getLogger(__name__)


class MorpheusOrchestrator:
    """Main orchestration system for MORPHEUS multi-modal perception and learning."""
    
    def __init__(self, 
                 config: Union[str, Path, MorpheusConfig],
                 gasm_robotics_path: Optional[Union[str, Path]] = None,
                 database_config: Optional[Dict[str, Any]] = None):
        """Initialize MORPHEUS system with all components.
        
        Args:
            config: Configuration file path or MorpheusConfig object
            gasm_robotics_path: Optional path to GASM-Robotics directory
            database_config: Optional database configuration override
            
        Raises:
            RuntimeError: If initialization fails
            FileNotFoundError: If required paths don't exist
        """
        self.start_time = time.time()
        self.session_id = str(uuid.uuid4())
        self._shutdown_requested = False
        self._lock = threading.RLock()
        
        try:
            # Load configuration
            self.config = self._load_configuration(config)
            
            # Override paths if provided
            if gasm_robotics_path:
                self.config.gasm.roboting_path = str(gasm_robotics_path)
            
            # Initialize database
            db_config = database_config or self.config.database.dict()
            self.database = PostgreSQLStorage(db_config)
            
            # Initialize material bridge
            self.material_bridge = MaterialBridge(self.config.gasm.roboting_path)
            
            # Initialize perception components
            self._initialize_perception()
            
            # Initialize neural networks
            self._initialize_networks()
            
            # Initialize dream engine
            self._initialize_dream_engine()
            
            # Initialize GASM bridge if enabled
            self.gasm_bridge = None
            if self.config.gasm.enabled:
                try:
                    self.gasm_bridge = GASMBridge(self.config.gasm.dict())
                except Exception as e:
                    logger.warning(f"GASM bridge initialization failed: {e}")
            
            # Performance tracking
            self.metrics = SystemMetrics()
            self.perception_count = 0
            self.dream_count = 0
            self.error_count = 0
            
            # Callback registry
            self.callbacks = {
                'perception': [],
                'dream_start': [],
                'dream_end': [],
                'error': []
            }
            
            logger.info(f"MORPHEUS initialized successfully - Session: {self.session_id}")
            logger.info(f"Available materials: {list(self.material_bridge.materials.keys())}")
            logger.info(f"System mode: {self.config.system.mode}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MORPHEUS: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"MORPHEUS initialization failed: {e}")
    
    def _load_configuration(self, config: Union[str, Path, MorpheusConfig]) -> MorpheusConfig:
        """Load and validate configuration."""
        if isinstance(config, MorpheusConfig):
            return config
        
        config_manager = ConfigManager()
        loaded_config = config_manager.load_config(config, validate=True)
        
        # Validate configuration
        issues = loaded_config.validate_config()
        if issues:
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")
        
        return loaded_config
    
    def _initialize_perception(self):
        """Initialize perception components."""
        self.tactile_processor = None
        self.audio_processor = None
        
        if self.config.perception.tactile.enabled:
            self.tactile_processor = TactileProcessor(
                self.config.perception.tactile.dict(),
                self.material_bridge
            )
            logger.info("Tactile processor initialized")
        
        if self.config.perception.audio.enabled:
            self.audio_processor = AudioProcessor(
                self.config.perception.audio.dict()
            )
            logger.info("Audio processor initialized")
    
    def _initialize_networks(self):
        """Initialize neural networks."""
        self.fusion_network = None
        self.predictor = None
        
        if self.config.perception.fusion_enabled:
            self.fusion_network = SensoryFusionNetwork(
                self.config.networks.fusion.dict()
            )
            logger.info("Sensory fusion network initialized")
        
        if self.config.perception.prediction_enabled:
            self.predictor = SensoryPredictor(
                self.config.networks.predictor.dict()
            )
            logger.info("Forward model predictor initialized")
    
    def _initialize_dream_engine(self):
        """Initialize dream orchestration engine."""
        self.dream_engine = None
        
        if self.config.dream.enabled:
            dream_config = DreamConfig(**self.config.dream.dict())
            self.dream_engine = DreamOrchestrator(
                self.database,
                self.material_bridge,
                dream_config
            )
            logger.info("Dream orchestrator initialized")
    
    @contextmanager
    def session_context(self, session_id: Optional[str] = None):
        """Context manager for session tracking."""
        if session_id:
            old_session = self.session_id
            self.session_id = session_id
        
        start_time = time.time()
        try:
            yield self.session_id
        finally:
            duration = time.time() - start_time
            logger.debug(f"Session context duration: {duration:.3f}s")
            
            if session_id:
                self.session_id = old_session
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register event callback.
        
        Args:
            event_type: Event type ('perception', 'dream_start', 'dream_end', 'error')
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown callback event type: {event_type}")
    
    def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger registered callbacks for event type."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
    
    def perceive(self, 
                observation: Dict[str, Any],
                session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process multi-modal perception and store experience.
        
        Args:
            observation: Dictionary with sensor data and context
            session_id: Optional session ID override
            
        Returns:
            Perception results with embeddings and predictions
            
        Raises:
            ValueError: If observation is invalid
            RuntimeError: If perception processing fails
        """
        if self._shutdown_requested:
            raise RuntimeError("System is shutting down")
        
        with self._lock:
            self.perception_count += 1
            
        start_time = time.time()
        
        try:
            with self.session_context(session_id):
                # Validate observation
                if not isinstance(observation, dict):
                    raise ValueError("Observation must be a dictionary")
                
                timestamp = time.time()
                
                # Create experience object
                experience = SensoryExperience(
                    session_id=self.session_id,
                    timestamp=timestamp,
                    primary_material=observation.get('material', 'default'),
                    action_type=self._parse_action_type(observation.get('action_type')),
                    action_parameters=observation.get('action_params', {}),
                    success=observation.get('success', True),
                    reward=observation.get('reward', 0.0),
                    tags=observation.get('tags', []),
                    notes=observation.get('notes')
                )
                
                # Process tactile perception
                tactile_result = self._process_tactile(observation)
                if tactile_result:
                    experience.tactile = tactile_result
                
                # Process audio perception
                audio_result = self._process_audio(observation)
                if audio_result:
                    experience.audio = audio_result
                
                # Process visual perception
                visual_result = self._process_visual(observation)
                if visual_result:
                    experience.visual = visual_result
                
                # Compute embeddings
                experience.compute_embeddings()
                
                # Fuse modalities
                if self.fusion_network:
                    fused_embedding = self._fuse_modalities(experience)
                    experience.fused_embedding = fused_embedding
                
                # Store experience in database
                experience_id = self.database.store_experience(experience)
                
                # Predict next state if action provided
                prediction = None
                if 'action' in observation and self.predictor:
                    prediction = self._predict_next_state(
                        experience.fused_embedding or experience.tactile_embedding,
                        observation['action']
                    )
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time, True)
                
                # Trigger callbacks
                self._trigger_callbacks('perception', experience, prediction)
                
                result = {
                    'experience_id': experience_id,
                    'session_id': self.session_id,
                    'perception_count': self.perception_count,
                    'processing_time': processing_time,
                    'tactile': experience.tactile.to_dict() if experience.tactile else None,
                    'audio': experience.audio.to_dict() if experience.audio else None,
                    'visual': experience.visual.to_dict() if experience.visual else None,
                    'fused_embedding': experience.fused_embedding.tolist() if experience.fused_embedding is not None else None,
                    'prediction': prediction.to_dict() if prediction else None,
                    'material': experience.primary_material
                }
                
                logger.debug(f"Perception processed in {processing_time:.3f}s")
                return result
                
        except Exception as e:
            self.error_count += 1
            self._update_metrics(time.time() - start_time, False)
            self._trigger_callbacks('error', e, 'perception')
            logger.error(f"Perception failed: {e}")
            raise RuntimeError(f"Perception processing failed: {e}")
    
    def _process_tactile(self, observation: Dict[str, Any]) -> Optional[TactileSignature]:
        """Process tactile perception from observation."""
        if not self.tactile_processor or 'body_id' not in observation:
            return None
        
        try:
            return self.tactile_processor.process_contacts(
                observation['body_id'],
                observation.get('material')
            )
        except Exception as e:
            logger.warning(f"Tactile processing failed: {e}")
            return None
    
    def _process_audio(self, observation: Dict[str, Any]) -> Optional[AudioSignature]:
        """Process audio perception from observation."""
        if not self.audio_processor or 'robot_position' not in observation:
            return None
        
        try:
            return self.audio_processor.process_spatial_audio(
                Vector3D.from_array(np.array(observation['robot_position'])),
                Vector3D.from_array(np.array(observation.get('robot_velocity', [0, 0, 0])))
            )
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return None
    
    def _process_visual(self, observation: Dict[str, Any]) -> Optional[VisualSignature]:
        """Process visual perception from observation."""
        if 'visual_features' not in observation:
            return None
        
        try:
            features = np.array(observation['visual_features'])
            return VisualSignature(
                timestamp=time.time(),
                features=features,
                object_positions=[Vector3D.from_array(pos) for pos in observation.get('object_positions', [])],
                colors=observation.get('colors', []),
                textures=observation.get('textures', []),
                lighting_conditions=observation.get('lighting', {}),
                depth_map=np.array(observation['depth_map']) if 'depth_map' in observation else None
            )
        except Exception as e:
            logger.warning(f"Visual processing failed: {e}")
            return None
    
    def _parse_action_type(self, action_str: Optional[str]) -> Optional[ActionType]:
        """Parse action type string to enum."""
        if not action_str:
            return None
        
        action_mapping = {
            'touch': ActionType.TOUCH,
            'grasp': ActionType.GRASP,
            'grip': ActionType.GRASP,
            'push': ActionType.PUSH,
            'pull': ActionType.PULL,
            'lift': ActionType.LIFT,
            'slide': ActionType.SLIDE,
            'rotate': ActionType.ROTATE,
            'explore': ActionType.EXPLORE
        }
        
        return action_mapping.get(action_str.lower())
    
    def _fuse_modalities(self, experience: SensoryExperience) -> Optional[np.ndarray]:
        """Fuse multiple sensory modalities."""
        if not self.fusion_network:
            return None
        
        try:
            # Prepare input tensors
            tactile = torch.tensor(experience.tactile_embedding, dtype=torch.float32) if experience.tactile_embedding is not None else torch.zeros(64)
            audio = torch.tensor(experience.audio_embedding, dtype=torch.float32) if experience.audio_embedding is not None else torch.zeros(32)
            visual = torch.tensor(experience.visual_embedding, dtype=torch.float32) if experience.visual_embedding is not None else torch.zeros(128)
            
            # Stack inputs
            inputs = torch.stack([tactile, audio, visual]).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                fused = self.fusion_network(inputs)
                
            return fused.squeeze(0).numpy()
            
        except Exception as e:
            logger.warning(f"Sensory fusion failed: {e}")
            return None
    
    def _predict_next_state(self, 
                           current_state: Optional[np.ndarray],
                           action: Dict[str, Any]) -> Optional[PredictionResult]:
        """Predict next sensory state."""
        if not self.predictor or current_state is None:
            return None
        
        try:
            # Convert action to vector
            action_vec = np.zeros(7)  # 6DOF + gripper
            
            if 'position' in action:
                pos = np.array(action['position'])[:3]
                action_vec[:len(pos)] = pos
            if 'orientation' in action:
                orient = np.array(action['orientation'])[:3]
                action_vec[3:3+len(orient)] = orient
            if 'gripper' in action:
                action_vec[6] = float(action['gripper'])
            
            # Prepare tensors
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action_vec, dtype=torch.float32).unsqueeze(0)
            
            # Forward prediction
            with torch.no_grad():
                next_state, uncertainty = self.predictor(state_tensor, action_tensor)
                
            return PredictionResult(
                predicted_state=next_state.squeeze(0).numpy(),
                uncertainty=uncertainty.squeeze(0).numpy(),
                confidence=1.0 - uncertainty.mean().item(),
                prediction_horizon=0.1  # 100ms horizon
            )
            
        except Exception as e:
            logger.warning(f"State prediction failed: {e}")
            return None
    
    def dream(self, 
              duration: float = 60.0,
              session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enter dream state for experience replay and optimization.
        
        Args:
            duration: Dream duration in seconds
            session_id: Optional session ID override
            
        Returns:
            Dream session results
            
        Raises:
            RuntimeError: If dreaming fails or is disabled
        """
        if not self.dream_engine:
            raise RuntimeError("Dream engine is disabled")
        
        if self._shutdown_requested:
            raise RuntimeError("System is shutting down")
        
        with self._lock:
            self.dream_count += 1
        
        dream_id = str(uuid.uuid4())
        
        try:
            with self.session_context(session_id):
                logger.info(f"Entering dream state #{self.dream_count} (ID: {dream_id})")
                
                # Trigger dream start callbacks
                self._trigger_callbacks('dream_start', dream_id, duration)
                
                # Run dream session
                start_time = time.time()
                results = self.dream_engine.dream(duration)
                dream_time = time.time() - start_time
                
                # Update metrics
                self.metrics.dream_count = self.dream_count
                
                # Trigger dream end callbacks
                self._trigger_callbacks('dream_end', dream_id, results)
                
                logger.info(f"Dream session complete: {results.get('strategies_found', 0)} strategies discovered")
                
                return {
                    'dream_id': dream_id,
                    'session_id': self.session_id,
                    'dream_count': self.dream_count,
                    'duration_requested': duration,
                    'duration_actual': dream_time,
                    **results
                }
                
        except Exception as e:
            self.error_count += 1
            self._trigger_callbacks('error', e, 'dream')
            logger.error(f"Dream session failed: {e}")
            raise RuntimeError(f"Dream processing failed: {e}")
    
    def predict_material_interaction(self, 
                                    material1: str,
                                    material2: str,
                                    scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict sensory outcome of material interaction.
        
        Args:
            material1: First material name
            material2: Second material name
            scenario: Optional scenario parameters
            
        Returns:
            Predicted interaction properties and sensory signatures
        """
        try:
            # Get interaction properties from material bridge
            interaction = self.material_bridge.compute_interaction(
                material1, material2,
                contact_force=scenario.get('force', 1.0) if scenario else 1.0,
                relative_velocity=scenario.get('velocity', 0.0) if scenario else 0.0
            )
            
            # Get material properties
            mat1_props = self.material_bridge.get_material(material1)
            mat2_props = self.material_bridge.get_material(material2)
            
            if not mat1_props or not mat2_props:
                raise ValueError(f"Unknown materials: {material1}, {material2}")
            
            # Predict tactile signature
            tactile_prediction = mat1_props.compute_tactile_signature(
                contact_force=scenario.get('force', 1.0) if scenario else 1.0,
                contact_velocity=scenario.get('velocity', 0.0) if scenario else 0.0
            )
            
            # Predict audio signature
            audio_prediction = mat1_props.compute_audio_signature(
                impact_velocity=scenario.get('impact_velocity', 0.1) if scenario else 0.1
            )
            
            return {
                'materials': [material1, material2],
                'interaction': interaction.to_dict() if hasattr(interaction, 'to_dict') else interaction,
                'predicted_tactile': tactile_prediction,
                'predicted_audio': audio_prediction,
                'confidence': 0.8,
                'scenario': scenario or {}
            }
            
        except Exception as e:
            logger.error(f"Material interaction prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def get_learned_strategies(self, 
                              category: Optional[str] = None,
                              min_confidence: float = 0.0,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Get learned strategies from database.
        
        Args:
            category: Optional category filter
            min_confidence: Minimum confidence threshold
            limit: Maximum number of strategies to return
            
        Returns:
            List of strategy dictionaries
        """
        try:
            return self.database.get_best_strategies(
                category=category,
                min_confidence=min_confidence,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to retrieve strategies: {e}")
            return []
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of current or specified session.
        
        Args:
            session_id: Optional session ID (defaults to current session)
            
        Returns:
            Session summary dictionary
        """
        target_session = session_id or self.session_id
        
        try:
            # Get recent experiences for this session
            recent_experiences = self.database.get_recent_experiences(
                hours=24, limit=1000, session_id=target_session
            )
            
            # Calculate statistics
            materials_used = set()
            success_count = 0
            total_reward = 0.0
            action_types = set()
            
            for exp in recent_experiences:
                if exp.get('primary_material'):
                    materials_used.add(exp['primary_material'])
                if exp.get('success'):
                    success_count += 1
                if exp.get('reward'):
                    total_reward += exp['reward']
                if exp.get('action_type'):
                    action_types.add(exp['action_type'])
            
            # Get system uptime
            uptime = time.time() - self.start_time
            
            return {
                'session_id': target_session,
                'system_uptime': uptime,
                'perception_count': self.perception_count,
                'dream_count': self.dream_count,
                'error_count': self.error_count,
                'materials_explored': list(materials_used),
                'action_types_used': list(action_types),
                'success_rate': success_count / len(recent_experiences) if recent_experiences else 0.0,
                'average_reward': total_reward / len(recent_experiences) if recent_experiences else 0.0,
                'total_experiences': len(recent_experiences),
                'learned_strategies': len(self.get_learned_strategies()),
                'system_mode': self.config.system.mode,
                'components_active': {
                    'tactile': self.tactile_processor is not None,
                    'audio': self.audio_processor is not None,
                    'fusion': self.fusion_network is not None,
                    'predictor': self.predictor is not None,
                    'dream': self.dream_engine is not None,
                    'gasm': self.gasm_bridge is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return {
                'session_id': target_session,
                'error': str(e),
                'perception_count': self.perception_count,
                'dream_count': self.dream_count,
                'error_count': self.error_count
            }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system performance metrics."""
        with self._lock:
            self.metrics.timestamp = datetime.now()
            self.metrics.perception_count = self.perception_count
            self.metrics.dream_count = self.dream_count
            
            # Get database size
            try:
                self.metrics.database_size = self.database.get_database_size()
            except:
                pass
            
            # Get memory usage
            try:
                import psutil
                process = psutil.Process()
                self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except:
                pass
            
            return self.metrics
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics."""
        # Update average processing time with exponential moving average
        alpha = 0.1
        if self.metrics.average_processing_time == 0:
            self.metrics.average_processing_time = processing_time
        else:
            self.metrics.average_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics.average_processing_time
            )
        
        # Update success rate
        total_operations = self.perception_count + self.dream_count
        if total_operations > 0:
            success_count = total_operations - self.error_count
            self.metrics.success_rate = success_count / total_operations
    
    def cleanup(self):
        """Clean up all resources and connections."""
        logger.info(f"Cleaning up MORPHEUS resources for session {self.session_id}")
        
        self._shutdown_requested = True
        
        try:
            # Cleanup database connections
            if hasattr(self.database, 'cleanup'):
                self.database.cleanup()
            
            # Cleanup neural networks
            if self.fusion_network:
                del self.fusion_network
            if self.predictor:
                del self.predictor
            
            # Cleanup GASM bridge
            if self.gasm_bridge and hasattr(self.gasm_bridge, 'cleanup'):
                self.gasm_bridge.cleanup()
            
            # Clear callbacks
            self.callbacks.clear()
            
            uptime = time.time() - self.start_time
            logger.info(f"Session {self.session_id} ended after {uptime:.1f}s")
            logger.info(f"Final stats: {self.perception_count} perceptions, {self.dream_count} dreams, {self.error_count} errors")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        
        if exc_type:
            logger.error(f"Exception in MORPHEUS context: {exc_type.__name__}: {exc_val}")
            return False
        
        return True
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            if not self._shutdown_requested:
                self.cleanup()
        except:
            pass  # Ignore errors during destruction


# Convenience functions for common operations

def create_morpheus_system(config_path: Union[str, Path],
                          gasm_path: Optional[Union[str, Path]] = None,
                          database_config: Optional[Dict[str, Any]] = None) -> MorpheusOrchestrator:
    """Create and initialize MORPHEUS system.
    
    Args:
        config_path: Path to configuration file
        gasm_path: Optional GASM-Robotics path
        database_config: Optional database configuration
        
    Returns:
        Initialized MorpheusOrchestrator
    """
    return MorpheusOrchestrator(config_path, gasm_path, database_config)


def quick_perception_test(morpheus: MorpheusOrchestrator,
                         material: str = 'steel',
                         force: float = 5.0) -> Dict[str, Any]:
    """Quick perception test with default parameters.
    
    Args:
        morpheus: MORPHEUS orchestrator instance
        material: Material to test
        force: Contact force in Newtons
        
    Returns:
        Perception results
    """
    observation = {
        'material': material,
        'body_id': 1,
        'robot_position': [0, 0, 0.5],
        'robot_velocity': [0.1, 0, 0],
        'contact_force': force,
        'forces': [force, 0, 0],
        'action_type': 'touch',
        'action_params': {'pressure': force},
        'success': True,
        'reward': 1.0
    }
    
    return morpheus.perceive(observation)