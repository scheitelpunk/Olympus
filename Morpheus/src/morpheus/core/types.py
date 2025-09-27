"""Core type definitions for MORPHEUS system.

This module contains all dataclasses and type definitions used throughout
the MORPHEUS multi-modal perception and optimization system.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from enum import Enum, auto


class SensorType(Enum):
    """Enumeration of sensor types."""
    TACTILE = auto()
    AUDIO = auto()
    VISUAL = auto()
    FORCE = auto()
    PROPRIOCEPTIVE = auto()


class MaterialType(Enum):
    """Enumeration of material types."""
    METAL = auto()
    PLASTIC = auto()
    RUBBER = auto()
    GLASS = auto()
    COMPOSITE = auto()
    UNKNOWN = auto()


class ActionType(Enum):
    """Enumeration of action types."""
    TOUCH = auto()
    GRASP = auto()
    PUSH = auto()
    PULL = auto()
    LIFT = auto()
    SLIDE = auto()
    ROTATE = auto()
    EXPLORE = auto()


@dataclass
class Vector3D:
    """3D vector representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3D':
        """Create from numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def magnitude(self) -> float:
        """Calculate magnitude."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        """Return normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D()
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)


@dataclass
class ContactPoint:
    """Individual contact point data."""
    position: Vector3D
    normal: Vector3D
    force_magnitude: float
    object_a: int
    object_b: int
    link_a: int = -1
    link_b: int = -1
    friction_force: Vector3D = field(default_factory=Vector3D)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'position': [self.position.x, self.position.y, self.position.z],
            'normal': [self.normal.x, self.normal.y, self.normal.z],
            'force_magnitude': self.force_magnitude,
            'object_a': self.object_a,
            'object_b': self.object_b,
            'link_a': self.link_a,
            'link_b': self.link_b,
            'friction_force': [self.friction_force.x, self.friction_force.y, self.friction_force.z]
        }


@dataclass
class MaterialProperties:
    """Complete material properties from GASM-Robotics."""
    name: str
    material_type: MaterialType = MaterialType.UNKNOWN
    color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    friction: float = 0.5
    restitution: float = 0.5
    density: float = 1000.0  # kg/m³
    young_modulus: float = 1e9  # Pa
    poisson_ratio: float = 0.3
    
    # Derived properties for perception
    thermal_conductivity: Optional[float] = None
    surface_roughness: Optional[float] = None
    hardness_shore: Optional[float] = None
    
    def __post_init__(self):
        """Compute derived properties after initialization."""
        # Estimate thermal conductivity from Young's modulus
        if self.thermal_conductivity is None:
            if self.young_modulus > 100e9:  # Metal-like
                self.thermal_conductivity = 200.0  # W/mK
            elif self.young_modulus < 10e9:  # Polymer/rubber
                self.thermal_conductivity = 0.2
            else:
                self.thermal_conductivity = 50.0
        
        # Estimate surface roughness from friction
        if self.surface_roughness is None:
            self.surface_roughness = self.friction * 10.0  # μm
        
        # Estimate Shore hardness from Young's modulus
        if self.hardness_shore is None:
            # Simplified conversion
            self.hardness_shore = min(100, np.log10(self.young_modulus + 1) * 10)
    
    def get_material_type(self) -> MaterialType:
        """Determine material type from properties."""
        if self.young_modulus > 100e9:
            return MaterialType.METAL
        elif self.young_modulus < 1e9 and self.restitution > 0.6:
            return MaterialType.RUBBER
        elif self.density < 2000 and self.young_modulus < 10e9:
            return MaterialType.PLASTIC
        elif self.young_modulus > 50e9 and self.density > 2200:
            return MaterialType.GLASS
        else:
            return MaterialType.COMPOSITE


@dataclass
class TactileSignature:
    """Complete tactile signature from material interaction."""
    timestamp: float
    material: str
    contact_points: List[ContactPoint]
    total_force: float
    contact_area: float
    pressure: float  # N/m²
    texture_descriptor: str  # 'smooth', 'rough', 'textured', 'sticky'
    hardness: float  # 0-1 scale
    temperature_feel: float  # 0-1 scale (0=cool, 1=warm)
    vibration_spectrum: np.ndarray  # Frequency spectrum
    grip_quality: float  # 0-1 scale
    deformation: float  # mm
    stiffness: float  # N/m
    
    def to_embedding(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert to fixed-size embedding vector."""
        embedding = np.zeros(embedding_dim)
        
        # Basic physical properties (8 dims)
        embedding[0] = min(self.total_force / 100, 1.0)  # Normalized force
        embedding[1] = min(self.contact_area / 0.01, 1.0)  # cm²
        embedding[2] = min(self.pressure / 1000, 1.0)  # kPa
        embedding[3] = self.hardness
        embedding[4] = self.temperature_feel
        embedding[5] = self.grip_quality
        embedding[6] = min(len(self.contact_points) / 10, 1.0)
        embedding[7] = min(self.deformation / 10, 1.0)  # mm
        
        # Texture encoding (4 dims)
        texture_map = {'smooth': [1,0,0,0], 'rough': [0,1,0,0], 
                      'textured': [0,0,1,0], 'sticky': [0,0,0,1]}
        texture_vec = texture_map.get(self.texture_descriptor, [0.25,0.25,0.25,0.25])
        embedding[8:12] = texture_vec
        
        # Vibration spectrum (32 dims)
        if len(self.vibration_spectrum) > 0:
            # Resample to fit remaining space
            spectrum_size = min(32, embedding_dim - 20)
            if len(self.vibration_spectrum) != spectrum_size:
                from scipy import signal
                resampled = signal.resample(self.vibration_spectrum, spectrum_size)
            else:
                resampled = self.vibration_spectrum[:spectrum_size]
            
            # Normalize
            if np.max(np.abs(resampled)) > 0:
                resampled = resampled / np.max(np.abs(resampled))
            embedding[12:12+spectrum_size] = resampled
        
        # Spatial distribution (remaining dims)
        remaining_start = 12 + min(32, embedding_dim - 20)
        if self.contact_points and remaining_start < embedding_dim:
            positions = np.array([cp.position.to_array() for cp in self.contact_points])
            centroid = np.mean(positions, axis=0)
            spread = np.std(positions, axis=0)
            
            # Fit remaining space
            spatial_data = np.concatenate([centroid, spread])
            remaining_space = embedding_dim - remaining_start
            if len(spatial_data) > remaining_space:
                spatial_data = spatial_data[:remaining_space]
            
            embedding[remaining_start:remaining_start+len(spatial_data)] = spatial_data
        
        return embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp,
            'material': self.material,
            'contact_points': [cp.to_dict() for cp in self.contact_points],
            'total_force': self.total_force,
            'contact_area': self.contact_area,
            'pressure': self.pressure,
            'texture_descriptor': self.texture_descriptor,
            'hardness': self.hardness,
            'temperature_feel': self.temperature_feel,
            'vibration_spectrum': self.vibration_spectrum.tolist(),
            'grip_quality': self.grip_quality,
            'deformation': self.deformation,
            'stiffness': self.stiffness
        }


@dataclass
class AudioSignature:
    """Audio signature from material interactions."""
    timestamp: float
    source_position: Vector3D
    frequency_spectrum: np.ndarray
    amplitude: float
    dominant_frequency: float
    harmonics: List[float]
    decay_rate: float
    directivity: Vector3D  # Sound direction
    doppler_shift: float = 0.0
    echo_delay: float = 0.0
    reverberation_time: float = 0.0
    
    def to_embedding(self, embedding_dim: int = 32) -> np.ndarray:
        """Convert to fixed-size embedding vector."""
        embedding = np.zeros(embedding_dim)
        
        # Basic audio properties (8 dims)
        embedding[0] = min(self.amplitude / 100, 1.0)
        embedding[1] = min(self.dominant_frequency / 20000, 1.0)  # 0-20kHz
        embedding[2] = min(len(self.harmonics) / 10, 1.0)
        embedding[3] = min(self.decay_rate, 1.0)
        embedding[4] = self.doppler_shift / 0.1  # ±0.1 max
        embedding[5] = min(self.echo_delay / 0.1, 1.0)  # 0-100ms
        embedding[6] = min(self.reverberation_time / 2.0, 1.0)  # 0-2s
        embedding[7] = min(np.sum(self.harmonics) / 1000, 1.0)
        
        # Frequency spectrum (16 dims)
        if len(self.frequency_spectrum) > 0:
            spectrum_size = min(16, embedding_dim - 16)
            if len(self.frequency_spectrum) != spectrum_size:
                from scipy import signal
                resampled = signal.resample(self.frequency_spectrum, spectrum_size)
            else:
                resampled = self.frequency_spectrum[:spectrum_size]
            
            if np.max(np.abs(resampled)) > 0:
                resampled = resampled / np.max(np.abs(resampled))
            embedding[8:8+spectrum_size] = resampled
        
        # Directivity (remaining dims)
        remaining_start = 8 + min(16, embedding_dim - 16)
        if remaining_start < embedding_dim:
            dir_vec = self.directivity.to_array()
            remaining_space = embedding_dim - remaining_start
            embedding[remaining_start:remaining_start+min(3, remaining_space)] = dir_vec[:min(3, remaining_space)]
        
        return embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp,
            'source_position': self.source_position.to_array().tolist(),
            'frequency_spectrum': self.frequency_spectrum.tolist(),
            'amplitude': self.amplitude,
            'dominant_frequency': self.dominant_frequency,
            'harmonics': self.harmonics,
            'decay_rate': self.decay_rate,
            'directivity': self.directivity.to_array().tolist(),
            'doppler_shift': self.doppler_shift,
            'echo_delay': self.echo_delay,
            'reverberation_time': self.reverberation_time
        }


@dataclass
class VisualSignature:
    """Visual signature from perception."""
    timestamp: float
    features: np.ndarray
    object_positions: List[Vector3D]
    colors: List[List[float]]
    textures: List[str]
    lighting_conditions: Dict[str, float]
    depth_map: Optional[np.ndarray] = None
    
    def to_embedding(self, embedding_dim: int = 128) -> np.ndarray:
        """Convert to fixed-size embedding vector."""
        if len(self.features) >= embedding_dim:
            return self.features[:embedding_dim]
        
        embedding = np.zeros(embedding_dim)
        embedding[:len(self.features)] = self.features
        return embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp,
            'features': self.features.tolist(),
            'object_positions': [pos.to_array().tolist() for pos in self.object_positions],
            'colors': self.colors,
            'textures': self.textures,
            'lighting_conditions': self.lighting_conditions,
            'depth_map': self.depth_map.tolist() if self.depth_map is not None else None
        }


@dataclass
class SensoryExperience:
    """Complete multi-modal sensory experience."""
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    duration_ms: float = 0.0
    
    # Sensory signatures
    tactile: Optional[TactileSignature] = None
    audio: Optional[AudioSignature] = None
    visual: Optional[VisualSignature] = None
    
    # Embeddings
    tactile_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    visual_embedding: Optional[np.ndarray] = None
    fused_embedding: Optional[np.ndarray] = None
    
    # Material and interaction context
    primary_material: Optional[str] = None
    secondary_material: Optional[str] = None
    material_interaction: Optional[Dict[str, Any]] = None
    
    # Action context
    action_type: Optional[ActionType] = None
    action_parameters: Optional[Dict[str, Any]] = None
    
    # Outcome
    success: bool = True
    reward: float = 0.0
    
    # Physics data
    contact_points: Optional[List[ContactPoint]] = None
    forces: Optional[List[float]] = None
    torques: Optional[List[float]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def compute_embeddings(self):
        """Compute embeddings from signatures."""
        if self.tactile:
            self.tactile_embedding = self.tactile.to_embedding()
        if self.audio:
            self.audio_embedding = self.audio.to_embedding()
        if self.visual:
            self.visual_embedding = self.visual.to_embedding()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'experience_id': self.experience_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'tactile_data': self.tactile.to_dict() if self.tactile else None,
            'audio_data': self.audio.to_dict() if self.audio else None,
            'visual_data': self.visual.to_dict() if self.visual else None,
            'tactile_embedding': self.tactile_embedding.tolist() if self.tactile_embedding is not None else None,
            'audio_embedding': self.audio_embedding.tolist() if self.audio_embedding is not None else None,
            'visual_embedding': self.visual_embedding.tolist() if self.visual_embedding is not None else None,
            'fused_embedding': self.fused_embedding.tolist() if self.fused_embedding is not None else None,
            'primary_material': self.primary_material,
            'secondary_material': self.secondary_material,
            'material_interaction': self.material_interaction,
            'action_type': self.action_type.name if self.action_type else None,
            'action_parameters': self.action_parameters,
            'success': self.success,
            'reward': self.reward,
            'contact_points': [cp.to_dict() for cp in self.contact_points] if self.contact_points else None,
            'forces': self.forces,
            'torques': self.torques,
            'tags': self.tags,
            'notes': self.notes
        }


@dataclass
class MaterialInteraction:
    """Properties of interaction between two materials."""
    material1: str
    material2: str
    combined_friction: float
    combined_restitution: float
    effective_modulus: float
    contact_stiffness: float
    expected_sound_frequency: float
    thermal_contrast: float
    grip_prediction: bool
    bounce_prediction: bool
    contact_force: float = 1.0
    relative_velocity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'materials': [self.material1, self.material2],
            'combined_friction': self.combined_friction,
            'combined_restitution': self.combined_restitution,
            'effective_modulus': self.effective_modulus,
            'contact_stiffness': self.contact_stiffness,
            'expected_sound_frequency': self.expected_sound_frequency,
            'thermal_contrast': self.thermal_contrast,
            'grip_prediction': self.grip_prediction,
            'bounce_prediction': self.bounce_prediction,
            'contact_force': self.contact_force,
            'relative_velocity': self.relative_velocity
        }


@dataclass
class DreamSessionConfig:
    """Configuration for dream session."""
    replay_speed: float = 10.0
    variation_factor: float = 0.2
    exploration_rate: float = 0.3
    consolidation_threshold: float = 0.8
    min_improvement: float = 0.1
    max_iterations: int = 1000
    parallel_dreams: int = 4
    time_range_hours: float = 24.0
    max_experiences: int = 5000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'replay_speed': self.replay_speed,
            'variation_factor': self.variation_factor,
            'exploration_rate': self.exploration_rate,
            'consolidation_threshold': self.consolidation_threshold,
            'min_improvement': self.min_improvement,
            'max_iterations': self.max_iterations,
            'parallel_dreams': self.parallel_dreams,
            'time_range_hours': self.time_range_hours,
            'max_experiences': self.max_experiences
        }


@dataclass
class LearnedStrategy:
    """A strategy learned through dream optimization."""
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = "general"  # 'tactile', 'audio', 'motion', 'material'
    strategy_data: Dict[str, Any] = field(default_factory=dict)
    baseline_performance: float = 0.0
    improved_performance: float = 0.0
    improvement_ratio: float = 0.0
    confidence: float = 0.0
    applicable_materials: List[str] = field(default_factory=list)
    applicable_scenarios: List[str] = field(default_factory=list)
    times_used: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'category': self.category,
            'strategy_data': self.strategy_data,
            'baseline_performance': self.baseline_performance,
            'improved_performance': self.improved_performance,
            'improvement_ratio': self.improvement_ratio,
            'confidence': self.confidence,
            'applicable_materials': self.applicable_materials,
            'applicable_scenarios': self.applicable_scenarios,
            'times_used': self.times_used,
            'success_rate': self.success_rate,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None
        }


@dataclass
class PredictionResult:
    """Result of a forward model prediction."""
    predicted_state: np.ndarray
    uncertainty: np.ndarray
    confidence: float
    prediction_horizon: float  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'predicted_state': self.predicted_state.tolist(),
            'uncertainty': self.uncertainty.tolist(),
            'confidence': self.confidence,
            'prediction_horizon': self.prediction_horizon
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    perception_count: int = 0
    dream_count: int = 0
    strategies_learned: int = 0
    database_size: int = 0
    average_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'perception_count': self.perception_count,
            'dream_count': self.dream_count,
            'strategies_learned': self.strategies_learned,
            'database_size': self.database_size,
            'average_processing_time': self.average_processing_time,
            'memory_usage_mb': self.memory_usage_mb,
            'success_rate': self.success_rate
        }


# Type aliases for convenience
Embedding = np.ndarray
Timestamp = float
SessionId = str
ExperienceId = str
MaterialName = str