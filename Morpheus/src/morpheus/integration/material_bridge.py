"""Material bridge for GASM-Robotics integration.

Provides seamless integration with GASM-Robotics material definitions
and computes tactile and audio signatures from material properties.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import lru_cache
from scipy import signal

from ..core.types import (
    MaterialProperties, MaterialType, TactileSignature, AudioSignature,
    MaterialInteraction, Vector3D, ContactPoint
)

logger = logging.getLogger(__name__)


class MaterialBridge:
    """Bridge between GASM-Robotics materials and MORPHEUS perception."""
    
    def __init__(self, gasm_robotics_path: Union[str, Path]):
        """Initialize material bridge.
        
        Args:
            gasm_robotics_path: Path to GASM-Robotics directory
            
        Raises:
            FileNotFoundError: If GASM config file not found
            ValueError: If material configuration is invalid
        """
        self.gasm_path = Path(gasm_robotics_path)
        self.config_path = self.gasm_path / "assets/configs/simulation_params.yaml"
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"GASM config not found at {self.config_path}")
            
        self.materials = self._load_materials()
        self.interaction_cache = {}  # Cache computed interactions
        
        # Material type mappings
        self._type_mappings = {
            'steel': MaterialType.METAL,
            'aluminum': MaterialType.METAL,
            'plastic': MaterialType.PLASTIC,
            'rubber': MaterialType.RUBBER,
            'glass': MaterialType.GLASS
        }
        
        logger.info(f"MaterialBridge loaded {len(self.materials)} materials: {list(self.materials.keys())}")
        
    def _load_materials(self) -> Dict[str, MaterialProperties]:
        """Load material definitions from GASM configuration.
        
        Returns:
            Dictionary mapping material names to MaterialProperties
            
        Raises:
            ValueError: If material configuration is invalid
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load GASM config: {e}")
            
        materials = {}
        material_configs = config.get('materials', {})
        
        if not material_configs:
            logger.warning("No materials found in GASM config")
            
        for name, props in material_configs.items():
            try:
                # Validate required properties
                required_props = ['color', 'friction', 'restitution', 'density', 'young_modulus', 'poisson_ratio']
                for prop in required_props:
                    if prop not in props:
                        raise ValueError(f"Missing required property '{prop}' for material '{name}'")
                
                # Create MaterialProperties instance
                material_type = self._type_mappings.get(name.lower(), MaterialType.UNKNOWN)
                
                materials[name] = MaterialProperties(
                    name=name,
                    material_type=material_type,
                    color=props['color'],
                    friction=float(props['friction']),
                    restitution=float(props['restitution']),
                    density=float(props['density']),
                    young_modulus=float(props['young_modulus']),
                    poisson_ratio=float(props['poisson_ratio'])
                )
                
                logger.debug(f"Loaded material '{name}': {materials[name]}")
                
            except Exception as e:
                logger.error(f"Failed to load material '{name}': {e}")
                continue
                
        # Add default material if not exists
        if 'default' not in materials:
            materials['default'] = MaterialProperties(
                name='default',
                material_type=MaterialType.COMPOSITE,
                color=[0.5, 0.5, 0.5, 1.0],
                friction=0.5,
                restitution=0.5,
                density=1000,
                young_modulus=1e9,
                poisson_ratio=0.3
            )
            logger.info("Added default material")
            
        return materials
    
    def get_material(self, name: str) -> Optional[MaterialProperties]:
        """Get material properties by name.
        
        Args:
            name: Material name
            
        Returns:
            MaterialProperties or None if not found
        """
        material = self.materials.get(name)
        if material is None:
            logger.warning(f"Material '{name}' not found, using default")
            material = self.materials.get('default')
        return material
    
    def list_materials(self) -> List[str]:
        """Get list of available material names.
        
        Returns:
            List of material names
        """
        return list(self.materials.keys())
    
    def get_material_type(self, name: str) -> MaterialType:
        """Get material type for a given material name.
        
        Args:
            name: Material name
            
        Returns:
            MaterialType enum value
        """
        material = self.get_material(name)
        return material.material_type if material else MaterialType.UNKNOWN
    
    @lru_cache(maxsize=1000)
    def compute_tactile_signature(self, 
                                material_name: str,
                                contact_force: float = 1.0,
                                contact_velocity: float = 0.0,
                                contact_area: float = 0.001) -> Dict[str, float]:
        """Compute expected tactile signature from material properties.
        
        Args:
            material_name: Name of the material
            contact_force: Applied force in Newtons
            contact_velocity: Relative velocity in m/s
            contact_area: Contact area in m²
            
        Returns:
            Dictionary with tactile parameters
        """
        material = self.get_material(material_name)
        if not material:
            logger.warning(f"Unknown material '{material_name}', using default tactile signature")
            return self._default_tactile_signature()
        
        # Hardness from Young's modulus (logarithmic perception)
        hardness = np.clip(np.log10(material.young_modulus + 1) / 12, 0, 1)
        
        # Deformation under load (simplified Hertz contact)
        # F = (4/3) * E' * sqrt(R) * δ^(3/2), simplified as δ = F / (k * A)
        effective_modulus = material.young_modulus / (1 - material.poisson_ratio**2)
        contact_stiffness = effective_modulus * contact_area
        deformation = contact_force / (contact_stiffness + 1e-10)  # mm
        
        # Pressure calculation
        pressure = contact_force / (contact_area + 1e-10)  # Pa
        
        # Texture from friction and surface properties
        if contact_velocity < 0.001:  # Static contact
            texture_roughness = material.friction * 0.5
            texture_descriptor = 'smooth' if texture_roughness < 0.3 else 'textured'
        else:  # Sliding contact
            # Stick-slip frequency based on velocity and friction
            stick_slip_freq = contact_velocity * material.friction * 100  # Hz
            texture_roughness = material.friction * (1 + np.sin(stick_slip_freq) * 0.2)
            if texture_roughness < 0.3:
                texture_descriptor = 'smooth'
            elif texture_roughness < 0.7:
                texture_descriptor = 'textured'
            else:
                texture_descriptor = 'rough'
        
        # Temperature perception based on thermal conductivity
        if material.young_modulus > 100e9:  # Metal-like
            thermal_feel = 0.3  # Feels cool
        elif material.young_modulus < 10e9:  # Polymer/rubber
            thermal_feel = 0.7  # Feels warm
        else:
            thermal_feel = 0.5  # Neutral
        
        # Vibration damping from material properties
        vibration_damping = material.poisson_ratio * 2
        
        # Grip quality from friction and surface properties
        grip_quality = np.clip(material.friction / 1.5, 0, 1)
        
        # Stiffness perception
        stiffness = contact_stiffness / 1e6  # Normalized
        
        return {
            'hardness': hardness,
            'deformation_mm': deformation * 1000,  # Convert to mm
            'texture_roughness': texture_roughness,
            'texture_descriptor': texture_descriptor,
            'thermal_feel': thermal_feel,
            'vibration_damping': vibration_damping,
            'grip_quality': grip_quality,
            'elasticity': material.restitution,
            'weight_density': material.density / 10000,  # Normalized
            'pressure': pressure,
            'stiffness': stiffness
        }
    
    @lru_cache(maxsize=1000)
    def compute_audio_signature(self, 
                              material_name: str,
                              impact_velocity: float = 1.0,
                              object_size: float = 0.1) -> Dict[str, Union[float, List[float]]]:
        """Compute expected audio signature from material impact.
        
        Args:
            material_name: Name of the material
            impact_velocity: Impact velocity in m/s
            object_size: Characteristic size of object in m
            
        Returns:
            Dictionary with audio parameters
        """
        material = self.get_material(material_name)
        if not material:
            logger.warning(f"Unknown material '{material_name}', using default audio signature")
            return self._default_audio_signature()
        
        # Sound speed in material
        sound_speed = np.sqrt(material.young_modulus / material.density)
        
        # Fundamental frequency from object size and sound speed
        # For a bar: f = (n * c) / (2 * L), simplified
        fundamental_freq = sound_speed / (4 * object_size)
        fundamental_freq = np.clip(fundamental_freq, 20, 20000)  # Audible range
        
        # Amplitude from impact energy (kinetic energy -> sound energy)
        # Simplified: amplitude proportional to velocity and material stiffness
        amplitude = (1 - material.restitution) * impact_velocity * material.density / 10000
        amplitude = np.clip(amplitude, 0, 100)
        
        # Decay rate from internal damping
        # Higher Poisson ratio and lower stiffness = more damping
        decay_rate = material.poisson_ratio + (1 / (material.young_modulus / 1e9 + 1))
        decay_rate = np.clip(decay_rate, 0.01, 2.0)
        
        # Harmonics based on material type
        if material.material_type == MaterialType.METAL:
            harmonics = [1, 2, 3, 5, 7, 11]  # Rich harmonic content
            brightness = 0.8
        elif material.material_type == MaterialType.GLASS:
            harmonics = [1, 2, 4, 8]  # Clear, crystalline harmonics
            brightness = 0.9
        elif material.material_type == MaterialType.PLASTIC:
            harmonics = [1, 2, 3]  # Limited harmonics
            brightness = 0.4
        elif material.material_type == MaterialType.RUBBER:
            harmonics = [1]  # Mostly fundamental, very damped
            brightness = 0.1
        else:  # Composite or unknown
            harmonics = [1, 2, 3]
            brightness = 0.5
        
        # Generate harmonic frequencies
        harmonic_freqs = [fundamental_freq * h for h in harmonics]
        harmonic_freqs = [f for f in harmonic_freqs if f <= 20000]  # Filter audible
        
        return {
            'fundamental_freq': fundamental_freq,
            'amplitude': amplitude,
            'decay_rate': decay_rate,
            'harmonics': harmonic_freqs,
            'brightness': brightness,
            'sound_speed': sound_speed,
            'spectral_centroid': fundamental_freq * (1 + brightness)
        }
    
    @lru_cache(maxsize=10000)
    def compute_interaction(self, 
                          material1: str, 
                          material2: str,
                          contact_force: float = 1.0,
                          relative_velocity: float = 0.0) -> MaterialInteraction:
        """Compute interaction properties between two materials.
        
        Args:
            material1: First material name
            material2: Second material name
            contact_force: Contact force in Newtons
            relative_velocity: Relative velocity in m/s
            
        Returns:
            MaterialInteraction object with combined properties
        """
        mat1 = self.get_material(material1)
        mat2 = self.get_material(material2)
        
        if not mat1 or not mat2:
            logger.warning(f"Material not found: {material1} or {material2}")
            # Return default interaction
            return MaterialInteraction(
                material1=material1,
                material2=material2,
                combined_friction=0.5,
                combined_restitution=0.5,
                effective_modulus=1e9,
                contact_stiffness=1e6,
                expected_sound_frequency=1000,
                thermal_contrast=0.0,
                grip_prediction=False,
                bounce_prediction=False,
                contact_force=contact_force,
                relative_velocity=relative_velocity
            )
        
        # Combined friction (geometric mean for dissimilar materials)
        combined_friction = np.sqrt(mat1.friction * mat2.friction)
        
        # Combined restitution (minimum - energy loss dominates)
        combined_restitution = min(mat1.restitution, mat2.restitution)
        
        # Effective Young's modulus (harmonic mean for contact mechanics)
        effective_modulus = 2 * mat1.young_modulus * mat2.young_modulus / (
            mat1.young_modulus + mat2.young_modulus + 1e-10
        )
        
        # Contact stiffness estimation
        contact_stiffness = effective_modulus * 0.001  # Simplified, depends on geometry
        
        # Expected contact sound frequency
        average_density = (mat1.density + mat2.density) / 2
        sound_freq = np.sqrt(effective_modulus / average_density) / 10  # Simplified
        sound_freq = np.clip(sound_freq, 20, 20000)
        
        # Thermal conductivity difference (affects perceived temperature)
        thermal_contrast = abs(mat1.thermal_conductivity - mat2.thermal_conductivity) / 200
        thermal_contrast = np.clip(thermal_contrast, 0, 1)
        
        # Grip prediction based on friction and surface compatibility
        grip_prediction = combined_friction > 0.7 and abs(mat1.young_modulus - mat2.young_modulus) < 100e9
        
        # Bounce prediction
        bounce_prediction = combined_restitution > 0.5
        
        return MaterialInteraction(
            material1=material1,
            material2=material2,
            combined_friction=combined_friction,
            combined_restitution=combined_restitution,
            effective_modulus=effective_modulus,
            contact_stiffness=contact_stiffness,
            expected_sound_frequency=sound_freq,
            thermal_contrast=thermal_contrast,
            grip_prediction=grip_prediction,
            bounce_prediction=bounce_prediction,
            contact_force=contact_force,
            relative_velocity=relative_velocity
        )
    
    def predict_sensory_outcome(self,
                               scenario: Dict[str, Union[str, float, List[float]]]) -> Dict[str, Union[str, Dict, float]]:
        """Predict complete sensory outcome for a scenario.
        
        Args:
            scenario: Dictionary with materials, forces, velocities, etc.
            
        Returns:
            Predicted sensory signatures with confidence
        """
        material = scenario.get('material', 'default')
        if isinstance(material, list) and len(material) > 0:
            material = material[0]
        elif not isinstance(material, str):
            material = 'default'
            
        mat_props = self.get_material(material)
        
        if not mat_props:
            return {
                'material': material,
                'error': 'Material not found',
                'confidence': 0.0
            }
            
        # Extract scenario parameters with defaults
        force = scenario.get('force', 1.0)
        velocity = scenario.get('velocity', 0.0)
        impact_velocity = scenario.get('impact_velocity', 0.5)
        contact_area = scenario.get('contact_area', 0.001)
        object_size = scenario.get('object_size', 0.1)
        
        # Tactile prediction
        tactile = self.compute_tactile_signature(
            material_name=material,
            contact_force=float(force),
            contact_velocity=float(velocity),
            contact_area=float(contact_area)
        )
        
        # Audio prediction
        audio = self.compute_audio_signature(
            material_name=material,
            impact_velocity=float(impact_velocity),
            object_size=float(object_size)
        )
        
        # Confidence based on material knowledge and scenario completeness
        base_confidence = 0.8 if material != 'default' else 0.5
        scenario_completeness = len([k for k in ['force', 'velocity', 'impact_velocity'] if k in scenario]) / 3
        confidence = base_confidence * (0.5 + 0.5 * scenario_completeness)
        
        return {
            'material': material,
            'material_type': mat_props.material_type.name,
            'tactile': tactile,
            'audio': audio,
            'confidence': confidence,
            'scenario_used': {
                'force': force,
                'velocity': velocity,
                'impact_velocity': impact_velocity,
                'contact_area': contact_area,
                'object_size': object_size
            }
        }
    
    def generate_vibration_spectrum(self,
                                  material_name: str,
                                  force_history: List[float],
                                  sampling_rate: float = 1000.0) -> np.ndarray:
        """Generate vibration spectrum from force history and material properties.
        
        Args:
            material_name: Name of the material
            force_history: Time series of forces
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Frequency spectrum as numpy array
        """
        if not force_history or len(force_history) < 10:
            return np.zeros(32)
            
        material = self.get_material(material_name)
        if not material:
            return np.zeros(32)
        
        # Convert to numpy array and remove DC component
        forces = np.array(force_history)
        forces = forces - np.mean(forces)
        
        # Apply window to reduce spectral leakage
        if len(forces) > 1:
            window = signal.hann(len(forces))
            forces = forces * window
        
        # Compute FFT
        fft = np.fft.rfft(forces)
        freqs = np.fft.rfftfreq(len(forces), 1/sampling_rate)
        
        # Material-specific frequency response
        # Higher stiffness -> higher frequency content
        # Higher damping -> more attenuation at high frequencies
        stiffness_factor = material.young_modulus / 200e9  # Normalized
        damping_factor = material.poisson_ratio
        
        # Apply material transfer function
        material_response = np.exp(-damping_factor * freqs / 1000) * (1 + stiffness_factor)
        fft_modified = fft * material_response[:len(fft)]
        
        # Bin into 32 frequency bands (log scale for perception)
        freq_bins = np.logspace(np.log10(1), np.log10(sampling_rate/2), 33)
        binned_spectrum = np.zeros(32)
        
        for i in range(32):
            mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i+1])
            if np.any(mask):
                binned_spectrum[i] = np.mean(np.abs(fft_modified[mask]))
                
        # Normalize
        if np.max(binned_spectrum) > 0:
            binned_spectrum = binned_spectrum / np.max(binned_spectrum)
            
        return binned_spectrum
    
    def _default_tactile_signature(self) -> Dict[str, float]:
        """Return default tactile signature."""
        return {
            'hardness': 0.5,
            'deformation_mm': 1.0,
            'texture_roughness': 0.5,
            'texture_descriptor': 'smooth',
            'thermal_feel': 0.5,
            'vibration_damping': 0.5,
            'grip_quality': 0.5,
            'elasticity': 0.5,
            'weight_density': 0.1,
            'pressure': 1000.0,
            'stiffness': 0.5
        }
    
    def _default_audio_signature(self) -> Dict[str, Union[float, List[float]]]:
        """Return default audio signature."""
        return {
            'fundamental_freq': 1000.0,
            'amplitude': 10.0,
            'decay_rate': 1.0,
            'harmonics': [1000.0, 2000.0],
            'brightness': 0.5,
            'sound_speed': 1000.0,
            'spectral_centroid': 1500.0
        }
    
    def clear_cache(self):
        """Clear interaction cache."""
        self.interaction_cache.clear()
        # Clear LRU caches
        self.compute_tactile_signature.cache_clear()
        self.compute_audio_signature.cache_clear()
        self.compute_interaction.cache_clear()
        logger.info("Material bridge caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'tactile_signature': {
                'hits': self.compute_tactile_signature.cache_info().hits,
                'misses': self.compute_tactile_signature.cache_info().misses,
                'size': self.compute_tactile_signature.cache_info().currsize
            },
            'audio_signature': {
                'hits': self.compute_audio_signature.cache_info().hits,
                'misses': self.compute_audio_signature.cache_info().misses,
                'size': self.compute_audio_signature.cache_info().currsize
            },
            'interaction': {
                'hits': self.compute_interaction.cache_info().hits,
                'misses': self.compute_interaction.cache_info().misses,
                'size': self.compute_interaction.cache_info().currsize
            }
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"MaterialBridge(materials={len(self.materials)}, path={self.gasm_path})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MaterialBridge(gasm_path='{self.gasm_path}', materials={list(self.materials.keys())})"