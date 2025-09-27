"""Advanced tactile processing with material-aware perception.

Provides sophisticated tactile processing capabilities that integrate
with PyBullet physics simulation and GASM-Robotics material properties
to generate rich tactile signatures and embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import cdist
import logging
from collections import deque
import time

# PyBullet import with fallback
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyBullet not available. Using simulation mode.")

from ..core.types import (
    ContactPoint, TactileSignature, Vector3D, MaterialType
)
from ..integration.material_bridge import MaterialBridge

logger = logging.getLogger(__name__)


@dataclass
class TactileProcessorConfig:
    """Configuration for tactile processor."""
    sensitivity: float = 0.01  # Minimum force in Newtons
    sampling_rate: int = 1000  # Hz
    vibration_window: float = 0.1  # seconds
    max_contact_points: int = 50
    embedding_dim: int = 64
    use_materials: bool = True
    force_threshold: float = 0.001  # N
    contact_area_threshold: float = 1e-6  # m²
    enable_vibration_analysis: bool = True
    enable_texture_classification: bool = True
    spatial_resolution: float = 0.001  # m


class ContactAnalyzer:
    """Analyzes contact points for spatial and temporal patterns."""
    
    def __init__(self, config: TactileProcessorConfig):
        self.config = config
        self.contact_history = deque(maxlen=100)
    
    def analyze_contact_distribution(self, contacts: List[ContactPoint]) -> Dict[str, float]:
        """Analyze spatial distribution of contact points.
        
        Args:
            contacts: List of contact points
            
        Returns:
            Dictionary with spatial distribution metrics
        """
        if not contacts:
            return {'centroid': [0, 0, 0], 'spread': 0, 'cluster_count': 0, 'area_estimate': 0}
        
        # Extract positions
        positions = np.array([cp.position.to_array() for cp in contacts])
        forces = np.array([cp.force_magnitude for cp in contacts])
        
        # Weighted centroid
        total_force = np.sum(forces)
        if total_force > 0:
            centroid = np.average(positions, weights=forces, axis=0)
        else:
            centroid = np.mean(positions, axis=0)
        
        # Spread (weighted standard deviation)
        if len(positions) > 1:
            distances = np.linalg.norm(positions - centroid, axis=1)
            if total_force > 0:
                spread = np.sqrt(np.average(distances**2, weights=forces))
            else:
                spread = np.std(distances)
        else:
            spread = 0
        
        # Cluster analysis (simple density-based)
        cluster_count = self._estimate_clusters(positions)
        
        # Contact area estimation
        area_estimate = self._estimate_contact_area(positions, forces)
        
        return {
            'centroid': centroid.tolist(),
            'spread': spread,
            'cluster_count': cluster_count,
            'area_estimate': area_estimate,
            'contact_density': len(contacts) / (area_estimate + 1e-10)
        }
    
    def _estimate_clusters(self, positions: np.ndarray, threshold: float = 0.01) -> int:
        """Estimate number of contact clusters."""
        if len(positions) < 2:
            return len(positions)
        
        # Simple clustering based on distance threshold
        distances = cdist(positions, positions)
        clusters = 0
        visited = set()
        
        for i in range(len(positions)):
            if i not in visited:
                # Start new cluster
                clusters += 1
                cluster_points = [i]
                visited.add(i)
                
                # Find all points within threshold
                j = 0
                while j < len(cluster_points):
                    current = cluster_points[j]
                    for k in range(len(positions)):
                        if k not in visited and distances[current, k] < threshold:
                            cluster_points.append(k)
                            visited.add(k)
                    j += 1
        
        return clusters
    
    def _estimate_contact_area(self, positions: np.ndarray, forces: np.ndarray) -> float:
        """Estimate total contact area from contact points."""
        if len(positions) == 0:
            return 0
        
        if len(positions) == 1:
            # Single point - estimate based on force (Hertz contact)
            return max(0.001 * np.sqrt(forces[0]), 1e-6)
        
        # Multiple points - convex hull approximation
        try:
            from scipy.spatial import ConvexHull
            
            # Project to 2D (remove dimension with least variance)
            variances = np.var(positions, axis=0)
            keep_dims = np.argsort(variances)[1:]  # Keep 2 largest variance dims
            positions_2d = positions[:, keep_dims]
            
            if len(positions_2d) >= 3:
                hull = ConvexHull(positions_2d)
                area = hull.volume  # In 2D, volume is area
                return max(area, 1e-6)
            else:
                # Line segment - estimate width
                length = np.linalg.norm(positions_2d[-1] - positions_2d[0])
                return max(length * 0.001, 1e-6)  # Assume 1mm width
                
        except ImportError:
            # Fallback to bounding box
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            dimensions = max_pos - min_pos
            
            # Remove smallest dimension and compute area
            sorted_dims = np.sort(dimensions)
            area = sorted_dims[1] * sorted_dims[2]  # Two largest dimensions
            return max(area, 1e-6)
        
        except Exception:
            # Final fallback
            return 0.001


class VibrationAnalyzer:
    """Analyzes vibration patterns from force history."""
    
    def __init__(self, config: TactileProcessorConfig):
        self.config = config
        self.force_history = deque(maxlen=int(config.sampling_rate * config.vibration_window))
        self.time_history = deque(maxlen=int(config.sampling_rate * config.vibration_window))
    
    def add_force_sample(self, force: float, timestamp: float):
        """Add force sample to history.
        
        Args:
            force: Force magnitude in Newtons
            timestamp: Timestamp of measurement
        """
        self.force_history.append(force)
        self.time_history.append(timestamp)
    
    def analyze_vibration(self) -> np.ndarray:
        """Analyze vibration from force history.
        
        Returns:
            Frequency spectrum as numpy array (32 bins)
        """
        if len(self.force_history) < 10:
            return np.zeros(32)
        
        # Convert to numpy arrays
        forces = np.array(self.force_history)
        times = np.array(self.time_history)
        
        # Check for consistent sampling
        dt = np.mean(np.diff(times)) if len(times) > 1 else 1.0 / self.config.sampling_rate
        actual_rate = 1.0 / dt if dt > 0 else self.config.sampling_rate
        
        # Remove DC component and trends
        forces_detrended = signal.detrend(forces)
        
        # Apply window to reduce spectral leakage
        if len(forces_detrended) > 1:
            window = signal.hann(len(forces_detrended))
            forces_windowed = forces_detrended * window
        else:
            forces_windowed = forces_detrended
        
        # Compute FFT
        fft = np.fft.rfft(forces_windowed)
        freqs = np.fft.rfftfreq(len(forces_windowed), 1/actual_rate)
        
        # Bin into 32 frequency bands (log scale for perception)
        max_freq = min(actual_rate / 2, 1000)  # Up to 1kHz
        freq_bins = np.logspace(np.log10(1), np.log10(max_freq), 33)
        binned_spectrum = np.zeros(32)
        
        for i in range(32):
            mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i+1])
            if np.any(mask):
                binned_spectrum[i] = np.mean(np.abs(fft[mask]))
        
        # Normalize
        max_magnitude = np.max(binned_spectrum)
        if max_magnitude > 0:
            binned_spectrum = binned_spectrum / max_magnitude
        
        return binned_spectrum
    
    def get_vibration_features(self) -> Dict[str, float]:
        """Extract vibration features.
        
        Returns:
            Dictionary with vibration characteristics
        """
        spectrum = self.analyze_vibration()
        
        if np.sum(spectrum) == 0:
            return {
                'spectral_centroid': 0,
                'spectral_rolloff': 0,
                'spectral_spread': 0,
                'dominant_frequency_bin': 0,
                'total_energy': 0
            }
        
        # Spectral centroid (brightness)
        freqs = np.arange(32)
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        
        # Spectral rolloff (85% energy threshold)
        cumsum = np.cumsum(spectrum)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        spectral_rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else 31
        
        # Spectral spread (variance around centroid)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / np.sum(spectrum))
        
        # Dominant frequency
        dominant_frequency_bin = np.argmax(spectrum)
        
        # Total energy
        total_energy = np.sum(spectrum ** 2)
        
        return {
            'spectral_centroid': spectral_centroid / 32,  # Normalized
            'spectral_rolloff': spectral_rolloff / 32,
            'spectral_spread': spectral_spread / 32,
            'dominant_frequency_bin': dominant_frequency_bin,
            'total_energy': total_energy
        }


class TextureClassifier:
    """Classifies surface texture from contact patterns."""
    
    def __init__(self, config: TactileProcessorConfig):
        self.config = config
    
    def classify_texture(self, 
                        vibration_features: Dict[str, float],
                        material_roughness: float,
                        contact_pattern: Dict[str, Any]) -> str:
        """Classify texture based on multiple cues.
        
        Args:
            vibration_features: Features from vibration analysis
            material_roughness: Material-based roughness estimate
            contact_pattern: Spatial contact pattern analysis
            
        Returns:
            Texture classification string
        """
        # Extract relevant features
        high_freq_energy = vibration_features.get('spectral_rolloff', 0)
        vibration_energy = vibration_features.get('total_energy', 0)
        contact_density = contact_pattern.get('contact_density', 0)
        cluster_count = contact_pattern.get('cluster_count', 1)
        
        # Texture classification rules
        if material_roughness < 0.2 and high_freq_energy < 0.3 and vibration_energy < 0.1:
            return 'smooth'
        elif material_roughness > 0.8 or high_freq_energy > 0.7 or vibration_energy > 0.8:
            return 'rough'
        elif vibration_energy > 0.3 and cluster_count > 1:
            return 'textured'
        elif contact_density > 10 and material_roughness > 0.6:
            return 'sticky'
        elif material_roughness > 0.4:
            return 'textured'
        else:
            return 'smooth'
    
    def get_texture_confidence(self, 
                             texture: str,
                             vibration_features: Dict[str, float],
                             material_roughness: float) -> float:
        """Calculate confidence in texture classification.
        
        Args:
            texture: Classified texture
            vibration_features: Vibration features used
            material_roughness: Material roughness
            
        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.6
        
        # Confidence modifiers based on feature consistency
        if texture == 'smooth':
            if material_roughness < 0.3 and vibration_features.get('total_energy', 0) < 0.2:
                base_confidence += 0.3
        elif texture == 'rough':
            if material_roughness > 0.7 and vibration_features.get('total_energy', 0) > 0.5:
                base_confidence += 0.3
        elif texture == 'textured':
            if 0.3 < material_roughness < 0.7 and vibration_features.get('spectral_spread', 0) > 0.3:
                base_confidence += 0.2
        
        return min(base_confidence, 1.0)


class TactileProcessor:
    """Advanced tactile processor with material-aware perception."""
    
    def __init__(self, 
                config: TactileProcessorConfig,
                material_bridge: MaterialBridge):
        """Initialize tactile processor.
        
        Args:
            config: Processor configuration
            material_bridge: Bridge to GASM materials
        """
        self.config = config
        self.material_bridge = material_bridge
        
        # Analysis components
        self.contact_analyzer = ContactAnalyzer(config)
        self.vibration_analyzer = VibrationAnalyzer(config)
        self.texture_classifier = TextureClassifier(config)
        
        # Processing statistics
        self.processing_count = 0
        self.total_processing_time = 0
        
        logger.info(f"TactileProcessor initialized with config: {config}")
        logger.info(f"PyBullet available: {PYBULLET_AVAILABLE}")
    
    def process_contacts(self, 
                        body_id: Optional[int] = None,
                        contact_points: Optional[List[Dict[str, Any]]] = None,
                        material_name: Optional[str] = None,
                        timestamp: Optional[float] = None) -> Optional[TactileSignature]:
        """Process contact points into tactile signature.
        
        Args:
            body_id: PyBullet body ID (if using PyBullet)
            contact_points: Manual contact points (if not using PyBullet)
            material_name: Material name if known
            timestamp: Timestamp of measurement
            
        Returns:
            TactileSignature or None if no contacts
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Get contact points from PyBullet or manual input
        if body_id is not None and PYBULLET_AVAILABLE:
            raw_contacts = self._get_pybullet_contacts(body_id)
        elif contact_points is not None:
            raw_contacts = self._parse_manual_contacts(contact_points)
        else:
            logger.warning("No contact data provided")
            return None
        
        # Filter contacts by sensitivity
        filtered_contacts = [
            cp for cp in raw_contacts 
            if cp.force_magnitude > self.config.sensitivity
        ]
        
        if not filtered_contacts:
            return None
        
        # Limit number of contact points
        if len(filtered_contacts) > self.config.max_contact_points:
            # Keep the strongest contacts
            filtered_contacts.sort(key=lambda cp: cp.force_magnitude, reverse=True)
            filtered_contacts = filtered_contacts[:self.config.max_contact_points]
        
        # Analyze contacts
        total_force = sum(cp.force_magnitude for cp in filtered_contacts)
        contact_distribution = self.contact_analyzer.analyze_contact_distribution(filtered_contacts)
        contact_area = contact_distribution['area_estimate']
        pressure = total_force / (contact_area + 1e-10)
        
        # Add force sample to vibration analyzer
        self.vibration_analyzer.add_force_sample(total_force, timestamp)
        
        # Get material properties
        if material_name and self.config.use_materials:
            material_props = self.material_bridge.get_material(material_name)
            if material_props:
                # Compute material-based tactile signature
                material_tactile = self.material_bridge.compute_tactile_signature(
                    material_name=material_name,
                    contact_force=total_force,
                    contact_velocity=self._estimate_contact_velocity(),
                    contact_area=contact_area
                )
            else:
                material_tactile = self._default_material_tactile()
        else:
            material_tactile = self._default_material_tactile()
            material_name = material_name or 'unknown'
        
        # Vibration analysis
        vibration_spectrum = np.zeros(32)
        vibration_features = {}
        
        if self.config.enable_vibration_analysis:
            vibration_spectrum = self.vibration_analyzer.analyze_vibration()
            vibration_features = self.vibration_analyzer.get_vibration_features()
        
        # Texture classification
        texture_descriptor = 'smooth'
        if self.config.enable_texture_classification:
            texture_descriptor = self.texture_classifier.classify_texture(
                vibration_features,
                material_tactile.get('texture_roughness', 0.5),
                contact_distribution
            )
        
        # Create tactile signature
        signature = TactileSignature(
            timestamp=timestamp,
            material=material_name,
            contact_points=filtered_contacts,
            total_force=total_force,
            contact_area=contact_area,
            pressure=pressure,
            texture_descriptor=texture_descriptor,
            hardness=material_tactile.get('hardness', 0.5),
            temperature_feel=material_tactile.get('thermal_feel', 0.5) * 40,  # 0-40°C
            vibration_spectrum=vibration_spectrum,
            grip_quality=material_tactile.get('grip_quality', 0.5),
            deformation=material_tactile.get('deformation_mm', 1.0),
            stiffness=material_tactile.get('stiffness', 0.5)
        )
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processing_count += 1
        self.total_processing_time += processing_time
        
        logger.debug(f"Processed tactile signature: {len(filtered_contacts)} contacts, "
                    f"{total_force:.2f}N, {texture_descriptor} texture")
        
        return signature
    
    def _get_pybullet_contacts(self, body_id: int) -> List[ContactPoint]:
        """Get contact points from PyBullet.
        
        Args:
            body_id: PyBullet body ID
            
        Returns:
            List of ContactPoint objects
        """
        if not PYBULLET_AVAILABLE:
            return []
        
        try:
            contact_points = p.getContactPoints(bodyA=body_id)
            parsed_contacts = []
            
            for cp in contact_points:
                contact = ContactPoint(
                    position=Vector3D.from_array(np.array(cp[5])),  # positionOnA
                    normal=Vector3D.from_array(np.array(cp[7])),    # contactNormalOnB
                    force_magnitude=cp[9],  # normalForce
                    object_a=cp[1],
                    object_b=cp[2],
                    link_a=cp[3],
                    link_b=cp[4],
                    friction_force=Vector3D.from_array(np.array(cp[10]) if len(cp) > 10 else np.zeros(3))
                )
                
                if contact.force_magnitude > self.config.force_threshold:
                    parsed_contacts.append(contact)
            
            return parsed_contacts
            
        except Exception as e:
            logger.error(f"Error getting PyBullet contacts: {e}")
            return []
    
    def _parse_manual_contacts(self, contact_data: List[Dict[str, Any]]) -> List[ContactPoint]:
        """Parse manually provided contact data.
        
        Args:
            contact_data: List of contact dictionaries
            
        Returns:
            List of ContactPoint objects
        """
        parsed_contacts = []
        
        for cp_data in contact_data:
            try:
                contact = ContactPoint(
                    position=Vector3D.from_array(np.array(cp_data.get('position', [0, 0, 0]))),
                    normal=Vector3D.from_array(np.array(cp_data.get('normal', [0, 0, 1]))),
                    force_magnitude=float(cp_data.get('force', 0)),
                    object_a=cp_data.get('object_a', -1),
                    object_b=cp_data.get('object_b', -1),
                    link_a=cp_data.get('link_a', -1),
                    link_b=cp_data.get('link_b', -1)
                )
                
                if contact.force_magnitude > self.config.force_threshold:
                    parsed_contacts.append(contact)
                    
            except Exception as e:
                logger.warning(f"Failed to parse contact data: {e}")
                continue
        
        return parsed_contacts
    
    def _estimate_contact_velocity(self) -> float:
        """Estimate contact velocity from force history.
        
        Returns:
            Estimated velocity in m/s
        """
        if len(self.vibration_analyzer.force_history) < 2:
            return 0.0
        
        # Simple derivative estimation
        forces = np.array(list(self.vibration_analyzer.force_history))
        times = np.array(list(self.vibration_analyzer.time_history))
        
        if len(forces) > 10:
            # Use last 10 samples for velocity estimation
            force_derivative = np.diff(forces[-10:])
            time_derivative = np.diff(times[-10:])
            
            # Avoid division by zero
            valid_dt = time_derivative > 1e-6
            if np.any(valid_dt):
                velocity_estimates = force_derivative[valid_dt] / time_derivative[valid_dt]
                velocity_estimate = np.mean(np.abs(velocity_estimates)) * 0.01  # Scaling
                return min(velocity_estimate, 10.0)  # Cap at 10 m/s
        
        return 0.0
    
    def _default_material_tactile(self) -> Dict[str, float]:
        """Return default tactile properties."""
        return {
            'hardness': 0.5,
            'deformation_mm': 1.0,
            'texture_roughness': 0.5,
            'thermal_feel': 0.5,
            'grip_quality': 0.5,
            'stiffness': 0.5
        }
    
    def reset(self):
        """Reset processor state."""
        self.vibration_analyzer.force_history.clear()
        self.vibration_analyzer.time_history.clear()
        self.contact_analyzer.contact_history.clear()
        
        logger.debug("TactileProcessor reset")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        avg_time = self.total_processing_time / max(self.processing_count, 1)
        
        return {
            'processing_count': self.processing_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'vibration_buffer_size': len(self.vibration_analyzer.force_history),
            'contact_history_size': len(self.contact_analyzer.contact_history)
        }
    
    def calibrate_sensitivity(self, 
                            test_contacts: List[Dict[str, Any]],
                            expected_detections: List[bool]) -> float:
        """Calibrate sensitivity threshold based on test data.
        
        Args:
            test_contacts: List of test contact configurations
            expected_detections: Expected detection results (True/False)
            
        Returns:
            Optimal sensitivity threshold
        """
        if len(test_contacts) != len(expected_detections):
            raise ValueError("Test contacts and expected detections must have same length")
        
        best_threshold = self.config.sensitivity
        best_accuracy = 0
        
        # Test different thresholds
        test_thresholds = np.logspace(-3, 1, 20)  # 0.001 to 10 N
        
        for threshold in test_thresholds:
            correct = 0
            
            for contact_data, expected in zip(test_contacts, expected_detections):
                # Temporarily set threshold
                old_threshold = self.config.sensitivity
                self.config.sensitivity = threshold
                
                # Test detection
                contacts = self._parse_manual_contacts([contact_data])
                detected = len(contacts) > 0
                
                if detected == expected:
                    correct += 1
                
                # Restore threshold
                self.config.sensitivity = old_threshold
            
            accuracy = correct / len(test_contacts)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Update configuration
        self.config.sensitivity = best_threshold
        
        logger.info(f"Calibrated sensitivity to {best_threshold:.4f}N (accuracy: {best_accuracy:.2%})")
        
        return best_threshold
    
    def __str__(self) -> str:
        """String representation."""
        return f"TactileProcessor(sensitivity={self.config.sensitivity}N, processed={self.processing_count})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"TactileProcessor(config={self.config}, "
                f"material_bridge={self.material_bridge}, "
                f"processed={self.processing_count})")