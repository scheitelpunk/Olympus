#!/usr/bin/env python3
"""
Audio Spatial Processor - 3D spatial audio processing for MORPHEUS.

This module processes spatial audio information including:
- 3D sound localization
- Doppler effect calculations
- Multi-source audio fusion
- Acoustic material property inference
- Real-time audio stream processing

Features:
- SIMD-optimized audio processing
- Multi-threaded source separation
- Psychoacoustic modeling
- Material-based audio prediction
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from scipy import signal
from scipy.spatial.distance import cdist
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class AudioSource:
    """Represents a single audio source in 3D space."""
    id: str
    position: np.ndarray  # 3D position
    velocity: np.ndarray  # 3D velocity for Doppler
    frequency: float      # Fundamental frequency
    amplitude: float      # Source amplitude
    material: Optional[str] = None
    is_active: bool = True
    last_update: float = 0.0
    
@dataclass 
class AudioSignature:
    """Complete audio signature from spatial processing."""
    timestamp: float
    sources: List[AudioSource]
    listener_position: np.ndarray
    listener_velocity: np.ndarray
    dominant_frequency: float
    frequency_spectrum: np.ndarray
    spatial_map: np.ndarray  # 2D map of audio intensity
    doppler_shifts: List[float]
    reverb_characteristics: Dict[str, float]
    material_predictions: Dict[str, float]
    
    def to_embedding(self) -> np.ndarray:
        """Convert audio signature to fixed-size embedding vector."""
        embedding = np.zeros(32)
        
        # Basic audio metrics (8 dims)
        embedding[0] = self.dominant_frequency / 20000  # Normalized to 20kHz
        embedding[1] = len(self.sources) / 10  # Up to 10 sources
        embedding[2] = np.mean([s.amplitude for s in self.sources]) if self.sources else 0
        embedding[3] = np.std([s.amplitude for s in self.sources]) if len(self.sources) > 1 else 0
        
        # Doppler characteristics (2 dims)
        if self.doppler_shifts:
            embedding[4] = np.mean(self.doppler_shifts)
            embedding[5] = np.std(self.doppler_shifts) if len(self.doppler_shifts) > 1 else 0
            
        # Reverb characteristics (4 dims)
        reverb_keys = ['rt60', 'clarity', 'warmth', 'spaciousness']
        for i, key in enumerate(reverb_keys[:4]):
            embedding[6 + i] = self.reverb_characteristics.get(key, 0)
            
        # Frequency spectrum (16 dims) - log-spaced frequency bins
        if len(self.frequency_spectrum) > 0:
            # Resample spectrum to 16 bins
            resampled_spectrum = signal.resample(self.frequency_spectrum, 16)
            embedding[10:26] = resampled_spectrum / (np.max(resampled_spectrum) + 1e-10)
            
        # Spatial characteristics (6 dims)
        if len(self.spatial_map) > 0:
            # Flatten spatial map and take first 6 principal components
            flat_map = self.spatial_map.flatten()
            if len(flat_map) >= 6:
                embedding[26:32] = flat_map[:6] / (np.max(flat_map) + 1e-10)
                
        return embedding

class AudioProcessor:
    """Advanced 3D spatial audio processor with material inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audio processor.
        
        Args:
            config: Configuration dictionary with audio parameters
        """
        self.config = config
        
        # Audio processing parameters
        self.max_sources = config.get('max_sources', 10)
        self.frequency_range = config.get('frequency_range', [20, 20000])  # Hz
        self.speed_of_sound = config.get('speed_of_sound', 343.0)  # m/s
        self.doppler_enabled = config.get('doppler_enabled', True)
        self.sampling_rate = config.get('sampling_rate', 44100)  # Hz
        
        # Psychoacoustic parameters
        self.hrtf_enabled = config.get('hrtf_enabled', True)
        self.room_model_enabled = config.get('room_model_enabled', True)
        self.material_inference_enabled = config.get('material_inference', True)
        
        # Processing state
        self.active_sources = {}
        self.listener_position = np.array([0.0, 0.0, 0.0])
        self.listener_velocity = np.array([0.0, 0.0, 0.0])
        self.listener_orientation = np.array([1.0, 0.0, 0.0])  # Forward vector
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance metrics
        self.processing_times = []
        self.processed_frames = 0
        
        # Initialize HRTF database (simplified)
        self._init_hrtf_database()
        
        # Initialize material acoustic models
        self._init_material_models()
        
        logger.info(f"AudioProcessor initialized: {self.max_sources} sources, {self.frequency_range} Hz")
        
    def _init_hrtf_database(self):
        """Initialize Head-Related Transfer Function database."""
        
        # Simplified HRTF - in practice would load from measured data
        self.hrtf_azimuths = np.linspace(0, 2*np.pi, 36)  # 10° resolution
        self.hrtf_elevations = np.linspace(-np.pi/2, np.pi/2, 18)  # 10° resolution
        
        # Generate basic HRTF filters (simplified model)
        self.hrtf_filters = {}
        for azim in self.hrtf_azimuths:
            for elev in self.hrtf_elevations:
                # Simple HRTF approximation
                left_gain = self._compute_hrtf_gain(azim, elev, 'left')
                right_gain = self._compute_hrtf_gain(azim, elev, 'right')
                delay_samples = self._compute_itd(azim)  # Interaural Time Difference
                
                self.hrtf_filters[(azim, elev)] = {
                    'left_gain': left_gain,
                    'right_gain': right_gain,
                    'itd_samples': delay_samples
                }
                
    def _compute_hrtf_gain(self, azimuth: float, elevation: float, ear: str) -> float:
        """Compute HRTF gain for given direction and ear."""
        
        # Simplified HRTF model
        if ear == 'left':
            # Left ear is more sensitive to sounds from the left
            lateral_factor = np.cos(azimuth - np.pi/2)
        else:
            # Right ear is more sensitive to sounds from the right
            lateral_factor = np.cos(azimuth + np.pi/2)
            
        # Elevation affects both ears similarly
        elevation_factor = np.cos(elevation)
        
        # Head shadowing effect
        head_shadow = 1.0 - 0.3 * np.abs(np.sin(azimuth))
        
        gain = max(0.1, lateral_factor * elevation_factor * head_shadow)
        
        return gain
        
    def _compute_itd(self, azimuth: float) -> int:
        """Compute Interaural Time Difference in samples."""
        
        # Spherical head model
        head_radius = 0.0875  # ~8.75cm average head radius
        max_itd = head_radius / self.speed_of_sound
        
        # ITD varies with azimuth
        itd_seconds = max_itd * np.sin(azimuth)
        itd_samples = int(itd_seconds * self.sampling_rate)
        
        return itd_samples
        
    def _init_material_models(self):
        """Initialize acoustic models for different materials."""
        
        # Acoustic properties of common materials
        self.material_acoustics = {
            'steel': {
                'reflection_coeff': 0.9,
                'absorption_coeff': 0.1,
                'resonant_freq': 8000,  # High frequency ringing
                'decay_rate': 0.3,
                'hardness_factor': 1.0
            },
            'aluminum': {
                'reflection_coeff': 0.85,
                'absorption_coeff': 0.15,
                'resonant_freq': 6000,
                'decay_rate': 0.4,
                'hardness_factor': 0.8
            },
            'wood': {
                'reflection_coeff': 0.4,
                'absorption_coeff': 0.6,
                'resonant_freq': 2000,  # Warm, low frequencies
                'decay_rate': 0.7,
                'hardness_factor': 0.3
            },
            'plastic': {
                'reflection_coeff': 0.6,
                'absorption_coeff': 0.4,
                'resonant_freq': 3000,
                'decay_rate': 0.5,
                'hardness_factor': 0.4
            },
            'rubber': {
                'reflection_coeff': 0.2,
                'absorption_coeff': 0.8,  # High absorption
                'resonant_freq': 500,   # Low, muffled sounds
                'decay_rate': 0.9,
                'hardness_factor': 0.1
            },
            'concrete': {
                'reflection_coeff': 0.95,
                'absorption_coeff': 0.05,
                'resonant_freq': 1000,
                'decay_rate': 0.2,  # Long reverb
                'hardness_factor': 0.9
            }
        }
        
    def update_listener(self, 
                       position: np.ndarray, 
                       velocity: Optional[np.ndarray] = None,
                       orientation: Optional[np.ndarray] = None):
        """Update listener position and orientation."""
        
        with self._lock:
            self.listener_position = np.array(position)
            
            if velocity is not None:
                self.listener_velocity = np.array(velocity)
                
            if orientation is not None:
                # Normalize orientation vector
                self.listener_orientation = orientation / (np.linalg.norm(orientation) + 1e-10)
                
    def add_audio_source(self, 
                        source_id: str,
                        position: np.ndarray,
                        frequency: float,
                        amplitude: float,
                        velocity: Optional[np.ndarray] = None,
                        material: Optional[str] = None):
        """Add or update an audio source."""
        
        with self._lock:
            source = AudioSource(
                id=source_id,
                position=np.array(position),
                velocity=np.array(velocity) if velocity is not None else np.zeros(3),
                frequency=frequency,
                amplitude=amplitude,
                material=material,
                is_active=True,
                last_update=time.time()
            )
            
            self.active_sources[source_id] = source
            
            # Limit number of sources
            if len(self.active_sources) > self.max_sources:
                # Remove oldest inactive source
                oldest_id = min(self.active_sources.keys(), 
                              key=lambda k: self.active_sources[k].last_update)
                del self.active_sources[oldest_id]
                
    def remove_audio_source(self, source_id: str):
        """Remove an audio source."""
        
        with self._lock:
            if source_id in self.active_sources:
                del self.active_sources[source_id]
                
    def process_spatial_audio(self, 
                             robot_position: np.ndarray,
                             robot_velocity: Optional[np.ndarray] = None) -> Optional[AudioSignature]:
        """
        Process current spatial audio scene.
        
        Args:
            robot_position: Current robot position (listener)
            robot_velocity: Robot velocity for Doppler calculation
            
        Returns:
            AudioSignature with processed spatial audio data
        """
        start_time = time.time()
        
        # Update listener position
        self.update_listener(robot_position, robot_velocity)
        
        with self._lock:
            if not self.active_sources:
                return None
                
            active_sources = list(self.active_sources.values())
            
        # Process sources in parallel
        processed_sources = self._process_sources_parallel(active_sources)
        
        # Compute spatial audio characteristics
        signature = self._compute_audio_signature(
            processed_sources,
            robot_position,
            robot_velocity or np.zeros(3)
        )
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            
        self.processed_frames += 1
        
        return signature
        
    def _process_sources_parallel(self, sources: List[AudioSource]) -> List[Dict[str, Any]]:
        """Process multiple audio sources in parallel."""
        
        processed_sources = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=min(4, len(sources))) as executor:
            futures = []
            
            for source in sources:
                future = executor.submit(self._process_single_source, source)
                futures.append((source, future))
                
            # Collect results
            for source, future in futures:
                try:
                    processed = future.result(timeout=0.1)  # 100ms timeout per source
                    processed_sources.append(processed)
                except Exception as e:
                    logger.error(f"Failed to process audio source {source.id}: {e}")
                    
        return processed_sources
        
    def _process_single_source(self, source: AudioSource) -> Dict[str, Any]:
        """Process a single audio source."""
        
        # Calculate distance and direction
        relative_pos = source.position - self.listener_position
        distance = np.linalg.norm(relative_pos)
        
        if distance < 1e-6:
            distance = 1e-6  # Avoid division by zero
            
        direction = relative_pos / distance
        
        # Calculate Doppler shift
        doppler_shift = 0.0
        if self.doppler_enabled:
            relative_velocity = np.dot(source.velocity - self.listener_velocity, direction)
            doppler_shift = self._compute_doppler_shift(source.frequency, relative_velocity)
            
        # Calculate attenuation due to distance
        attenuation = self._compute_distance_attenuation(distance)
        
        # Apply HRTF if enabled
        hrtf_data = None
        if self.hrtf_enabled:
            azimuth, elevation = self._cartesian_to_spherical(direction)
            hrtf_data = self._get_hrtf_data(azimuth, elevation)
            
        # Material-based acoustic processing
        material_effects = self._apply_material_acoustics(source, distance)
        
        # Air absorption (frequency-dependent)
        air_absorption = self._compute_air_absorption(distance, source.frequency)
        
        return {
            'source': source,
            'distance': distance,
            'direction': direction,
            'doppler_shift': doppler_shift,
            'attenuation': attenuation,
            'hrtf_data': hrtf_data,
            'material_effects': material_effects,
            'air_absorption': air_absorption,
            'effective_amplitude': source.amplitude * attenuation * air_absorption,
            'effective_frequency': source.frequency + doppler_shift
        }
        
    def _compute_doppler_shift(self, frequency: float, relative_velocity: float) -> float:
        """Compute Doppler frequency shift."""
        
        # Doppler effect: f' = f * (v + vr) / (v + vs)
        # where vr = radial velocity of receiver (listener)
        # vs = radial velocity of source
        
        if abs(relative_velocity) < 0.01:  # Avoid computation for very small velocities
            return 0.0
            
        # Simplified Doppler formula
        doppler_factor = self.speed_of_sound / (self.speed_of_sound - relative_velocity)
        shifted_frequency = frequency * doppler_factor
        
        return shifted_frequency - frequency
        
    def _compute_distance_attenuation(self, distance: float) -> float:
        """Compute amplitude attenuation due to distance."""
        
        # Inverse square law with minimum distance to avoid infinite amplitude
        min_distance = 0.1  # 10cm minimum
        effective_distance = max(distance, min_distance)
        
        attenuation = 1.0 / (effective_distance ** 2)
        
        return attenuation
        
    def _cartesian_to_spherical(self, direction: np.ndarray) -> Tuple[float, float]:
        """Convert Cartesian direction to spherical coordinates (azimuth, elevation)."""
        
        x, y, z = direction
        
        # Azimuth: angle in XY plane from positive X axis
        azimuth = np.arctan2(y, x)
        
        # Elevation: angle from XY plane
        elevation = np.arcsin(np.clip(z, -1, 1))
        
        return azimuth, elevation
        
    def _get_hrtf_data(self, azimuth: float, elevation: float) -> Dict[str, float]:
        """Get HRTF data for given direction."""
        
        # Find closest HRTF measurement
        azim_idx = np.argmin(np.abs(self.hrtf_azimuths - azimuth))
        elev_idx = np.argmin(np.abs(self.hrtf_elevations - elevation))
        
        closest_azim = self.hrtf_azimuths[azim_idx]
        closest_elev = self.hrtf_elevations[elev_idx]
        
        return self.hrtf_filters[(closest_azim, closest_elev)]
        
    def _apply_material_acoustics(self, source: AudioSource, distance: float) -> Dict[str, Any]:
        """Apply material-specific acoustic effects."""
        
        if not source.material or source.material not in self.material_acoustics:
            # Default material properties
            return {
                'reflection': 0.5,
                'absorption': 0.5,
                'resonance': 1.0,
                'decay': 0.5
            }
            
        material_props = self.material_acoustics[source.material]
        
        # Distance-dependent effects
        reverb_strength = material_props['reflection_coeff'] * np.exp(-distance / 10.0)
        absorption_effect = 1.0 - material_props['absorption_coeff'] * (1.0 - np.exp(-distance / 5.0))
        
        # Frequency-dependent resonance
        freq_ratio = source.frequency / material_props['resonant_freq']
        resonance_factor = np.exp(-0.5 * ((freq_ratio - 1.0) ** 2))
        
        return {
            'reflection': reverb_strength,
            'absorption': absorption_effect,
            'resonance': resonance_factor,
            'decay': material_props['decay_rate'],
            'material': source.material
        }
        
    def _compute_air_absorption(self, distance: float, frequency: float) -> float:
        """Compute air absorption attenuation."""
        
        # Simplified air absorption model
        # High frequencies are absorbed more than low frequencies
        
        # Absorption coefficient (dB/m) - simplified model
        freq_khz = frequency / 1000.0
        absorption_coeff = 0.1 * (freq_khz ** 1.5) / 1000.0  # dB/m
        
        # Convert to linear attenuation
        absorption_db = absorption_coeff * distance
        attenuation = 10 ** (-absorption_db / 20.0)
        
        return np.clip(attenuation, 0.001, 1.0)
        
    def _compute_audio_signature(self, 
                               processed_sources: List[Dict[str, Any]],
                               listener_pos: np.ndarray,
                               listener_vel: np.ndarray) -> AudioSignature:
        """Compute complete audio signature from processed sources."""
        
        if not processed_sources:
            return AudioSignature(
                timestamp=time.time(),
                sources=[],
                listener_position=listener_pos,
                listener_velocity=listener_vel,
                dominant_frequency=0,
                frequency_spectrum=np.zeros(32),
                spatial_map=np.zeros((8, 8)),
                doppler_shifts=[],
                reverb_characteristics={},
                material_predictions={}
            )
            
        # Extract data from processed sources
        sources = [ps['source'] for ps in processed_sources]
        effective_amplitudes = [ps['effective_amplitude'] for ps in processed_sources]
        effective_frequencies = [ps['effective_frequency'] for ps in processed_sources]
        doppler_shifts = [ps['doppler_shift'] for ps in processed_sources]
        
        # Find dominant frequency (highest amplitude)
        if effective_amplitudes:
            dominant_idx = np.argmax(effective_amplitudes)
            dominant_frequency = effective_frequencies[dominant_idx]
        else:
            dominant_frequency = 0
            
        # Compute frequency spectrum
        frequency_spectrum = self._compute_frequency_spectrum(
            effective_frequencies, 
            effective_amplitudes
        )
        
        # Compute spatial audio map
        spatial_map = self._compute_spatial_map(processed_sources, listener_pos)
        
        # Compute reverb characteristics
        reverb_characteristics = self._compute_reverb_characteristics(processed_sources)
        
        # Predict materials based on audio characteristics
        material_predictions = self._predict_materials(processed_sources)
        
        return AudioSignature(
            timestamp=time.time(),
            sources=sources,
            listener_position=listener_pos,
            listener_velocity=listener_vel,
            dominant_frequency=dominant_frequency,
            frequency_spectrum=frequency_spectrum,
            spatial_map=spatial_map,
            doppler_shifts=doppler_shifts,
            reverb_characteristics=reverb_characteristics,
            material_predictions=material_predictions
        )
        
    def _compute_frequency_spectrum(self, 
                                  frequencies: List[float], 
                                  amplitudes: List[float]) -> np.ndarray:
        """Compute frequency spectrum from active sources."""
        
        # Create frequency bins (log-spaced)
        freq_bins = np.logspace(
            np.log10(max(self.frequency_range[0], 1)), 
            np.log10(self.frequency_range[1]), 
            33  # 32 bins + 1 for edges
        )
        
        spectrum = np.zeros(32)
        
        # Bin frequencies and sum amplitudes
        for freq, amp in zip(frequencies, amplitudes):
            if freq <= 0:
                continue
                
            # Find appropriate bin
            bin_idx = np.searchsorted(freq_bins, freq) - 1
            bin_idx = np.clip(bin_idx, 0, 31)
            
            spectrum[bin_idx] += amp
            
        # Normalize
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
            
        return spectrum
        
    def _compute_spatial_map(self, 
                           processed_sources: List[Dict[str, Any]], 
                           listener_pos: np.ndarray) -> np.ndarray:
        """Compute 2D spatial map of audio intensity."""
        
        # Create 2D grid around listener (8x8 grid, 2m x 2m)
        grid_size = 8
        grid_extent = 1.0  # ±1m around listener
        
        x_coords = np.linspace(-grid_extent, grid_extent, grid_size)
        y_coords = np.linspace(-grid_extent, grid_extent, grid_size)
        
        spatial_map = np.zeros((grid_size, grid_size))
        
        # Project 3D sources onto 2D grid
        for ps in processed_sources:
            source_pos = ps['source'].position
            relative_pos = source_pos - listener_pos
            
            # Project to XY plane
            x, y = relative_pos[0], relative_pos[1]
            
            # Find grid indices
            x_idx = np.searchsorted(x_coords, x)
            y_idx = np.searchsorted(y_coords, y)
            
            x_idx = np.clip(x_idx, 0, grid_size - 1)
            y_idx = np.clip(y_idx, 0, grid_size - 1)
            
            # Add amplitude to grid
            spatial_map[y_idx, x_idx] += ps['effective_amplitude']
            
        # Apply Gaussian smoothing (simplified)
        from scipy.ndimage import gaussian_filter
        try:
            spatial_map = gaussian_filter(spatial_map, sigma=0.5)
        except:
            # Fallback if scipy is not available
            pass
        
        return spatial_map
        
    def _compute_reverb_characteristics(self, 
                                      processed_sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute reverb characteristics from material properties."""
        
        if not processed_sources:
            return {'rt60': 0, 'clarity': 0, 'warmth': 0, 'spaciousness': 0}
            
        # Aggregate material effects
        total_reflection = 0
        total_absorption = 0
        total_amplitude = 0
        
        for ps in processed_sources:
            material_effects = ps['material_effects']
            amplitude = ps['effective_amplitude']
            
            total_reflection += material_effects['reflection'] * amplitude
            total_absorption += material_effects['absorption'] * amplitude
            total_amplitude += amplitude
            
        if total_amplitude > 0:
            avg_reflection = total_reflection / total_amplitude
            avg_absorption = total_absorption / total_amplitude
        else:
            avg_reflection = 0.5
            avg_absorption = 0.5
            
        # Compute reverb metrics
        rt60 = -60.0 / (20 * np.log10(avg_absorption + 0.01))  # Reverberation time
        clarity = avg_reflection  # C80 clarity index approximation
        warmth = 1.0 - avg_absorption  # Warmth from low frequency content
        spaciousness = avg_reflection * 0.7  # Spatial impression
        
        return {
            'rt60': np.clip(rt60, 0, 10),  # Max 10 seconds
            'clarity': np.clip(clarity, 0, 1),
            'warmth': np.clip(warmth, 0, 1),
            'spaciousness': np.clip(spaciousness, 0, 1)
        }
        
    def _predict_materials(self, 
                         processed_sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict materials based on acoustic characteristics."""
        
        if not self.material_inference_enabled or not processed_sources:
            return {}
            
        material_scores = defaultdict(float)
        
        for ps in processed_sources:
            source = ps['source']
            material_effects = ps['material_effects']
            
            # If material is known, use it directly
            if source.material and source.material in self.material_acoustics:
                material_scores[source.material] += ps['effective_amplitude']
                continue
                
            # Predict material from acoustic characteristics
            for material, props in self.material_acoustics.items():
                score = 0.0
                
                # Frequency-based matching
                freq_diff = abs(source.frequency - props['resonant_freq']) / props['resonant_freq']
                freq_score = np.exp(-freq_diff)  # Closer frequencies score higher
                score += freq_score * 0.4
                
                # Reflection-based matching
                reflection_diff = abs(material_effects['reflection'] - props['reflection_coeff'])
                reflection_score = 1.0 - reflection_diff
                score += reflection_score * 0.3
                
                # Absorption-based matching
                absorption_diff = abs(material_effects['absorption'] - (1 - props['absorption_coeff']))
                absorption_score = 1.0 - absorption_diff
                score += absorption_score * 0.3
                
                material_scores[material] += score * ps['effective_amplitude']
                
        # Normalize scores
        total_score = sum(material_scores.values())
        if total_score > 0:
            material_scores = {k: v / total_score for k, v in material_scores.items()}
            
        return dict(material_scores)
        
    def cleanup_inactive_sources(self, max_age: float = 5.0):
        """Remove sources that haven't been updated recently."""
        
        current_time = time.time()
        
        with self._lock:
            inactive_sources = [
                source_id for source_id, source in self.active_sources.items()
                if current_time - source.last_update > max_age
            ]
            
            for source_id in inactive_sources:
                del self.active_sources[source_id]
                
        if inactive_sources:
            logger.info(f"Removed {len(inactive_sources)} inactive audio sources")
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get audio processing performance metrics."""
        
        metrics = {
            'processed_frames': self.processed_frames,
            'active_sources': len(self.active_sources),
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'processing_fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        }
        
        return metrics
        
    def set_room_acoustics(self, 
                          room_size: Tuple[float, float, float],
                          wall_materials: List[str],
                          humidity: float = 0.5,
                          temperature: float = 20.0):
        """Set room acoustic properties for realistic reverb simulation."""
        
        # Update speed of sound based on temperature
        self.speed_of_sound = 331.3 * np.sqrt(1 + temperature / 273.15)
        
        # Compute room reverb characteristics
        room_volume = np.prod(room_size)
        
        # Average absorption from wall materials
        total_absorption = 0
        for material in wall_materials:
            if material in self.material_acoustics:
                total_absorption += self.material_acoustics[material]['absorption_coeff']
        avg_absorption = total_absorption / max(1, len(wall_materials))
        
        # Update HRTF processing based on room characteristics
        # (In practice, would modify HRTF filters based on room acoustics)
        
        logger.info(f"Updated room acoustics: {room_size}m, avg absorption: {avg_absorption:.2f}")
        
    def process_audio_stream(self, 
                           audio_data: np.ndarray, 
                           sample_rate: int) -> Dict[str, Any]:
        """Process real audio stream data for source detection and analysis."""
        
        # This would be used with real audio input
        # For now, return simulated analysis
        
        analysis = {
            'detected_sources': [],
            'dominant_frequencies': [],
            'estimated_materials': [],
            'spatial_activity': np.zeros((8, 8))
        }
        
        if len(audio_data) == 0:
            return analysis
            
        # Perform FFT analysis
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        
        # Find peaks in frequency domain
        magnitude = np.abs(fft)
        peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        
        # Extract dominant frequencies
        peak_freqs = freqs[peaks]
        peak_mags = magnitude[peaks]
        
        # Sort by magnitude
        sorted_indices = np.argsort(peak_mags)[::-1]
        
        analysis['dominant_frequencies'] = peak_freqs[sorted_indices[:5]].tolist()
        
        # Estimate materials based on frequency characteristics
        for freq in peak_freqs[:3]:  # Top 3 frequencies
            best_material = None
            best_score = 0
            
            for material, props in self.material_acoustics.items():
                # Score based on how close frequency is to material's resonant frequency
                freq_ratio = freq / props['resonant_freq']
                score = np.exp(-0.5 * ((np.log(freq_ratio)) ** 2))
                
                if score > best_score:
                    best_score = score
                    best_material = material
                    
            if best_material and best_score > 0.3:
                analysis['estimated_materials'].append({
                    'material': best_material,
                    'confidence': best_score,
                    'frequency': freq
                })
                
        return analysis