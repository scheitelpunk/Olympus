#!/usr/bin/env python3
"""
Dream Orchestrator - Experience replay and strategy optimization system.

This module implements the core dream functionality of MORPHEUS, allowing
for parallel experience replay, variation generation, and strategy learning.

Features:
- Multi-threaded dream sessions
- Experience variation generation  
- Strategy consolidation and ranking
- Comprehensive performance metrics
- Neural pattern optimization
"""

import time
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import torch
import torch.nn as nn
from scipy.stats import entropy

logger = logging.getLogger(__name__)

@dataclass
class DreamConfig:
    """Configuration for dream session parameters."""
    replay_speed: float = 10.0
    variation_factor: float = 0.2
    exploration_rate: float = 0.3
    consolidation_threshold: float = 0.8
    min_improvement: float = 0.1
    max_iterations: int = 1000
    parallel_dreams: int = 4
    experience_sample_rate: float = 0.8
    strategy_merge_threshold: float = 0.15
    neural_learning_rate: float = 0.001

@dataclass
class ExperienceReplay:
    """Single experience replay result."""
    original_experience: Dict[str, Any]
    variations: List[Dict[str, Any]]
    improvements: List[float]
    strategies_found: List[Dict[str, Any]]
    replay_time: float
    neural_updates: int

@dataclass
class DreamSession:
    """Complete dream session results."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    config: DreamConfig
    experiences_processed: int
    variations_generated: int
    strategies_discovered: int
    strategies_consolidated: int
    average_improvement: float
    best_improvement: float
    neural_convergence: float
    compute_metrics: Dict[str, float]

class NeuralStrategyOptimizer(nn.Module):
    """Neural network for strategy optimization and pattern recognition."""
    
    def __init__(self, state_dim: int = 128, action_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Strategy encoding network
        self.strategy_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        # Improvement prediction network
        self.improvement_predictor = nn.Sequential(
            nn.Linear(64, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Strategy similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),  # Two 64-dim encodings
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through strategy optimizer."""
        combined = torch.cat([state, action], dim=-1)
        encoding = self.strategy_encoder(combined)
        improvement = self.improvement_predictor(encoding)
        return encoding, improvement
    
    def compute_similarity(self, encoding1: torch.Tensor, encoding2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two strategy encodings."""
        combined = torch.cat([encoding1, encoding2], dim=-1)
        return self.similarity_net(combined)

class DreamOrchestrator:
    """Advanced dream orchestrator with parallel processing and neural optimization."""
    
    def __init__(self, 
                 database,
                 material_bridge,
                 config: DreamConfig):
        """
        Initialize dream orchestrator.
        
        Args:
            database: Database connection for experience storage
            material_bridge: Bridge to material properties
            config: Dream configuration parameters
        """
        self.db = database
        self.material_bridge = material_bridge
        self.config = config
        
        # Initialize neural optimizer
        self.neural_optimizer = NeuralStrategyOptimizer()
        self.optimizer = torch.optim.Adam(
            self.neural_optimizer.parameters(), 
            lr=config.neural_learning_rate
        )
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.session_metrics = {
            'total_dreams': 0,
            'total_experiences_processed': 0,
            'total_strategies_found': 0,
            'average_session_time': 0,
            'neural_training_epochs': 0
        }
        
        # Strategy cache for fast similarity comparisons
        self.strategy_cache = {}
        
        logger.info("DreamOrchestrator initialized with neural optimization")
        
    def dream(self, 
              duration_seconds: float = 60,
              focus_materials: Optional[List[str]] = None,
              focus_actions: Optional[List[str]] = None) -> DreamSession:
        """
        Run comprehensive dream session with parallel processing.
        
        Args:
            duration_seconds: Maximum dream duration
            focus_materials: Optional list of materials to focus on
            focus_actions: Optional list of actions to focus on
            
        Returns:
            Complete dream session results
        """
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting dream session {session_id} for {duration_seconds:.1f}s")
        
        # Get experiences for replay
        experiences = self._get_dream_experiences(
            focus_materials=focus_materials,
            focus_actions=focus_actions
        )
        
        if not experiences:
            logger.warning("No experiences found for dream session")
            return self._create_empty_session(session_id, start_time)
            
        logger.info(f"Processing {len(experiences)} experiences in parallel")
        
        # Initialize session tracking
        session = DreamSession(
            session_id=session_id,
            start_time=start_time,
            end_time=None,
            config=self.config,
            experiences_processed=0,
            variations_generated=0,
            strategies_discovered=0,
            strategies_consolidated=0,
            average_improvement=0,
            best_improvement=0,
            neural_convergence=0,
            compute_metrics={}
        )
        
        # Parallel experience processing
        all_replays = self._process_experiences_parallel(
            experiences, 
            duration_seconds
        )
        
        # Consolidate and optimize strategies
        consolidated_strategies = self._consolidate_strategies_neural(all_replays)
        
        # Store best strategies
        stored_count = self._store_strategies(consolidated_strategies)
        
        # Update session metrics
        end_time = time.time()
        session.end_time = end_time
        session.experiences_processed = len(experiences)
        session.variations_generated = sum(len(replay.variations) for replay in all_replays)
        session.strategies_discovered = len(consolidated_strategies)
        session.strategies_consolidated = stored_count
        
        if consolidated_strategies:
            improvements = [s['improvement_ratio'] for s in consolidated_strategies]
            session.average_improvement = np.mean(improvements)
            session.best_improvement = max(improvements)
            
        # Neural convergence metric
        session.neural_convergence = self._compute_neural_convergence()
        
        # Performance metrics
        session.compute_metrics = {
            'session_duration': end_time - start_time,
            'experiences_per_second': len(experiences) / (end_time - start_time),
            'variations_per_experience': session.variations_generated / max(1, len(experiences)),
            'strategy_discovery_rate': len(consolidated_strategies) / max(1, len(experiences)),
            'neural_updates': sum(replay.neural_updates for replay in all_replays)
        }
        
        # Update global metrics
        self._update_global_metrics(session)
        
        # Store dream session in database
        self._store_dream_session(session)
        
        logger.info(f"Dream session complete: {stored_count} strategies learned")
        logger.info(f"Neural convergence: {session.neural_convergence:.3f}")
        
        return session
        
    def _get_dream_experiences(self, 
                              focus_materials: Optional[List[str]] = None,
                              focus_actions: Optional[List[str]] = None,
                              hours: int = 24,
                              limit: int = 5000) -> List[Dict[str, Any]]:
        """Get experiences for dream processing with filtering."""
        
        # Get recent experiences
        experiences = self.db.get_recent_experiences(hours=hours, limit=limit)
        
        # Apply filters
        filtered = []
        for exp in experiences:
            # Material filter
            if focus_materials:
                if exp.get('primary_material') not in focus_materials:
                    continue
                    
            # Action filter
            if focus_actions:
                if exp.get('action_type') not in focus_actions:
                    continue
                    
            # Sample experiences based on config
            if np.random.random() < self.config.experience_sample_rate:
                filtered.append(exp)
                
        logger.info(f"Filtered to {len(filtered)} experiences for dreaming")
        return filtered
        
    def _process_experiences_parallel(self, 
                                    experiences: List[Dict],
                                    max_duration: float) -> List[ExperienceReplay]:
        """Process experiences in parallel threads."""
        
        # Split experiences into chunks for parallel processing
        chunk_size = max(1, len(experiences) // self.config.parallel_dreams)
        chunks = [experiences[i:i + chunk_size] 
                 for i in range(0, len(experiences), chunk_size)]
        
        all_replays = []
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_dreams) as executor:
            # Submit chunks to threads
            for chunk_id, chunk in enumerate(chunks):
                future = executor.submit(
                    self._process_experience_chunk,
                    chunk,
                    chunk_id,
                    max_duration / len(chunks)
                )
                futures.append(future)
                
            # Collect results as they complete
            for future in as_completed(futures, timeout=max_duration + 10):
                try:
                    chunk_replays = future.result(timeout=5)
                    all_replays.extend(chunk_replays)
                except Exception as e:
                    logger.error(f"Dream chunk processing failed: {e}")
                    
        return all_replays
        
    def _process_experience_chunk(self, 
                                chunk: List[Dict],
                                chunk_id: int,
                                max_time: float) -> List[ExperienceReplay]:
        """Process a chunk of experiences in a single thread."""
        
        replays = []
        start_time = time.time()
        
        logger.debug(f"Processing chunk {chunk_id} with {len(chunk)} experiences")
        
        for exp in chunk:
            # Check time limit
            if time.time() - start_time > max_time:
                logger.debug(f"Chunk {chunk_id} hit time limit")
                break
                
            replay = self._replay_single_experience(exp)
            if replay:
                replays.append(replay)
                
        logger.debug(f"Chunk {chunk_id} completed: {len(replays)} replays")
        return replays
        
    def _replay_single_experience(self, experience: Dict[str, Any]) -> Optional[ExperienceReplay]:
        """Replay and generate variations for a single experience."""
        
        start_time = time.time()
        
        # Generate variations
        variations = self._generate_experience_variations(experience)
        
        if not variations:
            return None
            
        # Evaluate improvements
        improvements = []
        strategies_found = []
        neural_updates = 0
        
        for variation in variations:
            improvement = self._evaluate_variation_neural(experience, variation)
            improvements.append(improvement)
            
            # Extract strategy if improvement is significant
            if improvement > self.config.min_improvement:
                strategy = self._extract_strategy(experience, variation, improvement)
                strategies_found.append(strategy)
                
                # Update neural network
                neural_updates += self._update_neural_network(experience, variation, improvement)
                
        return ExperienceReplay(
            original_experience=experience,
            variations=variations,
            improvements=improvements,
            strategies_found=strategies_found,
            replay_time=time.time() - start_time,
            neural_updates=neural_updates
        )
        
    def _generate_experience_variations(self, 
                                      experience: Dict[str, Any], 
                                      max_variations: int = 15) -> List[Dict[str, Any]]:
        """Generate comprehensive variations of an experience."""
        
        variations = []
        
        # Material variations
        variations.extend(self._generate_material_variations(experience, 5))
        
        # Physical parameter variations
        variations.extend(self._generate_physics_variations(experience, 5))
        
        # Action parameter variations
        variations.extend(self._generate_action_variations(experience, 5))
        
        # Neural-guided variations
        variations.extend(self._generate_neural_variations(experience, 3))
        
        # Limit total variations
        return variations[:max_variations]
        
    def _generate_material_variations(self, 
                                    experience: Dict[str, Any], 
                                    count: int) -> List[Dict[str, Any]]:
        """Generate material-based variations."""
        
        variations = []
        current_material = experience.get('primary_material', 'default')
        
        # Get available materials
        available_materials = list(self.material_bridge.materials.keys())
        other_materials = [m for m in available_materials if m != current_material]
        
        # Sample different materials
        for _ in range(min(count, len(other_materials))):
            new_material = np.random.choice(other_materials)
            
            variation = experience.copy()
            variation['primary_material'] = new_material
            variation['variation_type'] = 'material'
            variation['variation_source'] = current_material
            variation['variation_target'] = new_material
            
            # Update material-dependent properties
            mat_props = self.material_bridge.get_material(new_material)
            if mat_props:
                variation['predicted_friction'] = mat_props.friction
                variation['predicted_hardness'] = mat_props.young_modulus
                
            variations.append(variation)
            
        return variations
        
    def _generate_physics_variations(self, 
                                   experience: Dict[str, Any], 
                                   count: int) -> List[Dict[str, Any]]:
        """Generate physics parameter variations."""
        
        variations = []
        
        # Force variations
        if experience.get('forces'):
            original_forces = np.array(experience['forces'])
            
            for _ in range(count // 2):
                # Scale forces
                scale_factor = np.random.uniform(0.3, 3.0)
                
                variation = experience.copy()
                variation['forces'] = (original_forces * scale_factor).tolist()
                variation['variation_type'] = 'force'
                variation['force_scale'] = scale_factor
                
                variations.append(variation)
                
        # Velocity variations
        if 'robot_velocity' in experience:
            original_velocity = np.array(experience['robot_velocity'])
            
            for _ in range(count // 2):
                # Add velocity noise
                noise = np.random.normal(0, self.config.variation_factor, 3)
                
                variation = experience.copy()
                variation['robot_velocity'] = (original_velocity + noise).tolist()
                variation['variation_type'] = 'velocity'
                
                variations.append(variation)
                
        return variations
        
    def _generate_action_variations(self, 
                                  experience: Dict[str, Any], 
                                  count: int) -> List[Dict[str, Any]]:
        """Generate action parameter variations."""
        
        variations = []
        action_params = experience.get('action_params', {})
        
        if not action_params:
            return variations
            
        for _ in range(count):
            variation = experience.copy()
            new_params = action_params.copy()
            
            # Vary numeric parameters
            for key, value in action_params.items():
                if isinstance(value, (int, float)):
                    # Add proportional noise
                    noise_std = abs(value) * self.config.variation_factor
                    noise = np.random.normal(0, noise_std)
                    new_params[key] = value + noise
                    
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # Vary array parameters
                    if all(isinstance(v, (int, float)) for v in value):
                        arr = np.array(value)
                        noise = np.random.normal(0, np.std(arr) * self.config.variation_factor, arr.shape)
                        new_params[key] = (arr + noise).tolist()
                        
            variation['action_params'] = new_params
            variation['variation_type'] = 'action'
            
            variations.append(variation)
            
        return variations
        
    def _generate_neural_variations(self, 
                                  experience: Dict[str, Any], 
                                  count: int) -> List[Dict[str, Any]]:
        """Generate neural network guided variations."""
        
        variations = []
        
        # Convert experience to neural input
        state = self._experience_to_state(experience)
        action = self._experience_to_action(experience)
        
        if state is None or action is None:
            return variations
            
        # Generate variations using neural network gradients
        for _ in range(count):
            # Add learned noise pattern
            with torch.no_grad():
                encoding, predicted_improvement = self.neural_optimizer(state, action)
                
            # Create variation based on encoding
            variation = experience.copy()
            
            # Apply neural-guided modifications
            if 'action_params' in experience:
                params = experience['action_params'].copy()
                # Use encoding to guide parameter changes
                encoding_np = encoding.squeeze().numpy()
                
                # Apply encoding-based modifications to parameters
                for i, (key, value) in enumerate(params.items()):
                    if isinstance(value, (int, float)) and i < len(encoding_np):
                        # Use encoding value as modification guide
                        modifier = encoding_np[i % len(encoding_np)] * 0.1
                        params[key] = value * (1 + modifier)
                        
                variation['action_params'] = params
                
            variation['variation_type'] = 'neural'
            variation['neural_confidence'] = predicted_improvement.item()
            
            variations.append(variation)
            
        return variations
        
    def _evaluate_variation_neural(self, 
                                 original: Dict[str, Any], 
                                 variation: Dict[str, Any]) -> float:
        """Evaluate variation improvement using neural network."""
        
        # Convert to neural inputs
        orig_state = self._experience_to_state(original)
        orig_action = self._experience_to_action(original)
        var_state = self._experience_to_state(variation)
        var_action = self._experience_to_action(variation)
        
        if any(x is None for x in [orig_state, orig_action, var_state, var_action]):
            # Fall back to heuristic evaluation
            return self._evaluate_variation_heuristic(original, variation)
            
        # Neural evaluation
        with torch.no_grad():
            _, orig_improvement = self.neural_optimizer(orig_state, orig_action)
            _, var_improvement = self.neural_optimizer(var_state, var_action)
            
        # Improvement is difference in predicted performance
        improvement = (var_improvement - orig_improvement).item()
        
        # Add heuristic component
        heuristic_improvement = self._evaluate_variation_heuristic(original, variation)
        
        # Combine neural and heuristic evaluations
        combined_improvement = 0.7 * improvement + 0.3 * heuristic_improvement
        
        return max(0, combined_improvement)
        
    def _evaluate_variation_heuristic(self, 
                                    original: Dict[str, Any], 
                                    variation: Dict[str, Any]) -> float:
        """Heuristic evaluation of variation improvement."""
        
        improvement = 0.0
        variation_type = variation.get('variation_type', 'unknown')
        
        # Material-based improvements
        if variation_type == 'material':
            orig_material = original.get('primary_material', 'default')
            new_material = variation.get('primary_material', 'default')
            
            orig_props = self.material_bridge.get_material(orig_material)
            new_props = self.material_bridge.get_material(new_material)
            
            if orig_props and new_props:
                action_type = original.get('action_type', '').lower()
                
                # Task-specific material preferences
                if 'grip' in action_type or 'hold' in action_type:
                    # Prefer higher friction for gripping
                    improvement += (new_props.friction - orig_props.friction) * 0.5
                    
                elif 'slide' in action_type or 'move' in action_type:
                    # Prefer lower friction for sliding
                    improvement += (orig_props.friction - new_props.friction) * 0.3
                    
                elif 'impact' in action_type or 'hit' in action_type:
                    # Prefer higher restitution for impacts
                    improvement += (new_props.restitution - orig_props.restitution) * 0.4
                    
        # Force optimization
        elif variation_type == 'force':
            scale_factor = variation.get('force_scale', 1.0)
            # Prefer moderate forces (not too high or low)
            optimal_scale = 1.0
            deviation = abs(scale_factor - optimal_scale)
            improvement = max(0, 0.5 - deviation)
            
        # Action optimization
        elif variation_type == 'action':
            # Reward exploration
            improvement = self.config.exploration_rate * 0.5
            
        # Neural variations
        elif variation_type == 'neural':
            neural_confidence = variation.get('neural_confidence', 0)
            improvement = neural_confidence * 0.3
            
        return np.clip(improvement, 0, 1)
        
    def _consolidate_strategies_neural(self, 
                                     replays: List[ExperienceReplay]) -> List[Dict[str, Any]]:
        """Consolidate strategies using neural similarity measures."""
        
        # Collect all strategies
        all_strategies = []
        for replay in replays:
            all_strategies.extend(replay.strategies_found)
            
        if not all_strategies:
            return []
            
        logger.info(f"Consolidating {len(all_strategies)} strategies")
        
        # Convert strategies to neural encodings
        strategy_encodings = []
        for strategy in all_strategies:
            encoding = self._strategy_to_encoding(strategy)
            if encoding is not None:
                strategy_encodings.append(encoding)
            else:
                strategy_encodings.append(torch.zeros(64))  # Default encoding
                
        # Group similar strategies using neural similarity
        consolidated = self._group_similar_strategies_neural(
            all_strategies, 
            strategy_encodings
        )
        
        # Rank by improvement ratio
        consolidated.sort(key=lambda x: x['improvement_ratio'], reverse=True)
        
        logger.info(f"Consolidated to {len(consolidated)} unique strategies")
        return consolidated
        
    def _group_similar_strategies_neural(self, 
                                       strategies: List[Dict[str, Any]], 
                                       encodings: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Group similar strategies using neural similarity network."""
        
        if not strategies:
            return []
            
        consolidated = []
        processed = set()
        
        for i, strategy in enumerate(strategies):
            if i in processed:
                continue
                
            # Find similar strategies
            similar_indices = [i]
            
            for j, other_strategy in enumerate(strategies[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Compute neural similarity
                similarity = self._compute_neural_similarity(
                    encodings[i], 
                    encodings[j]
                ).item()
                
                if similarity > (1.0 - self.config.strategy_merge_threshold):
                    similar_indices.append(j)
                    processed.add(j)
                    
            # Merge similar strategies
            similar_strategies = [strategies[idx] for idx in similar_indices]
            merged_strategy = self._merge_strategies(similar_strategies)
            
            consolidated.append(merged_strategy)
            processed.add(i)
            
        return consolidated
        
    def _compute_neural_similarity(self, 
                                 encoding1: torch.Tensor, 
                                 encoding2: torch.Tensor) -> torch.Tensor:
        """Compute neural similarity between strategy encodings."""
        
        with torch.no_grad():
            similarity = self.neural_optimizer.compute_similarity(encoding1, encoding2)
            
        return similarity
        
    def _strategy_to_encoding(self, strategy: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Convert strategy to neural encoding."""
        
        try:
            # Extract key features from strategy
            features = np.zeros(64)
            
            # Strategy type encoding
            strategy_types = ['material', 'force', 'action', 'neural', 'velocity']
            category = strategy.get('category', 'unknown')
            if category in strategy_types:
                features[strategy_types.index(category)] = 1.0
                
            # Improvement ratio
            features[5] = strategy.get('improvement_ratio', 0)
            
            # Confidence
            features[6] = strategy.get('confidence', 0)
            
            # Material encoding
            materials = strategy.get('applicable_materials', [])
            if materials:
                # Simple hash-based encoding of materials
                for i, mat in enumerate(materials[:5]):
                    features[10 + i] = hash(mat) % 100 / 100.0
                    
            # Fill remaining features with strategy data hash
            strategy_str = str(strategy.get('strategy_data', {}))
            hash_val = hash(strategy_str)
            for i in range(15, 64):
                features[i] = ((hash_val >> i) % 2) * 0.1
                
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Failed to encode strategy: {e}")
            return None
            
    def _experience_to_state(self, experience: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Convert experience to state tensor."""
        
        try:
            # Use fused embedding if available
            if 'fused_embedding' in experience and experience['fused_embedding']:
                embedding = np.array(experience['fused_embedding'])
                if embedding.shape[0] >= 128:
                    return torch.tensor(embedding[:128], dtype=torch.float32)
                else:
                    # Pad to 128 dimensions
                    padded = np.zeros(128)
                    padded[:embedding.shape[0]] = embedding
                    return torch.tensor(padded, dtype=torch.float32)
                    
            # Fall back to constructing state from available data
            state = np.zeros(128)
            
            # Material encoding
            material = experience.get('primary_material', 'default')
            mat_props = self.material_bridge.get_material(material)
            if mat_props:
                state[0] = mat_props.friction
                state[1] = mat_props.restitution
                state[2] = np.log10(mat_props.young_modulus + 1) / 12  # Normalized
                state[3] = mat_props.density / 10000  # Normalized
                
            # Forces
            if 'forces' in experience and experience['forces']:
                forces = np.array(experience['forces'])[:3]  # First 3 components
                state[4:4+len(forces)] = forces / 100.0  # Normalized
                
            # Success and reward
            state[7] = 1.0 if experience.get('success', False) else 0.0
            state[8] = np.clip(experience.get('reward', 0) / 10.0, -1, 1)
            
            return torch.tensor(state, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Failed to convert experience to state: {e}")
            return None
            
    def _experience_to_action(self, experience: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Convert experience to action tensor."""
        
        try:
            action = np.zeros(16)
            
            # Action type encoding
            action_types = ['grip', 'push', 'pull', 'lift', 'slide', 'touch', 'impact']
            action_type = experience.get('action_type', 'unknown').lower()
            if action_type in action_types:
                action[action_types.index(action_type)] = 1.0
                
            # Action parameters
            params = experience.get('action_params', {})
            if params:
                param_values = []
                for value in params.values():
                    if isinstance(value, (int, float)):
                        param_values.append(float(value))
                    elif isinstance(value, (list, tuple)):
                        param_values.extend([float(v) for v in value if isinstance(v, (int, float))])
                        
                # Fill action vector with parameter values
                for i, val in enumerate(param_values[:8]):
                    action[8 + i] = val
                    
            return torch.tensor(action, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Failed to convert experience to action: {e}")
            return None
            
    def _update_neural_network(self, 
                              original: Dict[str, Any], 
                              variation: Dict[str, Any], 
                              improvement: float) -> int:
        """Update neural network with experience data."""
        
        if improvement < self.config.min_improvement:
            return 0
            
        # Convert to neural inputs
        state = self._experience_to_state(variation)
        action = self._experience_to_action(variation)
        
        if state is None or action is None:
            return 0
            
        try:
            # Forward pass
            encoding, predicted_improvement = self.neural_optimizer(state.unsqueeze(0), action.unsqueeze(0))
            
            # Loss is difference between predicted and actual improvement
            target = torch.tensor([[improvement]], dtype=torch.float32)
            loss = nn.MSELoss()(predicted_improvement, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return 1
            
        except Exception as e:
            logger.error(f"Neural network update failed: {e}")
            return 0
            
    def _extract_strategy(self, 
                        original: Dict[str, Any], 
                        variation: Dict[str, Any], 
                        improvement: float) -> Dict[str, Any]:
        """Extract learned strategy from successful variation."""
        
        strategy = {
            'id': str(uuid.uuid4()),
            'name': f"{variation.get('variation_type', 'unknown')}_{int(time.time())}",
            'category': variation.get('variation_type', 'general'),
            'strategy_data': {
                'original': self._compress_experience(original),
                'variation': self._compress_experience(variation),
                'changes': self._compute_changes(original, variation),
                'context': {
                    'material': original.get('primary_material'),
                    'action': original.get('action_type'),
                    'success_rate': 1.0 if variation.get('success', True) else 0.0
                }
            },
            'baseline_performance': original.get('reward', 0),
            'improved_performance': original.get('reward', 0) + improvement,
            'improvement_ratio': improvement,
            'confidence': min(0.5 + improvement, 1.0),
            'applicable_materials': [original.get('primary_material', 'unknown')],
            'applicable_scenarios': [original.get('action_type', 'unknown')],
            'neural_encoding': self._strategy_to_encoding({'improvement_ratio': improvement, 'category': variation.get('variation_type')}).tolist() if self._strategy_to_encoding({'improvement_ratio': improvement, 'category': variation.get('variation_type')}) is not None else None,
            'created_at': time.time(),
            'times_used': 0,
            'success_rate': 1.0
        }
        
        return strategy
        
    def _merge_strategies(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar strategies into consolidated strategy."""
        
        if len(strategies) == 1:
            return strategies[0]
            
        # Take strategy with best improvement as base
        best_strategy = max(strategies, key=lambda s: s.get('improvement_ratio', 0))
        merged = best_strategy.copy()
        
        # Merge applicable materials and scenarios
        all_materials = set()
        all_scenarios = set()
        
        for strategy in strategies:
            all_materials.update(strategy.get('applicable_materials', []))
            all_scenarios.update(strategy.get('applicable_scenarios', []))
            
        merged['applicable_materials'] = list(all_materials)
        merged['applicable_scenarios'] = list(all_scenarios)
        
        # Average confidence
        confidences = [s.get('confidence', 0) for s in strategies]
        merged['confidence'] = np.mean(confidences)
        
        # Update metadata
        merged['merged_from'] = len(strategies)
        merged['name'] = f"merged_{best_strategy['category']}_{int(time.time())}"
        
        return merged
        
    def _compress_experience(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Compress experience to essential data for strategy storage."""
        
        return {
            'material': exp.get('primary_material'),
            'action': exp.get('action_type'),
            'success': exp.get('success'),
            'reward': exp.get('reward'),
            'key_params': {
                k: v for k, v in exp.get('action_params', {}).items()
                if isinstance(v, (int, float, str))
            },
            'forces': exp.get('forces'),
            'timestamp': exp.get('timestamp')
        }
        
    def _compute_changes(self, original: Dict[str, Any], variation: Dict[str, Any]) -> Dict[str, Any]:
        """Compute what changed between original and variation."""
        
        changes = {}
        
        # Track changes in key fields
        key_fields = ['primary_material', 'forces', 'action_params', 'robot_velocity']
        
        for field in key_fields:
            if field in original and field in variation:
                orig_val = original[field]
                var_val = variation[field]
                
                if orig_val != var_val:
                    changes[field] = {
                        'from': orig_val,
                        'to': var_val,
                        'change_type': variation.get('variation_type', 'unknown')
                    }
                    
                    # Compute magnitude of change for numeric values
                    if isinstance(orig_val, (list, tuple)) and isinstance(var_val, (list, tuple)):
                        if all(isinstance(x, (int, float)) for x in orig_val + var_val):
                            orig_arr = np.array(orig_val)
                            var_arr = np.array(var_val)
                            if orig_arr.shape == var_arr.shape:
                                changes[field]['magnitude'] = float(np.linalg.norm(var_arr - orig_arr))
                                
        return changes
        
    def _store_strategies(self, strategies: List[Dict[str, Any]]) -> int:
        """Store consolidated strategies in database."""
        
        stored_count = 0
        
        for strategy in strategies[:50]:  # Limit to top 50 strategies
            try:
                # Convert numpy arrays to lists for JSON storage
                strategy_clean = self._clean_strategy_for_storage(strategy)
                
                # Store in database (assuming method exists)
                if hasattr(self.db, 'store_learned_strategy'):
                    self.db.store_learned_strategy(strategy_clean)
                    stored_count += 1
                else:
                    # Alternative storage method
                    logger.warning("Database does not support strategy storage")
                    
            except Exception as e:
                logger.error(f"Failed to store strategy: {e}")
                
        return stored_count
        
    def _clean_strategy_for_storage(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Clean strategy data for database storage."""
        
        clean_strategy = {}
        
        for key, value in strategy.items():
            if isinstance(value, np.ndarray):
                clean_strategy[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                clean_strategy[key] = float(value)
            elif isinstance(value, dict):
                clean_strategy[key] = self._clean_nested_dict(value)
            else:
                clean_strategy[key] = value
                
        return clean_strategy
        
    def _clean_nested_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively clean nested dictionary."""
        
        clean_dict = {}
        
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                clean_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                clean_dict[key] = float(value)
            elif isinstance(value, dict):
                clean_dict[key] = self._clean_nested_dict(value)
            else:
                clean_dict[key] = value
                
        return clean_dict
        
    def _compute_neural_convergence(self) -> float:
        """Compute neural network convergence metric."""
        
        try:
            # Simple convergence metric based on parameter variance
            total_variance = 0.0
            param_count = 0
            
            for param in self.neural_optimizer.parameters():
                if param.requires_grad:
                    param_var = torch.var(param).item()
                    total_variance += param_var
                    param_count += 1
                    
            if param_count > 0:
                avg_variance = total_variance / param_count
                # Convert to convergence (lower variance = higher convergence)
                convergence = 1.0 / (1.0 + avg_variance)
                return convergence
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to compute neural convergence: {e}")
            return 0.0
            
    def _update_global_metrics(self, session: DreamSession):
        """Update global performance metrics."""
        
        with self._lock:
            self.session_metrics['total_dreams'] += 1
            self.session_metrics['total_experiences_processed'] += session.experiences_processed
            self.session_metrics['total_strategies_found'] += session.strategies_discovered
            
            # Update running average of session time
            total_dreams = self.session_metrics['total_dreams']
            prev_avg = self.session_metrics['average_session_time']
            session_time = session.end_time - session.start_time if session.end_time else 0
            
            self.session_metrics['average_session_time'] = (
                (prev_avg * (total_dreams - 1) + session_time) / total_dreams
            )
            
    def _store_dream_session(self, session: DreamSession):
        """Store dream session in database."""
        
        try:
            session_data = {
                'session_id': session.session_id,
                'config': asdict(session.config),
                'experience_count': session.experiences_processed,
                'time_range_hours': 24,
                'replay_count': session.experiences_processed,
                'variations_generated': session.variations_generated,
                'compute_time_seconds': session.end_time - session.start_time if session.end_time else 0,
                'strategies_found': session.strategies_discovered,
                'consolidated_memories': session.strategies_consolidated,
                'average_improvement': session.average_improvement,
                'best_improvement': session.best_improvement,
                'neural_convergence': session.neural_convergence,
                'improvements': session.compute_metrics,
                'notes': f"Neural-enhanced dream session with {session.config.parallel_dreams} parallel threads"
            }
            
            if hasattr(self.db, 'store_dream_session'):
                self.db.store_dream_session(session_data)
            else:
                logger.warning("Database does not support dream session storage")
                
        except Exception as e:
            logger.error(f"Failed to store dream session: {e}")
            
    def _create_empty_session(self, session_id: str, start_time: float) -> DreamSession:
        """Create empty session when no experiences are found."""
        
        return DreamSession(
            session_id=session_id,
            start_time=start_time,
            end_time=time.time(),
            config=self.config,
            experiences_processed=0,
            variations_generated=0,
            strategies_discovered=0,
            strategies_consolidated=0,
            average_improvement=0,
            best_improvement=0,
            neural_convergence=0,
            compute_metrics={'session_duration': time.time() - start_time}
        )
        
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get global session metrics."""
        
        with self._lock:
            return self.session_metrics.copy()
            
    def reset_neural_network(self):
        """Reset neural network to initial state."""
        
        logger.info("Resetting neural strategy optimizer")
        
        # Reinitialize network
        self.neural_optimizer = NeuralStrategyOptimizer()
        self.optimizer = torch.optim.Adam(
            self.neural_optimizer.parameters(),
            lr=self.config.neural_learning_rate
        )
        
        # Clear strategy cache
        self.strategy_cache.clear()
        
    def save_neural_state(self, filepath: str):
        """Save neural network state to file."""
        
        try:
            torch.save({
                'model_state_dict': self.neural_optimizer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': asdict(self.config),
                'session_metrics': self.session_metrics
            }, filepath)
            
            logger.info(f"Neural state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save neural state: {e}")
            
    def load_neural_state(self, filepath: str):
        """Load neural network state from file."""
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.neural_optimizer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'session_metrics' in checkpoint:
                self.session_metrics = checkpoint['session_metrics']
                
            logger.info(f"Neural state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load neural state: {e}")
