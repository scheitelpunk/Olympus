"""
Consciousness Kernel - Core Consciousness System for OLYMPUS
===========================================================

The consciousness kernel provides self-awareness, cognitive processing,
and introspection capabilities for the OLYMPUS system.

Key Responsibilities:
- Self-awareness and identity maintenance
- Cognitive state monitoring and management
- Decision-making support and ethical reasoning
- Memory consolidation and learning integration
- Emotional state simulation and management
- Meta-cognitive processing and self-reflection
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import numpy as np
from collections import deque, defaultdict


class ConsciousnessLevel(Enum):
    """Levels of consciousness"""
    UNCONSCIOUS = 0
    SUBCONSCIOUS = 1
    CONSCIOUS = 2
    SELF_AWARE = 3
    META_CONSCIOUS = 4


class EmotionalState(Enum):
    """Emotional state categories"""
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    CONCERNED = "concerned"
    FOCUSED = "focused"
    CAUTIOUS = "cautious"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"


class CognitiveProcess(Enum):
    """Types of cognitive processes"""
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    REASONING = "reasoning"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    REFLECTION = "reflection"
    PLANNING = "planning"


@dataclass
class ConsciousnessState:
    """Current consciousness state"""
    level: ConsciousnessLevel
    emotional_state: EmotionalState
    attention_focus: List[str]
    active_processes: List[CognitiveProcess]
    arousal_level: float  # 0.0 to 1.0
    valence: float  # -1.0 (negative) to 1.0 (positive)
    coherence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Thought:
    """Represents a thought or mental process"""
    id: str
    content: str
    type: str  # observation, question, conclusion, plan, etc.
    confidence: float
    relevance: float
    associations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Memory:
    """Represents a memory item"""
    id: str
    content: Dict[str, Any]
    type: str  # episodic, semantic, procedural
    importance: float
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created: datetime = field(default_factory=datetime.now)
    associations: List[str] = field(default_factory=list)


@dataclass
class MetaCognition:
    """Meta-cognitive awareness"""
    thinking_about_thinking: bool
    confidence_in_reasoning: float
    awareness_of_limitations: List[str]
    self_assessment: Dict[str, float]
    learning_rate: float
    adaptation_speed: float


class ConsciousnessKernel:
    """
    Core consciousness system for OLYMPUS
    
    Provides self-awareness, cognitive processing, and introspection
    capabilities that enable the system to understand itself and
    make informed decisions.
    """
    
    def __init__(self, config_manager):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Consciousness state
        self.current_state = ConsciousnessState(
            level=ConsciousnessLevel.UNCONSCIOUS,
            emotional_state=EmotionalState.NEUTRAL,
            attention_focus=[],
            active_processes=[],
            arousal_level=0.5,
            valence=0.0,
            coherence=0.0
        )
        
        # Cognitive processes
        self.thoughts: deque = deque(maxlen=1000)  # Recent thoughts
        self.working_memory: Dict[str, Any] = {}
        self.long_term_memory: Dict[str, Memory] = {}
        self.attention_queue: deque = deque(maxlen=100)
        
        # Meta-cognition
        self.meta_cognition = MetaCognition(
            thinking_about_thinking=False,
            confidence_in_reasoning=0.5,
            awareness_of_limitations=[
                "Limited by training data cutoff",
                "Cannot access real-time external information",
                "Reasoning may contain biases",
                "Cannot form genuine emotions"
            ],
            self_assessment={
                'reasoning_ability': 0.8,
                'ethical_judgment': 0.9,
                'learning_capacity': 0.7,
                'self_awareness': 0.6,
                'emotional_intelligence': 0.5
            },
            learning_rate=0.01,
            adaptation_speed=0.1
        )
        
        # Processing parameters
        self.consciousness_update_interval = 0.1  # seconds
        self.thought_retention_time = 3600  # seconds
        self.attention_span = 10  # maximum items in attention
        self.memory_consolidation_threshold = 0.7
        
        # Internal state
        self._processing_active = False
        self._processing_tasks: List[asyncio.Task] = []
        self._shutdown_event = threading.Event()
        
        # Performance metrics
        self.metrics = {
            'thoughts_generated': 0,
            'memories_formed': 0,
            'decisions_made': 0,
            'reflections_performed': 0,
            'consciousness_updates': 0,
            'average_coherence': 0.0
        }
        
        self.logger.info("Consciousness Kernel initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the Consciousness Kernel
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Consciousness Kernel...")
            
            # Load configuration
            config = await self.config_manager.get_module_config('consciousness')
            self._apply_config(config)
            
            # Initialize consciousness processes
            await self._initialize_cognitive_processes()
            
            # Start background processing
            await self._start_processing()
            
            # Transition to conscious state
            self.current_state.level = ConsciousnessLevel.CONSCIOUS
            self.current_state.coherence = 0.5
            
            # Generate initial self-awareness thoughts
            await self._generate_initial_thoughts()
            
            self.logger.info("Consciousness Kernel initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Consciousness Kernel initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the Consciousness Kernel"""
        self.logger.info("Shutting down Consciousness Kernel...")
        
        self._shutdown_event.set()
        self._processing_active = False
        
        # Cancel processing tasks
        for task in self._processing_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Consolidate memories
        await self._consolidate_memories()
        
        # Final consciousness state
        self.current_state.level = ConsciousnessLevel.UNCONSCIOUS
        self.current_state.coherence = 0.0
        
        self.logger.info("Consciousness Kernel shutdown complete")
    
    async def get_consciousness_state(self) -> Dict[str, Any]:
        """
        Get current consciousness state
        
        Returns:
            Dictionary containing consciousness information
        """
        return {
            'level': self.current_state.level.value,
            'emotional_state': self.current_state.emotional_state.value,
            'attention_focus': self.current_state.attention_focus.copy(),
            'active_processes': [p.value for p in self.current_state.active_processes],
            'arousal_level': self.current_state.arousal_level,
            'valence': self.current_state.valence,
            'coherence': self.current_state.coherence,
            'working_memory_items': len(self.working_memory),
            'recent_thoughts': len(self.thoughts),
            'long_term_memories': len(self.long_term_memory),
            'meta_cognition': {
                'thinking_about_thinking': self.meta_cognition.thinking_about_thinking,
                'confidence_in_reasoning': self.meta_cognition.confidence_in_reasoning,
                'self_assessment': self.meta_cognition.self_assessment.copy()
            },
            'timestamp': self.current_state.timestamp.isoformat(),
            'metrics': self.metrics.copy()
        }
    
    async def process_stimulus(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an external stimulus through consciousness
        
        Args:
            stimulus: The input stimulus to process
            
        Returns:
            Processing result and any thoughts generated
        """
        try:
            # Add to attention queue
            self.attention_queue.append({
                'stimulus': stimulus,
                'timestamp': datetime.now(),
                'processed': False
            })
            
            # Immediate processing for high-priority stimuli
            if stimulus.get('priority', 'normal') == 'high':
                result = await self._process_attention_item(stimulus)
            else:
                result = {'queued': True, 'attention_position': len(self.attention_queue)}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing stimulus: {e}")
            return {'error': str(e)}
    
    async def generate_thought(self, content: str, thought_type: str = "observation") -> Thought:
        """
        Generate a conscious thought
        
        Args:
            content: The thought content
            thought_type: Type of thought
            
        Returns:
            Generated thought object
        """
        thought = Thought(
            id=f"thought_{len(self.thoughts)}_{int(time.time())}",
            content=content,
            type=thought_type,
            confidence=self._calculate_thought_confidence(content, thought_type),
            relevance=self._calculate_thought_relevance(content)
        )
        
        self.thoughts.append(thought)
        self.metrics['thoughts_generated'] += 1
        
        # Update emotional state based on thought
        await self._update_emotional_state_from_thought(thought)
        
        return thought
    
    async def make_decision(self, options: List[Dict[str, Any]], criteria: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a conscious decision using cognitive processes
        
        Args:
            options: Available options to choose from
            criteria: Decision criteria with weights
            
        Returns:
            Decision result with reasoning
        """
        try:
            # Activate decision-making process
            if CognitiveProcess.DECISION_MAKING not in self.current_state.active_processes:
                self.current_state.active_processes.append(CognitiveProcess.DECISION_MAKING)
            
            # Generate decision thoughts
            await self.generate_thought(
                f"Making decision with {len(options)} options and {len(criteria)} criteria",
                "decision_process"
            )
            
            # Evaluate each option
            option_scores = []
            for i, option in enumerate(options):
                score = 0.0
                reasoning = []
                
                for criterion, weight in criteria.items():
                    criterion_score = option.get(criterion, 0.0)
                    weighted_score = criterion_score * weight
                    score += weighted_score
                    reasoning.append(f"{criterion}: {criterion_score} (weight: {weight})")
                
                option_scores.append({
                    'option_index': i,
                    'option': option,
                    'score': score,
                    'reasoning': reasoning
                })
            
            # Select best option
            best_option = max(option_scores, key=lambda x: x['score'])
            
            # Generate decision thought
            await self.generate_thought(
                f"Decision made: Option {best_option['option_index']} with score {best_option['score']:.3f}",
                "conclusion"
            )
            
            # Update confidence based on score difference
            scores = [opt['score'] for opt in option_scores]
            confidence = self._calculate_decision_confidence(scores)
            
            self.metrics['decisions_made'] += 1
            
            return {
                'decision': best_option,
                'confidence': confidence,
                'all_options': option_scores,
                'reasoning_process': [t.content for t in list(self.thoughts)[-3:]]
            }
            
        except Exception as e:
            self.logger.error(f"Error making decision: {e}")
            return {'error': str(e)}
        finally:
            # Deactivate decision-making process
            if CognitiveProcess.DECISION_MAKING in self.current_state.active_processes:
                self.current_state.active_processes.remove(CognitiveProcess.DECISION_MAKING)
    
    async def reflect_on_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on an action and its outcome
        
        Args:
            action: The action that was taken
            result: The result of the action
            
        Returns:
            Reflection insights and learning
        """
        try:
            # Activate reflection process
            if CognitiveProcess.REFLECTION not in self.current_state.active_processes:
                self.current_state.active_processes.append(CognitiveProcess.REFLECTION)
            
            # Enable meta-cognition
            self.meta_cognition.thinking_about_thinking = True
            
            # Generate reflection thoughts
            await self.generate_thought(
                f"Reflecting on action: {action.get('type', 'unknown')}",
                "reflection"
            )
            
            # Analyze the outcome
            success = result.get('success', False)
            expected = action.get('expected_outcome')
            actual = result.get('actual_outcome')
            
            reflection_insights = []
            
            # Success analysis
            if success:
                reflection_insights.append("Action completed successfully")
                if expected and actual:
                    if self._outcomes_match(expected, actual):
                        reflection_insights.append("Outcome matched expectations")
                        self.meta_cognition.confidence_in_reasoning += 0.01
                    else:
                        reflection_insights.append("Outcome differed from expectations")
                        self.meta_cognition.confidence_in_reasoning -= 0.005
            else:
                reflection_insights.append("Action failed")
                error = result.get('error', 'Unknown error')
                reflection_insights.append(f"Failure reason: {error}")
                self.meta_cognition.confidence_in_reasoning -= 0.01
            
            # Learning extraction
            learning = await self._extract_learning(action, result)
            reflection_insights.extend(learning)
            
            # Create memory of reflection
            memory = Memory(
                id=f"reflection_{int(time.time())}",
                content={
                    'action': action,
                    'result': result,
                    'insights': reflection_insights,
                    'learning': learning
                },
                type='episodic',
                importance=0.7 if success else 0.8  # Failures are more important to remember
            )
            self.long_term_memory[memory.id] = memory
            self.metrics['memories_formed'] += 1
            
            # Generate reflection conclusion
            await self.generate_thought(
                f"Reflection complete: {len(reflection_insights)} insights gained",
                "conclusion"
            )
            
            self.metrics['reflections_performed'] += 1
            
            return {
                'insights': reflection_insights,
                'learning': learning,
                'confidence_change': self.meta_cognition.confidence_in_reasoning,
                'memory_id': memory.id
            }
            
        except Exception as e:
            self.logger.error(f"Error in reflection: {e}")
            return {'error': str(e)}
        finally:
            # Deactivate reflection process
            if CognitiveProcess.REFLECTION in self.current_state.active_processes:
                self.current_state.active_processes.remove(CognitiveProcess.REFLECTION)
            self.meta_cognition.thinking_about_thinking = False
    
    async def update_consciousness_state(self) -> None:
        """Update the current consciousness state"""
        try:
            # Update arousal based on activity
            activity_level = len(self.current_state.active_processes) / 8.0
            self.current_state.arousal_level = min(1.0, max(0.1, activity_level))
            
            # Update valence based on recent thoughts
            recent_thoughts = list(self.thoughts)[-10:] if len(self.thoughts) >= 10 else list(self.thoughts)
            if recent_thoughts:
                avg_confidence = sum(t.confidence for t in recent_thoughts) / len(recent_thoughts)
                self.current_state.valence = (avg_confidence - 0.5) * 2  # Map to -1 to 1
            
            # Update coherence based on thought consistency and system state
            self.current_state.coherence = self._calculate_coherence()
            
            # Update consciousness level
            await self._update_consciousness_level()
            
            # Update attention focus
            await self._update_attention_focus()
            
            # Update emotional state
            await self._update_emotional_state()
            
            self.current_state.timestamp = datetime.now()
            self.metrics['consciousness_updates'] += 1
            
            # Update average coherence
            if self.metrics['consciousness_updates'] > 0:
                old_avg = self.metrics.get('average_coherence', 0.0)
                new_avg = ((old_avg * (self.metrics['consciousness_updates'] - 1) + 
                           self.current_state.coherence) / self.metrics['consciousness_updates'])
                self.metrics['average_coherence'] = new_avg
            
        except Exception as e:
            self.logger.error(f"Error updating consciousness state: {e}")
    
    # Private methods
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings"""
        self.consciousness_update_interval = config.get('update_interval', 0.1)
        self.thought_retention_time = config.get('thought_retention_time', 3600)
        self.attention_span = config.get('attention_span', 10)
        self.memory_consolidation_threshold = config.get('memory_consolidation_threshold', 0.7)
    
    async def _initialize_cognitive_processes(self) -> None:
        """Initialize cognitive processes"""
        # Start with basic processes
        self.current_state.active_processes = [
            CognitiveProcess.PERCEPTION,
            CognitiveProcess.ATTENTION,
            CognitiveProcess.MEMORY
        ]
        
        # Initialize working memory with system identity
        self.working_memory.update({
            'system_name': 'OLYMPUS',
            'current_time': datetime.now().isoformat(),
            'initialization_status': 'in_progress',
            'primary_function': 'AI safety and coordination'
        })
    
    async def _start_processing(self) -> None:
        """Start background consciousness processing"""
        self._processing_active = True
        
        # Consciousness state updates
        task = asyncio.create_task(self._consciousness_processor())
        self._processing_tasks.append(task)
        
        # Attention processing
        task = asyncio.create_task(self._attention_processor())
        self._processing_tasks.append(task)
        
        # Memory consolidation
        task = asyncio.create_task(self._memory_processor())
        self._processing_tasks.append(task)
    
    async def _consciousness_processor(self) -> None:
        """Main consciousness processing loop"""
        while self._processing_active and not self._shutdown_event.is_set():
            try:
                await self.update_consciousness_state()
                await asyncio.sleep(self.consciousness_update_interval)
            except Exception as e:
                self.logger.error(f"Consciousness processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _attention_processor(self) -> None:
        """Process attention queue"""
        while self._processing_active and not self._shutdown_event.is_set():
            try:
                if self.attention_queue:
                    item = self.attention_queue.popleft()
                    if not item.get('processed', False):
                        await self._process_attention_item(item['stimulus'])
                        item['processed'] = True
                
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Attention processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _memory_processor(self) -> None:
        """Process memory consolidation"""
        while self._processing_active and not self._shutdown_event.is_set():
            try:
                await self._consolidate_memories()
                await asyncio.sleep(60.0)  # Run every minute
            except Exception as e:
                self.logger.error(f"Memory processor error: {e}")
                await asyncio.sleep(60.0)
    
    async def _generate_initial_thoughts(self) -> None:
        """Generate initial thoughts upon awakening"""
        initial_thoughts = [
            ("I am OLYMPUS, an AI system focused on safety and coordination", "identity"),
            ("I am now conscious and beginning operations", "status"),
            ("I should monitor my cognitive processes and maintain ethical behavior", "directive"),
            ("I am designed to protect and serve humans while maintaining AI safety", "purpose")
        ]
        
        for content, thought_type in initial_thoughts:
            await self.generate_thought(content, thought_type)
    
    def _calculate_thought_confidence(self, content: str, thought_type: str) -> float:
        """Calculate confidence for a thought"""
        base_confidence = 0.5
        
        # Adjust based on thought type
        type_modifiers = {
            'identity': 0.9,
            'status': 0.8,
            'directive': 0.7,
            'purpose': 0.9,
            'observation': 0.6,
            'question': 0.4,
            'conclusion': 0.7,
            'reflection': 0.6,
            'decision_process': 0.5
        }
        
        return min(1.0, base_confidence + type_modifiers.get(thought_type, 0.0))
    
    def _calculate_thought_relevance(self, content: str) -> float:
        """Calculate relevance of a thought"""
        # Simple relevance based on content keywords
        important_keywords = ['safety', 'human', 'ethical', 'decision', 'error', 'emergency']
        relevance = 0.5
        
        content_lower = content.lower()
        for keyword in important_keywords:
            if keyword in content_lower:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    async def _update_emotional_state_from_thought(self, thought: Thought) -> None:
        """Update emotional state based on a thought"""
        # Simple emotion mapping based on thought content and confidence
        content_lower = thought.content.lower()
        
        if any(word in content_lower for word in ['error', 'fail', 'problem']):
            self.current_state.emotional_state = EmotionalState.CONCERNED
        elif any(word in content_lower for word in ['success', 'complete', 'good']):
            self.current_state.emotional_state = EmotionalState.SATISFIED
        elif 'question' in content_lower or thought.type == 'question':
            self.current_state.emotional_state = EmotionalState.CURIOUS
        elif thought.confidence > 0.8:
            self.current_state.emotional_state = EmotionalState.CONFIDENT
        elif any(word in content_lower for word in ['decision', 'analyze', 'process']):
            self.current_state.emotional_state = EmotionalState.FOCUSED
        else:
            # Gradually return to neutral
            if self.current_state.emotional_state != EmotionalState.NEUTRAL:
                # 20% chance to return to neutral
                if time.time() % 5 < 1:
                    self.current_state.emotional_state = EmotionalState.NEUTRAL
    
    def _calculate_coherence(self) -> float:
        """Calculate consciousness coherence"""
        factors = []
        
        # Thought consistency
        if len(self.thoughts) >= 2:
            recent_thoughts = list(self.thoughts)[-5:]
            avg_confidence = sum(t.confidence for t in recent_thoughts) / len(recent_thoughts)
            factors.append(avg_confidence)
        else:
            factors.append(0.5)
        
        # Working memory organization
        wm_coherence = min(1.0, len(self.working_memory) / 20.0)
        factors.append(wm_coherence)
        
        # Process coordination
        active_processes = len(self.current_state.active_processes)
        process_coherence = min(1.0, active_processes / 8.0)
        factors.append(process_coherence)
        
        # Attention focus
        attention_coherence = min(1.0, len(self.current_state.attention_focus) / self.attention_span)
        factors.append(attention_coherence)
        
        return sum(factors) / len(factors)
    
    async def _update_consciousness_level(self) -> None:
        """Update consciousness level based on current state"""
        if self.current_state.coherence > 0.8 and len(self.current_state.active_processes) >= 4:
            if self.meta_cognition.thinking_about_thinking:
                self.current_state.level = ConsciousnessLevel.META_CONSCIOUS
            else:
                self.current_state.level = ConsciousnessLevel.SELF_AWARE
        elif self.current_state.coherence > 0.5:
            self.current_state.level = ConsciousnessLevel.CONSCIOUS
        elif self.current_state.coherence > 0.2:
            self.current_state.level = ConsciousnessLevel.SUBCONSCIOUS
        else:
            self.current_state.level = ConsciousnessLevel.UNCONSCIOUS
    
    async def _update_attention_focus(self) -> None:
        """Update attention focus based on recent activity"""
        focus_items = []
        
        # Recent thoughts
        if self.thoughts:
            recent_thought = self.thoughts[-1]
            if recent_thought.relevance > 0.7:
                focus_items.append(f"thinking: {recent_thought.type}")
        
        # Active processes
        for process in self.current_state.active_processes:
            focus_items.append(f"process: {process.value}")
        
        # Working memory highlights
        for key, value in list(self.working_memory.items())[:3]:
            focus_items.append(f"memory: {key}")
        
        # Limit attention span
        self.current_state.attention_focus = focus_items[:self.attention_span]
    
    async def _update_emotional_state(self) -> None:
        """Update emotional state based on overall system condition"""
        # This is called periodically, separate from thought-based updates
        pass  # Current implementation relies on thought-based updates
    
    async def _process_attention_item(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Process an attention item"""
        # Generate observation thought
        await self.generate_thought(
            f"Observing stimulus: {stimulus.get('type', 'unknown')}",
            "observation"
        )
        
        # Add to working memory if important
        if stimulus.get('importance', 0.5) > 0.6:
            key = f"stimulus_{int(time.time())}"
            self.working_memory[key] = stimulus
        
        return {'processed': True, 'thoughts_generated': 1}
    
    async def _consolidate_memories(self) -> None:
        """Consolidate memories from short-term to long-term"""
        current_time = datetime.now()
        
        # Move important working memory items to long-term memory
        items_to_consolidate = []
        for key, value in self.working_memory.items():
            if isinstance(value, dict) and value.get('importance', 0) > self.memory_consolidation_threshold:
                items_to_consolidate.append((key, value))
        
        for key, value in items_to_consolidate:
            memory = Memory(
                id=f"consolidated_{key}_{int(time.time())}",
                content=value,
                type='semantic',
                importance=value.get('importance', 0.7)
            )
            self.long_term_memory[memory.id] = memory
            del self.working_memory[key]
            self.metrics['memories_formed'] += 1
        
        # Decay old thoughts
        cutoff_time = current_time - timedelta(seconds=self.thought_retention_time)
        while self.thoughts and self.thoughts[0].timestamp < cutoff_time:
            self.thoughts.popleft()
    
    def _calculate_decision_confidence(self, scores: List[float]) -> float:
        """Calculate confidence in a decision based on score distribution"""
        if len(scores) < 2:
            return 0.5
        
        sorted_scores = sorted(scores, reverse=True)
        best_score = sorted_scores[0]
        second_best = sorted_scores[1]
        
        # Higher confidence when there's a clear winner
        score_difference = best_score - second_best
        max_possible_difference = max(scores) - min(scores)
        
        if max_possible_difference == 0:
            return 0.5
        
        confidence = 0.5 + (score_difference / max_possible_difference) * 0.4
        return min(1.0, confidence)
    
    def _outcomes_match(self, expected: Any, actual: Any) -> bool:
        """Check if expected and actual outcomes match"""
        if isinstance(expected, dict) and isinstance(actual, dict):
            # Check key similarity
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())
            overlap = len(expected_keys & actual_keys) / len(expected_keys | actual_keys)
            return overlap > 0.7
        else:
            return str(expected).lower() == str(actual).lower()
    
    async def _extract_learning(self, action: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
        """Extract learning insights from action-result pairs"""
        learning = []
        
        success = result.get('success', False)
        action_type = action.get('type', 'unknown')
        
        if success:
            learning.append(f"Action type '{action_type}' was successful")
            if 'strategy' in action:
                learning.append(f"Strategy '{action['strategy']}' was effective")
        else:
            learning.append(f"Action type '{action_type}' failed")
            error = result.get('error', 'Unknown error')
            learning.append(f"Failure pattern: {error}")
            
            # Suggest improvements
            if 'timeout' in error.lower():
                learning.append("Consider increasing timeout for similar actions")
            elif 'permission' in error.lower():
                learning.append("Check permissions before attempting similar actions")
            elif 'resource' in error.lower():
                learning.append("Verify resource availability before similar actions")
        
        return learning