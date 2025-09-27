"""
Few-Shot Learning Module

Enables rapid learning from very few examples (typically 1-5 shots) using
advanced few-shot learning techniques with safety validation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class FewShotMethod(Enum):
    ONE_SHOT = "one_shot"
    THREE_SHOT = "three_shot"  
    FIVE_SHOT = "five_shot"
    VARIABLE_SHOT = "variable_shot"

class FewShotAlgorithm(Enum):
    PROTOTYPICAL = "prototypical_networks"
    SIAMESE = "siamese_networks"
    MATCHING = "matching_networks"
    RELATION = "relation_networks"
    MAML = "model_agnostic_meta_learning"
    MEMORY_AUGMENTED = "memory_augmented"

@dataclass
class FewShotExample:
    """Single example for few-shot learning"""
    id: str
    data: Any
    label: str
    domain: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class FewShotTask:
    """Few-shot learning task definition"""
    task_id: str
    name: str
    domain: str
    target_classes: List[str]
    support_examples: List[FewShotExample]  # Few examples for learning
    query_examples: List[FewShotExample]    # Examples for testing
    difficulty: float
    safety_requirements: str

@dataclass
class FewShotResult:
    """Result of few-shot learning"""
    success: bool
    learned_model: Any
    accuracy: float
    confidence: float
    learning_time: float
    examples_used: int
    predictions: List[Dict[str, Any]]
    safety_validated: bool
    warnings: List[str]
    metadata: Dict[str, Any]

class FewShotLearner:
    """Few-shot learning system with safety validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get('min_confidence', 0.7)
        self.safety_threshold = config.get('safety_threshold', 0.8)
        self.learning_history = []
        
        # Few-shot algorithms
        self.algorithms = {
            FewShotAlgorithm.PROTOTYPICAL: self._prototypical_learning,
            FewShotAlgorithm.SIAMESE: self._siamese_learning,
            FewShotAlgorithm.MATCHING: self._matching_learning,
            FewShotAlgorithm.RELATION: self._relation_learning,
            FewShotAlgorithm.MAML: self._maml_learning,
            FewShotAlgorithm.MEMORY_AUGMENTED: self._memory_augmented_learning
        }
        
    async def learn_few_shot(self, task: FewShotTask,
                            method: FewShotMethod = FewShotMethod.FIVE_SHOT,
                            algorithm: FewShotAlgorithm = FewShotAlgorithm.PROTOTYPICAL) -> FewShotResult:
        """Learn from few examples with safety validation"""
        
        logger.info(f"Starting few-shot learning: {task.name} using {method.value} with {algorithm.value}")
        
        start_time = datetime.now()
        
        # Validate task and examples
        validation_result = await self._validate_few_shot_task(task)
        if not validation_result["valid"]:
            return FewShotResult(
                success=False,
                learned_model=None,
                accuracy=0.0,
                confidence=0.0,
                learning_time=0.0,
                examples_used=0,
                predictions=[],
                safety_validated=False,
                warnings=[f"Task validation failed: {validation_result['reason']}"],
                metadata={}
            )
        
        # Select appropriate examples based on method
        selected_examples = await self._select_examples_for_method(
            task.support_examples, method
        )
        
        if len(selected_examples) == 0:
            return FewShotResult(
                success=False,
                learned_model=None,
                accuracy=0.0,
                confidence=0.0,
                learning_time=0.0,
                examples_used=0,
                predictions=[],
                safety_validated=False,
                warnings=["No valid examples found for learning"],
                metadata={}
            )
        
        # Safety check on examples
        safety_check = await self._safety_check_examples(selected_examples, task)
        if not safety_check.is_safe:
            return FewShotResult(
                success=False,
                learned_model=None,
                accuracy=0.0,
                confidence=0.0,
                learning_time=0.0,
                examples_used=0,
                predictions=[],
                safety_validated=False,
                warnings=[f"Examples failed safety check: {safety_check.reason}"],
                metadata={}
            )
        
        try:
            # Execute few-shot learning algorithm
            algorithm_func = self.algorithms[algorithm]
            learning_result = await algorithm_func(selected_examples, task)
            
            # Evaluate learned model
            evaluation_result = await self._evaluate_few_shot_model(
                learning_result["model"], task.query_examples
            )
            
            # Calculate learning metrics
            learning_time = (datetime.now() - start_time).total_seconds()
            
            # Validate learned model safety
            model_safety = await self._validate_model_safety(
                learning_result["model"], task
            )
            
            # Calculate overall confidence
            overall_confidence = min(
                learning_result.get("confidence", 0.8),
                evaluation_result["confidence"],
                model_safety["safety_score"]
            )
            
            # Record learning attempt
            await self._record_few_shot_learning(
                task, method, algorithm, learning_result, evaluation_result
            )
            
            return FewShotResult(
                success=evaluation_result["accuracy"] > 0.5,  # Minimum success threshold
                learned_model=learning_result["model"],
                accuracy=evaluation_result["accuracy"],
                confidence=overall_confidence,
                learning_time=learning_time,
                examples_used=len(selected_examples),
                predictions=evaluation_result["predictions"],
                safety_validated=model_safety["is_safe"],
                warnings=model_safety.get("warnings", []),
                metadata={
                    "method": method.value,
                    "algorithm": algorithm.value,
                    "task_difficulty": task.difficulty,
                    "learning_details": learning_result.get("details", {})
                }
            )
            
        except Exception as e:
            logger.error(f"Few-shot learning failed: {e}")
            learning_time = (datetime.now() - start_time).total_seconds()
            
            return FewShotResult(
                success=False,
                learned_model=None,
                accuracy=0.0,
                confidence=0.0,
                learning_time=learning_time,
                examples_used=len(selected_examples),
                predictions=[],
                safety_validated=False,
                warnings=[f"Learning failed: {str(e)}"],
                metadata={}
            )
    
    async def _prototypical_learning(self, examples: List[FewShotExample],
                                   task: FewShotTask) -> Dict[str, Any]:
        """Prototypical Networks few-shot learning"""
        
        # Compute embeddings for examples
        embeddings = await self._compute_example_embeddings(examples)
        
        # Compute class prototypes
        prototypes = {}
        for class_name in task.target_classes:
            class_examples = [e for e in examples if e.label == class_name]
            if class_examples:
                class_embeddings = [embeddings[e.id] for e in class_examples]
                prototype = await self._compute_prototype(class_embeddings)
                prototypes[class_name] = prototype
        
        # Create prototypical model
        model = {
            "type": "prototypical",
            "prototypes": prototypes,
            "embedding_function": await self._get_embedding_function(),
            "distance_metric": "euclidean"
        }
        
        # Estimate confidence based on prototype separation
        confidence = await self._estimate_prototype_confidence(prototypes)
        
        return {
            "model": model,
            "confidence": confidence,
            "details": {
                "num_prototypes": len(prototypes),
                "prototype_separation": await self._compute_prototype_separation(prototypes)
            }
        }
    
    async def _siamese_learning(self, examples: List[FewShotExample],
                              task: FewShotTask) -> Dict[str, Any]:
        """Siamese Networks few-shot learning"""
        
        # Create pairs for siamese training
        positive_pairs = await self._create_positive_pairs(examples)
        negative_pairs = await self._create_negative_pairs(examples)
        
        # Train siamese network
        siamese_model = await self._train_siamese_network(positive_pairs, negative_pairs)
        
        # Create similarity-based classifier
        model = {
            "type": "siamese",
            "siamese_network": siamese_model,
            "support_examples": examples,
            "similarity_threshold": await self._compute_similarity_threshold(examples)
        }
        
        # Estimate confidence based on training performance
        confidence = await self._estimate_siamese_confidence(siamese_model, examples)
        
        return {
            "model": model,
            "confidence": confidence,
            "details": {
                "positive_pairs": len(positive_pairs),
                "negative_pairs": len(negative_pairs),
                "similarity_threshold": model["similarity_threshold"]
            }
        }
    
    async def _matching_learning(self, examples: List[FewShotExample],
                               task: FewShotTask) -> Dict[str, Any]:
        """Matching Networks few-shot learning"""
        
        # Encode support set
        support_encodings = await self._encode_support_set(examples)
        
        # Create attention mechanism
        attention_model = await self._create_attention_model(support_encodings)
        
        # Create matching model
        model = {
            "type": "matching",
            "support_encodings": support_encodings,
            "attention_model": attention_model,
            "support_examples": examples
        }
        
        # Estimate confidence based on attention weights distribution
        confidence = await self._estimate_matching_confidence(attention_model, examples)
        
        return {
            "model": model,
            "confidence": confidence,
            "details": {
                "support_size": len(examples),
                "encoding_dimension": len(support_encodings[0]) if support_encodings else 0
            }
        }
    
    async def _relation_learning(self, examples: List[FewShotExample],
                               task: FewShotTask) -> Dict[str, Any]:
        """Relation Networks few-shot learning"""
        
        # Compute feature representations
        feature_representations = await self._compute_feature_representations(examples)
        
        # Train relation module
        relation_module = await self._train_relation_module(feature_representations, task)
        
        # Create relation model
        model = {
            "type": "relation",
            "feature_extractor": await self._get_feature_extractor(),
            "relation_module": relation_module,
            "support_features": feature_representations
        }
        
        # Estimate confidence based on relation scores
        confidence = await self._estimate_relation_confidence(relation_module, examples)
        
        return {
            "model": model,
            "confidence": confidence,
            "details": {
                "feature_dim": len(feature_representations[0]) if feature_representations else 0,
                "relation_score_range": await self._get_relation_score_range(relation_module)
            }
        }
    
    async def _maml_learning(self, examples: List[FewShotExample],
                           task: FewShotTask) -> Dict[str, Any]:
        """MAML-based few-shot learning"""
        
        # Initialize with meta-learned parameters
        meta_parameters = await self._get_meta_parameters()
        
        # Perform inner loop adaptation
        adapted_parameters = await self._inner_loop_adapt(meta_parameters, examples)
        
        # Create MAML model
        model = {
            "type": "maml",
            "adapted_parameters": adapted_parameters,
            "meta_parameters": meta_parameters,
            "adaptation_steps": self.config.get('maml_adaptation_steps', 5)
        }
        
        # Estimate confidence based on adaptation loss
        confidence = await self._estimate_maml_confidence(adapted_parameters, examples)
        
        return {
            "model": model,
            "confidence": confidence,
            "details": {
                "adaptation_loss": await self._compute_adaptation_loss(adapted_parameters, examples),
                "parameter_change": await self._compute_parameter_change(meta_parameters, adapted_parameters)
            }
        }
    
    async def _memory_augmented_learning(self, examples: List[FewShotExample],
                                       task: FewShotTask) -> Dict[str, Any]:
        """Memory-augmented few-shot learning"""
        
        # Initialize external memory
        memory = await self._initialize_memory(examples)
        
        # Train memory controller
        controller = await self._train_memory_controller(examples, memory)
        
        # Create memory-augmented model
        model = {
            "type": "memory_augmented",
            "memory": memory,
            "controller": controller,
            "memory_size": len(memory),
            "read_heads": self.config.get('memory_read_heads', 4)
        }
        
        # Estimate confidence based on memory utilization
        confidence = await self._estimate_memory_confidence(controller, memory)
        
        return {
            "model": model,
            "confidence": confidence,
            "details": {
                "memory_utilization": await self._compute_memory_utilization(memory),
                "controller_accuracy": await self._evaluate_controller(controller, examples)
            }
        }
    
    async def _select_examples_for_method(self, examples: List[FewShotExample],
                                        method: FewShotMethod) -> List[FewShotExample]:
        """Select appropriate number of examples based on method"""
        
        if method == FewShotMethod.ONE_SHOT:
            target_count = 1
        elif method == FewShotMethod.THREE_SHOT:
            target_count = 3
        elif method == FewShotMethod.FIVE_SHOT:
            target_count = 5
        else:  # VARIABLE_SHOT
            target_count = min(len(examples), 10)  # Use up to 10 examples
        
        # Select best examples based on confidence and diversity
        selected = await self._select_best_examples(examples, target_count)
        
        return selected
    
    async def _select_best_examples(self, examples: List[FewShotExample],
                                  count: int) -> List[FewShotExample]:
        """Select best examples based on confidence and diversity"""
        
        if len(examples) <= count:
            return examples
        
        # Sort by confidence
        sorted_examples = sorted(examples, key=lambda x: x.confidence, reverse=True)
        
        # Select diverse high-confidence examples
        selected = []
        for example in sorted_examples:
            if len(selected) >= count:
                break
                
            # Check diversity with already selected examples
            is_diverse = await self._check_example_diversity(example, selected)
            if is_diverse or len(selected) == 0:
                selected.append(example)
        
        return selected
    
    async def _evaluate_few_shot_model(self, model: Dict[str, Any],
                                     query_examples: List[FewShotExample]) -> Dict[str, Any]:
        """Evaluate few-shot learned model"""
        
        predictions = []
        correct_predictions = 0
        
        for example in query_examples:
            # Make prediction using the model
            prediction = await self._make_prediction(model, example)
            
            predictions.append({
                "example_id": example.id,
                "predicted_label": prediction["label"],
                "confidence": prediction["confidence"],
                "true_label": example.label,
                "correct": prediction["label"] == example.label
            })
            
            if prediction["label"] == example.label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(query_examples) if query_examples else 0.0
        
        # Calculate confidence based on prediction confidences
        avg_confidence = np.mean([p["confidence"] for p in predictions]) if predictions else 0.0
        
        return {
            "accuracy": accuracy,
            "confidence": avg_confidence,
            "predictions": predictions,
            "correct_count": correct_predictions,
            "total_count": len(query_examples)
        }
    
    async def _make_prediction(self, model: Dict[str, Any],
                             example: FewShotExample) -> Dict[str, Any]:
        """Make prediction using few-shot model"""
        
        model_type = model["type"]
        
        if model_type == "prototypical":
            return await self._predict_prototypical(model, example)
        elif model_type == "siamese":
            return await self._predict_siamese(model, example)
        elif model_type == "matching":
            return await self._predict_matching(model, example)
        elif model_type == "relation":
            return await self._predict_relation(model, example)
        elif model_type == "maml":
            return await self._predict_maml(model, example)
        elif model_type == "memory_augmented":
            return await self._predict_memory_augmented(model, example)
        else:
            return {"label": "unknown", "confidence": 0.0}
    
    # Helper methods (mock implementations for brevity)
    async def _validate_few_shot_task(self, task):
        return {"valid": True, "reason": ""}
    
    async def _safety_check_examples(self, examples, task):
        class SafetyCheck:
            def __init__(self):
                self.is_safe = True
                self.reason = ""
        return SafetyCheck()
    
    async def _validate_model_safety(self, model, task):
        return {"is_safe": True, "safety_score": 0.9, "warnings": []}
    
    async def _record_few_shot_learning(self, task, method, algorithm, learning_result, eval_result):
        record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "method": method.value,
            "algorithm": algorithm.value,
            "accuracy": eval_result["accuracy"],
            "confidence": learning_result.get("confidence", 0),
            "examples_used": len(task.support_examples)
        }
        self.learning_history.append(record)
    
    # Mock implementations for algorithm-specific methods
    async def _compute_example_embeddings(self, examples): 
        return {e.id: np.random.rand(128) for e in examples}
    async def _compute_prototype(self, embeddings): 
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(128)
    async def _get_embedding_function(self): 
        return {"type": "neural_embedding", "dimension": 128}
    async def _estimate_prototype_confidence(self, prototypes): 
        return 0.8
    async def _compute_prototype_separation(self, prototypes): 
        return 0.7
    async def _create_positive_pairs(self, examples): 
        return [(e, e) for e in examples[:5]]
    async def _create_negative_pairs(self, examples): 
        return [(examples[i], examples[j]) for i in range(min(3, len(examples))) for j in range(i+1, min(5, len(examples)))]
    async def _train_siamese_network(self, pos_pairs, neg_pairs): 
        return {"trained": True, "pairs": len(pos_pairs) + len(neg_pairs)}
    async def _compute_similarity_threshold(self, examples): 
        return 0.75
    async def _estimate_siamese_confidence(self, model, examples): 
        return 0.82
    async def _encode_support_set(self, examples): 
        return [np.random.rand(64) for _ in examples]
    async def _create_attention_model(self, encodings): 
        return {"attention_weights": np.random.rand(len(encodings))}
    async def _estimate_matching_confidence(self, model, examples): 
        return 0.79
    async def _compute_feature_representations(self, examples): 
        return [np.random.rand(256) for _ in examples]
    async def _train_relation_module(self, features, task): 
        return {"trained": True, "features": len(features)}
    async def _get_feature_extractor(self): 
        return {"type": "cnn", "output_dim": 256}
    async def _estimate_relation_confidence(self, model, examples): 
        return 0.81
    async def _get_relation_score_range(self, model): 
        return [0.0, 1.0]
    async def _get_meta_parameters(self): 
        return {"meta_learned": True, "params": np.random.rand(100)}
    async def _inner_loop_adapt(self, meta_params, examples): 
        return {"adapted": True, "steps": 5}
    async def _estimate_maml_confidence(self, params, examples): 
        return 0.83
    async def _compute_adaptation_loss(self, params, examples): 
        return 0.15
    async def _compute_parameter_change(self, meta_params, adapted_params): 
        return 0.3
    async def _initialize_memory(self, examples): 
        return {"memory_slots": len(examples) * 2, "content": "initialized"}
    async def _train_memory_controller(self, examples, memory): 
        return {"controller": "trained", "memory_size": len(examples)}
    async def _estimate_memory_confidence(self, controller, memory): 
        return 0.84
    async def _compute_memory_utilization(self, memory): 
        return 0.6
    async def _evaluate_controller(self, controller, examples): 
        return 0.88
    async def _check_example_diversity(self, example, selected): 
        return len(selected) < 3 or np.random.rand() > 0.5
    
    # Prediction methods
    async def _predict_prototypical(self, model, example): 
        return {"label": "predicted_class", "confidence": 0.85}
    async def _predict_siamese(self, model, example): 
        return {"label": "predicted_class", "confidence": 0.82}
    async def _predict_matching(self, model, example): 
        return {"label": "predicted_class", "confidence": 0.79}
    async def _predict_relation(self, model, example): 
        return {"label": "predicted_class", "confidence": 0.81}
    async def _predict_maml(self, model, example): 
        return {"label": "predicted_class", "confidence": 0.83}
    async def _predict_memory_augmented(self, model, example): 
        return {"label": "predicted_class", "confidence": 0.84}

    async def get_learning_history(self) -> List[Dict[str, Any]]:
        """Get few-shot learning history"""
        return self.learning_history.copy()
    
    async def create_few_shot_task(self, name: str, domain: str, 
                                 examples: List[FewShotExample],
                                 test_split: float = 0.3) -> FewShotTask:
        """Create a few-shot task from examples"""
        
        # Split examples into support and query sets
        split_idx = int(len(examples) * (1 - test_split))
        support_examples = examples[:split_idx]
        query_examples = examples[split_idx:]
        
        # Extract target classes
        target_classes = list(set(e.label for e in examples))
        
        # Assess task difficulty
        difficulty = await self._assess_task_difficulty(examples, target_classes)
        
        return FewShotTask(
            task_id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            domain=domain,
            target_classes=target_classes,
            support_examples=support_examples,
            query_examples=query_examples,
            difficulty=difficulty,
            safety_requirements="standard"
        )
    
    async def _assess_task_difficulty(self, examples: List[FewShotExample],
                                    target_classes: List[str]) -> float:
        """Assess difficulty of few-shot task"""
        
        # Factors affecting difficulty
        num_classes = len(target_classes)
        examples_per_class = len(examples) / num_classes if num_classes > 0 else 0
        avg_confidence = np.mean([e.confidence for e in examples]) if examples else 0
        
        # Simple difficulty calculation
        difficulty = 0.5  # Base difficulty
        difficulty += min(0.3, num_classes / 20)  # More classes = harder
        difficulty -= min(0.2, examples_per_class / 10)  # More examples per class = easier
        difficulty -= (avg_confidence - 0.5) * 0.4  # Higher confidence examples = easier
        
        return max(0.0, min(1.0, difficulty))