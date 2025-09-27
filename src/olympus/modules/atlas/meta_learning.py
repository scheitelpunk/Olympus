"""
Meta-Learning Module

Implements learn-to-learn capabilities enabling rapid adaptation to new tasks
with minimal training data through meta-learning algorithms.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class MetaLearningAlgorithm(Enum):
    MAML = "model_agnostic_meta_learning"
    REPTILE = "reptile"
    PROTOTYPICAL = "prototypical_networks"
    RELATION_NET = "relation_networks"
    MATCHING_NET = "matching_networks"
    GRADIENT_BASED = "gradient_based"

@dataclass
class Task:
    """Represents a learning task"""
    id: str
    name: str
    domain: str
    support_set: Any  # Training examples
    query_set: Any    # Test examples
    task_type: str
    difficulty: float
    metadata: Dict[str, Any]

@dataclass
class MetaLearningResult:
    """Result of meta-learning process"""
    success: bool
    adapted_model: Any
    adaptation_time: float
    final_performance: float
    learning_curve: List[float]
    confidence: float
    warnings: List[str]
    metadata: Dict[str, Any]

class MetaLearner:
    """Meta-learning system for rapid task adaptation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_model = None
        self.task_history = []
        self.performance_history = []
        self.adaptation_strategies = {}
        
        # Meta-learning algorithms
        self.algorithms = {
            MetaLearningAlgorithm.MAML: self._maml_meta_learn,
            MetaLearningAlgorithm.REPTILE: self._reptile_meta_learn,
            MetaLearningAlgorithm.PROTOTYPICAL: self._prototypical_meta_learn,
            MetaLearningAlgorithm.RELATION_NET: self._relation_network_meta_learn,
            MetaLearningAlgorithm.MATCHING_NET: self._matching_network_meta_learn,
            MetaLearningAlgorithm.GRADIENT_BASED: self._gradient_based_meta_learn
        }
        
    async def meta_train(self, tasks: List[Task], 
                        algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML,
                        epochs: int = 100) -> Dict[str, Any]:
        """Meta-train on a collection of tasks"""
        
        logger.info(f"Starting meta-training on {len(tasks)} tasks using {algorithm.value}")
        
        # Validate tasks
        validation_result = await self._validate_task_collection(tasks)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": f"Task validation failed: {validation_result['reason']}"
            }
        
        # Initialize meta-model
        await self._initialize_meta_model(tasks, algorithm)
        
        # Execute meta-learning algorithm
        meta_learn_func = self.algorithms[algorithm]
        training_result = await meta_learn_func(tasks, epochs)
        
        # Evaluate meta-learning performance
        evaluation_result = await self._evaluate_meta_learning(tasks)
        
        # Record training history
        await self._record_meta_training(tasks, algorithm, training_result, evaluation_result)
        
        return {
            "success": training_result["success"],
            "meta_model": self.meta_model,
            "training_performance": training_result,
            "evaluation_performance": evaluation_result,
            "algorithm_used": algorithm.value
        }
    
    async def rapid_adapt(self, new_task: Task, 
                         adaptation_steps: int = 5,
                         adaptation_lr: float = 0.01) -> MetaLearningResult:
        """Rapidly adapt to a new task using meta-learned knowledge"""
        
        logger.info(f"Rapid adaptation to task: {new_task.name}")
        
        if self.meta_model is None:
            return MetaLearningResult(
                success=False,
                adapted_model=None,
                adaptation_time=0.0,
                final_performance=0.0,
                learning_curve=[],
                confidence=0.0,
                warnings=["No meta-model available for adaptation"],
                metadata={}
            )
        
        start_time = datetime.now()
        
        try:
            # Clone meta-model for adaptation
            adapted_model = await self._clone_meta_model()
            
            # Perform rapid adaptation
            learning_curve = []
            for step in range(adaptation_steps):
                # Compute adaptation gradients
                gradients = await self._compute_adaptation_gradients(
                    adapted_model, new_task.support_set
                )
                
                # Update model parameters
                await self._update_model_parameters(adapted_model, gradients, adaptation_lr)
                
                # Evaluate current performance
                step_performance = await self._evaluate_on_query_set(
                    adapted_model, new_task.query_set
                )
                learning_curve.append(step_performance)
                
                logger.debug(f"Adaptation step {step}: performance = {step_performance}")
                
                # Early stopping if converged
                if step > 0 and abs(learning_curve[-1] - learning_curve[-2]) < 0.001:
                    break
            
            adaptation_time = (datetime.now() - start_time).total_seconds()
            final_performance = learning_curve[-1] if learning_curve else 0.0
            
            # Calculate adaptation confidence
            confidence = await self._calculate_adaptation_confidence(
                new_task, learning_curve, final_performance
            )
            
            # Safety and quality checks
            quality_check = await self._validate_adapted_model(adapted_model, new_task)
            warnings = []
            if not quality_check["passed"]:
                warnings.extend(quality_check["warnings"])
            
            # Record adaptation
            await self._record_adaptation(new_task, adaptation_steps, final_performance)
            
            return MetaLearningResult(
                success=True,
                adapted_model=adapted_model,
                adaptation_time=adaptation_time,
                final_performance=final_performance,
                learning_curve=learning_curve,
                confidence=confidence,
                warnings=warnings,
                metadata={
                    "adaptation_steps": len(learning_curve),
                    "adaptation_lr": adaptation_lr,
                    "task_difficulty": new_task.difficulty
                }
            )
            
        except Exception as e:
            logger.error(f"Rapid adaptation failed: {e}")
            adaptation_time = (datetime.now() - start_time).total_seconds()
            
            return MetaLearningResult(
                success=False,
                adapted_model=None,
                adaptation_time=adaptation_time,
                final_performance=0.0,
                learning_curve=[],
                confidence=0.0,
                warnings=[f"Adaptation failed: {str(e)}"],
                metadata={}
            )
    
    async def _maml_meta_learn(self, tasks: List[Task], epochs: int) -> Dict[str, Any]:
        """Model-Agnostic Meta-Learning (MAML) implementation"""
        
        meta_lr = self.config.get('meta_learning_rate', 0.001)
        inner_lr = self.config.get('inner_learning_rate', 0.01)
        inner_steps = self.config.get('inner_steps', 5)
        
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            meta_gradients = {}
            
            # Sample batch of tasks
            batch_tasks = await self._sample_task_batch(tasks)
            
            for task in batch_tasks:
                # Inner loop: adapt to task
                adapted_params = await self._inner_loop_adaptation(
                    task, inner_lr, inner_steps
                )
                
                # Outer loop: compute meta-gradient
                task_meta_gradient = await self._compute_meta_gradient(
                    task, adapted_params
                )
                
                # Accumulate meta-gradients
                meta_gradients = await self._accumulate_gradients(
                    meta_gradients, task_meta_gradient
                )
                
                # Compute task loss for monitoring
                task_loss = await self._compute_task_loss(task, adapted_params)
                epoch_loss += task_loss
            
            # Meta-update
            await self._meta_update(meta_gradients, meta_lr)
            
            avg_epoch_loss = epoch_loss / len(batch_tasks)
            training_losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"MAML Epoch {epoch}: Average loss = {avg_epoch_loss}")
        
        return {
            "success": True,
            "algorithm": "MAML",
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else float('inf')
        }
    
    async def _reptile_meta_learn(self, tasks: List[Task], epochs: int) -> Dict[str, Any]:
        """Reptile meta-learning implementation"""
        
        meta_lr = self.config.get('meta_learning_rate', 0.001)
        inner_lr = self.config.get('inner_learning_rate', 0.01)
        inner_steps = self.config.get('inner_steps', 5)
        
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Sample batch of tasks
            batch_tasks = await self._sample_task_batch(tasks)
            
            for task in batch_tasks:
                # Store initial parameters
                initial_params = await self._get_model_parameters()
                
                # Adapt to task (inner loop)
                for step in range(inner_steps):
                    gradients = await self._compute_adaptation_gradients(
                        self.meta_model, task.support_set
                    )
                    await self._update_model_parameters(
                        self.meta_model, gradients, inner_lr
                    )
                
                # Compute Reptile gradient
                adapted_params = await self._get_model_parameters()
                reptile_gradient = await self._compute_reptile_gradient(
                    initial_params, adapted_params
                )
                
                # Meta-update toward adapted parameters
                await self._reptile_meta_update(reptile_gradient, meta_lr)
                
                # Compute loss for monitoring
                task_loss = await self._evaluate_on_query_set(
                    self.meta_model, task.query_set
                )
                epoch_loss += task_loss
            
            avg_epoch_loss = epoch_loss / len(batch_tasks)
            training_losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Reptile Epoch {epoch}: Average loss = {avg_epoch_loss}")
        
        return {
            "success": True,
            "algorithm": "Reptile",
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else float('inf')
        }
    
    async def _prototypical_meta_learn(self, tasks: List[Task], epochs: int) -> Dict[str, Any]:
        """Prototypical Networks meta-learning implementation"""
        
        learning_rate = self.config.get('learning_rate', 0.001)
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Sample batch of tasks
            batch_tasks = await self._sample_task_batch(tasks)
            
            for task in batch_tasks:
                # Compute prototypes for each class
                prototypes = await self._compute_prototypes(task.support_set)
                
                # Compute distances and losses
                query_embeddings = await self._compute_embeddings(task.query_set)
                distances = await self._compute_prototype_distances(
                    query_embeddings, prototypes
                )
                
                # Compute prototypical loss
                task_loss = await self._compute_prototypical_loss(distances, task.query_set)
                epoch_loss += task_loss
                
                # Update embedding network
                gradients = await self._compute_embedding_gradients(task_loss)
                await self._update_embedding_network(gradients, learning_rate)
            
            avg_epoch_loss = epoch_loss / len(batch_tasks)
            training_losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Prototypical Epoch {epoch}: Average loss = {avg_epoch_loss}")
        
        return {
            "success": True,
            "algorithm": "Prototypical",
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else float('inf')
        }
    
    async def _relation_network_meta_learn(self, tasks: List[Task], epochs: int) -> Dict[str, Any]:
        """Relation Networks meta-learning implementation"""
        
        learning_rate = self.config.get('learning_rate', 0.001)
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            batch_tasks = await self._sample_task_batch(tasks)
            
            for task in batch_tasks:
                # Compute embeddings
                support_embeddings = await self._compute_embeddings(task.support_set)
                query_embeddings = await self._compute_embeddings(task.query_set)
                
                # Compute relations
                relations = await self._compute_relations(
                    support_embeddings, query_embeddings
                )
                
                # Compute relation scores
                relation_scores = await self._compute_relation_scores(relations)
                
                # Compute loss
                task_loss = await self._compute_relation_loss(relation_scores, task.query_set)
                epoch_loss += task_loss
                
                # Update networks
                gradients = await self._compute_relation_gradients(task_loss)
                await self._update_relation_networks(gradients, learning_rate)
            
            avg_epoch_loss = epoch_loss / len(batch_tasks)
            training_losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Relation Network Epoch {epoch}: Average loss = {avg_epoch_loss}")
        
        return {
            "success": True,
            "algorithm": "RelationNet",
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else float('inf')
        }
    
    async def _matching_network_meta_learn(self, tasks: List[Task], epochs: int) -> Dict[str, Any]:
        """Matching Networks meta-learning implementation"""
        
        learning_rate = self.config.get('learning_rate', 0.001)
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            batch_tasks = await self._sample_task_batch(tasks)
            
            for task in batch_tasks:
                # Encode support and query sets
                support_encoded = await self._encode_support_set(task.support_set)
                query_encoded = await self._encode_query_set(task.query_set)
                
                # Compute attention weights
                attention_weights = await self._compute_attention_weights(
                    support_encoded, query_encoded
                )
                
                # Compute predictions
                predictions = await self._compute_matching_predictions(
                    attention_weights, task.support_set
                )
                
                # Compute loss
                task_loss = await self._compute_matching_loss(predictions, task.query_set)
                epoch_loss += task_loss
                
                # Update networks
                gradients = await self._compute_matching_gradients(task_loss)
                await self._update_matching_networks(gradients, learning_rate)
            
            avg_epoch_loss = epoch_loss / len(batch_tasks)
            training_losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Matching Network Epoch {epoch}: Average loss = {avg_epoch_loss}")
        
        return {
            "success": True,
            "algorithm": "MatchingNet",
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else float('inf')
        }
    
    async def _gradient_based_meta_learn(self, tasks: List[Task], epochs: int) -> Dict[str, Any]:
        """Gradient-based meta-learning implementation"""
        
        meta_lr = self.config.get('meta_learning_rate', 0.001)
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            accumulated_gradients = {}
            
            batch_tasks = await self._sample_task_batch(tasks)
            
            for task in batch_tasks:
                # Compute task-specific gradients
                task_gradients = await self._compute_task_gradients(task)
                
                # Accumulate gradients
                accumulated_gradients = await self._accumulate_gradients(
                    accumulated_gradients, task_gradients
                )
                
                # Compute task loss
                task_loss = await self._evaluate_on_query_set(
                    self.meta_model, task.query_set
                )
                epoch_loss += task_loss
            
            # Meta-update
            await self._apply_accumulated_gradients(accumulated_gradients, meta_lr)
            
            avg_epoch_loss = epoch_loss / len(batch_tasks)
            training_losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Gradient-based Epoch {epoch}: Average loss = {avg_epoch_loss}")
        
        return {
            "success": True,
            "algorithm": "GradientBased",
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else float('inf')
        }
    
    # Helper methods (mock implementations for brevity)
    async def _validate_task_collection(self, tasks):
        return {"valid": True, "reason": ""}
    
    async def _initialize_meta_model(self, tasks, algorithm):
        self.meta_model = {"initialized": True, "algorithm": algorithm.value}
    
    async def _evaluate_meta_learning(self, tasks):
        return {"accuracy": 0.85, "loss": 0.15}
    
    async def _record_meta_training(self, tasks, algorithm, training_result, eval_result):
        record = {
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(tasks),
            "algorithm": algorithm.value,
            "final_loss": training_result.get("final_loss", 0),
            "evaluation_accuracy": eval_result.get("accuracy", 0)
        }
        self.task_history.append(record)
    
    async def _clone_meta_model(self):
        return {"cloned_from": self.meta_model, "timestamp": datetime.now().isoformat()}
    
    async def _compute_adaptation_gradients(self, model, support_set):
        # Mock gradient computation
        return {"gradients": "computed", "support_size": len(support_set) if hasattr(support_set, '__len__') else 1}
    
    async def _update_model_parameters(self, model, gradients, lr):
        # Mock parameter update
        model["updated"] = True
        model["learning_rate"] = lr
    
    async def _evaluate_on_query_set(self, model, query_set):
        # Mock evaluation - returns performance score
        return np.random.uniform(0.7, 0.95)
    
    async def _calculate_adaptation_confidence(self, task, learning_curve, final_performance):
        # Calculate confidence based on learning curve stability and final performance
        if len(learning_curve) < 2:
            return final_performance
        
        improvement = learning_curve[-1] - learning_curve[0]
        stability = 1.0 - np.std(learning_curve[-3:]) if len(learning_curve) >= 3 else 0.5
        
        return min(1.0, final_performance * 0.7 + improvement * 0.2 + stability * 0.1)
    
    async def _validate_adapted_model(self, model, task):
        # Mock model validation
        return {"passed": True, "warnings": []}
    
    async def _record_adaptation(self, task, steps, performance):
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "task_domain": task.domain,
            "adaptation_steps": steps,
            "final_performance": performance
        }
        self.performance_history.append(adaptation_record)
    
    # Additional mock methods for algorithm implementations
    async def _sample_task_batch(self, tasks): return tasks[:min(len(tasks), 8)]
    async def _inner_loop_adaptation(self, task, lr, steps): return {"adapted": True}
    async def _compute_meta_gradient(self, task, params): return {"meta_gradient": True}
    async def _accumulate_gradients(self, acc_grads, new_grads): return {**acc_grads, **new_grads}
    async def _compute_task_loss(self, task, params): return np.random.uniform(0.1, 0.5)
    async def _meta_update(self, gradients, lr): pass
    async def _get_model_parameters(self): return {"parameters": "current"}
    async def _compute_reptile_gradient(self, initial, adapted): return {"reptile_grad": True}
    async def _reptile_meta_update(self, gradient, lr): pass
    async def _compute_prototypes(self, support_set): return {"prototypes": "computed"}
    async def _compute_embeddings(self, data_set): return {"embeddings": "computed"}
    async def _compute_prototype_distances(self, embeddings, prototypes): return {"distances": "computed"}
    async def _compute_prototypical_loss(self, distances, query_set): return np.random.uniform(0.1, 0.5)
    async def _compute_embedding_gradients(self, loss): return {"embedding_grads": True}
    async def _update_embedding_network(self, gradients, lr): pass
    async def _compute_relations(self, support_emb, query_emb): return {"relations": "computed"}
    async def _compute_relation_scores(self, relations): return {"scores": "computed"}
    async def _compute_relation_loss(self, scores, query_set): return np.random.uniform(0.1, 0.5)
    async def _compute_relation_gradients(self, loss): return {"relation_grads": True}
    async def _update_relation_networks(self, gradients, lr): pass
    async def _encode_support_set(self, support_set): return {"support_encoded": True}
    async def _encode_query_set(self, query_set): return {"query_encoded": True}
    async def _compute_attention_weights(self, support, query): return {"attention": "computed"}
    async def _compute_matching_predictions(self, attention, support): return {"predictions": "computed"}
    async def _compute_matching_loss(self, predictions, query_set): return np.random.uniform(0.1, 0.5)
    async def _compute_matching_gradients(self, loss): return {"matching_grads": True}
    async def _update_matching_networks(self, gradients, lr): pass
    async def _compute_task_gradients(self, task): return {"task_gradients": True}
    async def _apply_accumulated_gradients(self, gradients, lr): pass

    async def get_meta_learning_history(self) -> Dict[str, Any]:
        """Get meta-learning training and adaptation history"""
        return {
            "training_history": self.task_history.copy(),
            "adaptation_history": self.performance_history.copy(),
            "meta_model_info": self.meta_model if self.meta_model else None
        }
    
    async def export_meta_model(self, filepath: str) -> bool:
        """Export trained meta-model"""
        if self.meta_model is None:
            return False
        
        try:
            # Mock export - would serialize actual model
            export_data = {
                "meta_model": self.meta_model,
                "config": self.config,
                "training_history": self.task_history,
                "export_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Meta-model exported to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Meta-model export failed: {e}")
            return False
    
    async def load_meta_model(self, filepath: str) -> bool:
        """Load pre-trained meta-model"""
        try:
            # Mock load - would deserialize actual model
            self.meta_model = {"loaded_from": filepath, "timestamp": datetime.now().isoformat()}
            logger.info(f"Meta-model loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Meta-model load failed: {e}")
            return False