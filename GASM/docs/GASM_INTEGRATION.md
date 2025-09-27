# GASM Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the Geometric Assembly State Machine (GASM) into robotic systems, external applications, and research platforms. It covers both high-level integration patterns and detailed implementation examples.

## Table of Contents

1. [Quick Start Integration](#quick-start-integration)
2. [Integration Patterns](#integration-patterns)
3. [API Integration](#api-integration)
4. [Hook System](#hook-system)
5. [Custom Extensions](#custom-extensions)
6. [Hardware Integration](#hardware-integration)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Quick Start Integration

### Basic Python Integration

```python
from gasm_bridge import create_bridge

# Initialize GASM bridge
bridge = create_bridge({
    "device": "cpu",  # or "cuda" if available
    "fallback_mode": True,
    "cache_enabled": True
})

# Process natural language instruction
instruction = "place the red block above the blue cube"
result = bridge.process(instruction)

if result["success"]:
    print(f"Generated {len(result['constraints'])} constraints")
    print(f"Target poses: {list(result['target_poses'].keys())}")
    print(f"Confidence: {result['confidence']:.2f}")
else:
    print(f"Processing failed: {result.get('error_message')}")
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from gasm_bridge import create_bridge
from pydantic import BaseModel

app = FastAPI()
gasm_bridge = create_bridge()

class SpatialInstruction(BaseModel):
    text: str
    entities: List[str] = []
    constraints: Dict[str, Any] = {}

@app.post("/process-instruction")
async def process_instruction(instruction: SpatialInstruction):
    try:
        result = gasm_bridge.process(instruction.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-constraints")
async def get_supported_constraints():
    return {
        "constraints": gasm_bridge.get_supported_constraints(),
        "sample_responses": gasm_bridge.get_sample_responses()
    }
```

### ROS Integration

```python
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from gasm_bridge import create_bridge
from utils_se3 import create_pose, pose_to_dict

class GASMROSNode:
    def __init__(self):
        rospy.init_node('gasm_spatial_reasoning')
        
        # Initialize GASM bridge
        self.bridge = create_bridge({
            "device": "cpu",
            "fallback_mode": True
        })
        
        # ROS publishers and subscribers
        self.pose_pub = rospy.Publisher('/target_poses', PoseStamped, queue_size=10)
        self.text_sub = rospy.Subscriber('/spatial_commands', String, self.text_callback)
        
        rospy.loginfo("GASM ROS node initialized")
    
    def text_callback(self, msg):
        """Process incoming text commands"""
        try:
            result = self.bridge.process(msg.data)
            
            if result["success"]:
                # Publish target poses
                for entity, pose_data in result["target_poses"].items():
                    pose_msg = self.se3_to_pose_msg(pose_data)
                    pose_msg.header.frame_id = "world"
                    pose_msg.header.stamp = rospy.Time.now()
                    self.pose_pub.publish(pose_msg)
                    
                rospy.loginfo(f"Processed instruction: {msg.data}")
            else:
                rospy.logwarn(f"Failed to process: {result.get('error_message')}")
                
        except Exception as e:
            rospy.logerr(f"Error processing command: {e}")
    
    def se3_to_pose_msg(self, pose_data):
        """Convert SE(3) pose to ROS Pose message"""
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = pose_data["position"][0]
        pose_msg.pose.position.y = pose_data["position"][1]
        pose_msg.pose.position.z = pose_data["position"][2]
        pose_msg.pose.orientation.x = pose_data["orientation"][0]
        pose_msg.pose.orientation.y = pose_data["orientation"][1]
        pose_msg.pose.orientation.z = pose_data["orientation"][2]
        pose_msg.pose.orientation.w = pose_data["orientation"][3]
        return pose_msg

if __name__ == '__main__':
    try:
        node = GASMROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## Integration Patterns

### 1. Service-Oriented Architecture (SOA)

```python
# GASM as a microservice
from flask import Flask, request, jsonify
from gasm_bridge import create_bridge

app = Flask(__name__)
bridge = create_bridge()

@app.route('/spatial/process', methods=['POST'])
def process_spatial_command():
    data = request.get_json()
    
    # Validate input
    if 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    # Process through GASM
    result = bridge.process(data['text'])
    
    # Return standardized response
    return jsonify({
        'success': result['success'],
        'constraints': result['constraints'],
        'target_poses': result['target_poses'],
        'confidence': result['confidence'],
        'processing_time': result['execution_time']
    })

@app.route('/spatial/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'supported_constraints': bridge.get_supported_constraints()
    })
```

### 2. Plugin Architecture

```python
# GASM as a plugin system
from abc import ABC, abstractmethod
from typing import Dict, Any

class GASMPlugin(ABC):
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        pass

class SpatialReasoningPlugin(GASMPlugin):
    def __init__(self):
        self.bridge = create_bridge()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = data.get('instruction', '')
        context = data.get('context', {})
        
        result = self.bridge.process(text)
        
        # Add plugin-specific processing
        result['plugin'] = 'spatial_reasoning'
        result['context'] = context
        
        return result
    
    def get_capabilities(self) -> List[str]:
        return self.bridge.get_supported_constraints()

# Plugin manager
class GASMPluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin: GASMPlugin):
        self.plugins[name] = plugin
    
    def process_with_plugin(self, plugin_name: str, data: Dict[str, Any]):
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].process(data)
        else:
            raise ValueError(f"Plugin {plugin_name} not found")
```

### 3. Event-Driven Architecture

```python
# Event-driven GASM integration
from typing import Callable, Dict, Any
import asyncio

class GASMEventSystem:
    def __init__(self):
        self.bridge = create_bridge()
        self.event_handlers = {}
        self.running = False
    
    def on(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    print(f"Error in handler: {e}")
    
    async def process_instruction_event(self, event_data):
        """Handle spatial instruction events"""
        result = self.bridge.process(event_data['text'])
        
        # Emit result events
        if result['success']:
            await self.emit('constraints_generated', {
                'constraints': result['constraints'],
                'confidence': result['confidence']
            })
            await self.emit('poses_generated', {
                'poses': result['target_poses'],
                'timestamp': time.time()
            })
        else:
            await self.emit('processing_failed', {
                'error': result.get('error_message'),
                'original_text': event_data['text']
            })

# Usage example
event_system = GASMEventSystem()

@event_system.on('spatial_instruction')
async def handle_instruction(data):
    await event_system.process_instruction_event(data)

@event_system.on('constraints_generated')
async def handle_constraints(data):
    print(f"Generated {len(data['constraints'])} constraints")
    # Forward to motion planner
    
@event_system.on('poses_generated')
async def handle_poses(data):
    print(f"Generated {len(data['poses'])} target poses")
    # Forward to robot controller
```

## API Integration

### REST API Endpoints

```yaml
# API specification (OpenAPI 3.0)
openapi: 3.0.0
info:
  title: GASM Spatial Reasoning API
  version: 1.0.0
  description: API for geometric assembly and spatial reasoning

paths:
  /api/v1/process:
    post:
      summary: Process spatial instruction
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                text:
                  type: string
                  description: Natural language spatial instruction
                entities:
                  type: array
                  items:
                    type: string
                  description: Known entities in the scene
                context:
                  type: object
                  description: Additional context information
              required:
                - text
      responses:
        '200':
          description: Successfully processed instruction
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/GASMResponse'
        '400':
          description: Invalid input
        '500':
          description: Processing error

  /api/v1/constraints:
    get:
      summary: Get supported constraint types
      responses:
        '200':
          description: List of supported constraints
          content:
            application/json:
              schema:
                type: object
                properties:
                  constraints:
                    type: array
                    items:
                      type: string

  /api/v1/validate-pose:
    post:
      summary: Validate SE(3) pose format
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SE3Pose'
      responses:
        '200':
          description: Pose validation result
          content:
            application/json:
              schema:
                type: object
                properties:
                  valid:
                    type: boolean
                  errors:
                    type: array
                    items:
                      type: string

components:
  schemas:
    GASMResponse:
      type: object
      properties:
        success:
          type: boolean
        constraints:
          type: array
          items:
            $ref: '#/components/schemas/SpatialConstraint'
        target_poses:
          type: object
          additionalProperties:
            $ref: '#/components/schemas/SE3Pose'
        confidence:
          type: number
          minimum: 0
          maximum: 1
        execution_time:
          type: number
        error_message:
          type: string
          nullable: true
        debug_info:
          type: object
          nullable: true

    SE3Pose:
      type: object
      properties:
        position:
          type: array
          items:
            type: number
          minItems: 3
          maxItems: 3
        orientation:
          type: array
          items:
            type: number
          minItems: 4
          maxItems: 4
        frame_id:
          type: string
          default: "world"
        confidence:
          type: number
          minimum: 0
          maximum: 1
          default: 1.0

    SpatialConstraint:
      type: object
      properties:
        type:
          type: string
          enum: [above, below, left, right, near, far, angle, distance]
        subject:
          type: string
        target:
          type: string
          nullable: true
        parameters:
          type: object
        priority:
          type: number
          minimum: 0
          maximum: 1
        tolerance:
          type: object
```

### WebSocket Integration

```python
# WebSocket server for real-time GASM processing
import asyncio
import websockets
import json
from gasm_bridge import create_bridge

class GASMWebSocketServer:
    def __init__(self):
        self.bridge = create_bridge()
        self.clients = set()
    
    async def register(self, websocket):
        """Register new client"""
        self.clients.add(websocket)
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': 'Connected to GASM WebSocket server',
            'supported_constraints': self.bridge.get_supported_constraints()
        }))
    
    async def unregister(self, websocket):
        """Unregister client"""
        self.clients.discard(websocket)
    
    async def process_message(self, websocket, message):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'spatial_instruction':
                # Process spatial instruction
                result = self.bridge.process(data.get('text', ''))
                
                # Send result back to client
                await websocket.send(json.dumps({
                    'type': 'gasm_response',
                    'request_id': data.get('request_id'),
                    'result': result
                }))
                
            elif data.get('type') == 'validate_pose':
                # Validate pose format
                pose = data.get('pose', {})
                is_valid = self.bridge.validate_pose(pose)
                
                await websocket.send(json.dumps({
                    'type': 'validation_result',
                    'request_id': data.get('request_id'),
                    'valid': is_valid
                }))
                
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handler(self, websocket, path):
        """Main WebSocket handler"""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

# Start WebSocket server
if __name__ == "__main__":
    server = GASMWebSocketServer()
    start_server = websockets.serve(server.handler, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
```

## Hook System

The GASM system provides a comprehensive hook system for customization and extension.

### Hook Registration

```python
from gasm_hooks import HookManager

# Initialize hook manager
hooks = HookManager()

# Register hooks
@hooks.register('preprocessing.text')
def preprocess_text(text: str, context: Dict[str, Any]) -> str:
    """Clean and normalize input text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase for processing
    text = text.lower()
    
    # Add context-specific processing
    if context.get('domain') == 'manufacturing':
        # Manufacturing-specific text processing
        text = text.replace('workpiece', 'object')
        text = text.replace('fixture', 'mount')
    
    return text

@hooks.register('postprocessing.constraints')
def postprocess_constraints(constraints: List[Dict], context: Dict[str, Any]) -> List[Dict]:
    """Post-process generated constraints"""
    # Add domain-specific constraint modifications
    for constraint in constraints:
        if constraint['type'] == 'above':
            # Add safety margin for manufacturing
            if 'safety_margin' not in constraint['parameters']:
                constraint['parameters']['safety_margin'] = 0.05  # 5cm
    
    return constraints

@hooks.register('validation.pose')
def validate_pose_custom(pose: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Custom pose validation logic"""
    # Check workspace bounds
    position = pose.get('position', [0, 0, 0])
    workspace_bounds = context.get('workspace_bounds', {
        'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]
    })
    
    # Validate position is within workspace
    if not (workspace_bounds['x'][0] <= position[0] <= workspace_bounds['x'][1]):
        return False
    if not (workspace_bounds['y'][0] <= position[1] <= workspace_bounds['y'][1]):
        return False
    if not (workspace_bounds['z'][0] <= position[2] <= workspace_bounds['z'][1]):
        return False
    
    return True
```

### Available Hook Points

#### Text Processing Hooks
```python
# Pre-processing hooks
hooks.register('preprocessing.text', text_cleaner_func)
hooks.register('preprocessing.entity_recognition', entity_extractor_func)
hooks.register('preprocessing.context_analysis', context_analyzer_func)

# Post-processing hooks
hooks.register('postprocessing.text', text_postprocessor_func)
hooks.register('postprocessing.entities', entity_postprocessor_func)
```

#### Constraint Processing Hooks
```python
# Constraint generation hooks
hooks.register('constraints.generation', constraint_generator_func)
hooks.register('constraints.validation', constraint_validator_func)
hooks.register('constraints.optimization', constraint_optimizer_func)

# Constraint post-processing hooks
hooks.register('postprocessing.constraints', constraint_postprocessor_func)
hooks.register('postprocessing.priorities', priority_adjuster_func)
```

#### Planning Hooks
```python
# Planning hooks
hooks.register('planning.preprocessing', planning_preprocessor_func)
hooks.register('planning.strategy_selection', strategy_selector_func)
hooks.register('planning.postprocessing', planning_postprocessor_func)

# Execution hooks
hooks.register('execution.preprocessing', execution_preprocessor_func)
hooks.register('execution.monitoring', execution_monitor_func)
hooks.register('execution.postprocessing', execution_postprocessor_func)
```

#### Learning Hooks
```python
# Learning hooks
hooks.register('learning.data_collection', data_collector_func)
hooks.register('learning.model_update', model_updater_func)
hooks.register('learning.evaluation', evaluation_func)
```

### Hook Context

```python
class HookContext:
    """Context object passed to hook functions"""
    
    def __init__(self):
        self.data = {}
        self.metadata = {}
        self.timestamp = time.time()
        self.request_id = str(uuid.uuid4())
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)

# Usage in hooks
@hooks.register('preprocessing.text')
def my_preprocessor(text: str, context: HookContext) -> str:
    # Access context data
    domain = context.get('domain', 'general')
    user_id = context.get_metadata('user_id')
    
    # Set results for downstream hooks
    context.set('processed_text', text.lower())
    context.set_metadata('preprocessing_time', time.time() - context.timestamp)
    
    return text.lower()
```

## Custom Extensions

### Custom Constraint Types

```python
from gasm_bridge import ConstraintType
from typing import Dict, Any
import numpy as np

class CustomConstraintType(ConstraintType):
    MAGNETIC_ATTRACTION = "magnetic_attraction"
    THERMAL_SAFETY = "thermal_safety"
    TOOL_ORIENTATION = "tool_orientation"

class MagneticAttractionConstraint:
    """Custom constraint for magnetic attraction between objects"""
    
    def __init__(self, subject: str, target: str, strength: float = 1.0):
        self.subject = subject
        self.target = target
        self.strength = strength
        self.type = CustomConstraintType.MAGNETIC_ATTRACTION
    
    def evaluate(self, positions: Dict[str, np.ndarray]) -> float:
        """Evaluate magnetic attraction energy"""
        if self.subject not in positions or self.target not in positions:
            return 0.0
        
        subject_pos = positions[self.subject]
        target_pos = positions[self.target]
        
        # Magnetic force follows inverse square law
        distance = np.linalg.norm(target_pos - subject_pos)
        energy = -self.strength / (distance ** 2 + 1e-6)  # Avoid division by zero
        
        return energy
    
    def gradient(self, positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradient for optimization"""
        if self.subject not in positions or self.target not in positions:
            return {}
        
        subject_pos = positions[self.subject]
        target_pos = positions[self.target]
        
        diff = target_pos - subject_pos
        distance = np.linalg.norm(diff)
        
        # Gradient of magnetic attraction
        grad_magnitude = 2 * self.strength / (distance ** 3 + 1e-6)
        grad_direction = diff / (distance + 1e-6)
        
        return {
            self.subject: grad_magnitude * grad_direction,
            self.target: -grad_magnitude * grad_direction
        }

# Register custom constraint with GASM
from gasm_core import ConstraintRegistry

ConstraintRegistry.register(
    CustomConstraintType.MAGNETIC_ATTRACTION,
    MagneticAttractionConstraint
)
```

### Custom Planning Strategies

```python
from planner import PlanningStrategy, MotionPlanner, PlanningResult
import numpy as np

class LearningBasedPlanner(MotionPlanner):
    """Custom planner using reinforcement learning"""
    
    def __init__(self, config=None, model_path=None):
        super().__init__(config)
        self.model_path = model_path
        self.policy_network = self._load_policy_network()
    
    def _load_policy_network(self):
        """Load pre-trained policy network"""
        if self.model_path:
            # Load your RL model here
            # return torch.load(self.model_path)
            pass
        return None
    
    def _plan_learning_based(self, current, target, constraints):
        """Learning-based planning strategy"""
        if self.policy_network is None:
            # Fall back to constrained planning
            return self._plan_constrained(current, target, constraints)
        
        # Prepare state representation
        state = self._encode_state(current, target, constraints)
        
        # Get action from policy network
        with torch.no_grad():
            action = self.policy_network(state)
        
        # Convert action to pose step
        next_pose = self._action_to_pose(current, action)
        
        # Validate and return result
        return PlanningResult(
            success=True,
            next_pose=next_pose,
            step_size=current.distance_to(next_pose),
            reasoning="Learning-based planning step"
        )
    
    def _encode_state(self, current, target, constraints):
        """Encode current state for neural network"""
        # Implement state encoding logic
        state_vector = np.concatenate([
            current.to_array(),
            target.to_array(),
            self._encode_constraints(constraints)
        ])
        return torch.tensor(state_vector, dtype=torch.float32)
    
    def _encode_constraints(self, constraints):
        """Encode constraints into feature vector"""
        # Implement constraint encoding
        return np.zeros(32)  # Placeholder
    
    def _action_to_pose(self, current, action):
        """Convert network action to pose"""
        # Implement action to pose conversion
        step = action.numpy()
        next_array = current.to_array() + step
        return Pose.from_array(next_array)

# Register custom planning strategy
PlanningStrategy.LEARNING_BASED = "learning_based"
```

### Custom Neural Network Modules

```python
import torch
import torch.nn as nn
from gasm_core import SE3InvariantAttention

class CustomSE3Layer(nn.Module):
    """Custom SE(3)-equivariant layer for domain-specific processing"""
    
    def __init__(self, feature_dim: int, domain: str = "manufacturing"):
        super().__init__()
        self.feature_dim = feature_dim
        self.domain = domain
        
        # Domain-specific processing layers
        if domain == "manufacturing":
            self.domain_layer = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        elif domain == "surgery":
            self.domain_layer = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh(),  # Smoother activations for precision tasks
                nn.Dropout(0.05),  # Lower dropout for stability
                nn.LayerNorm(feature_dim)
            )
        else:
            self.domain_layer = nn.Identity()
        
        # SE(3) equivariant processing
        self.se3_attention = SE3InvariantAttention(
            feature_dim=feature_dim,
            hidden_dim=feature_dim,
            num_heads=8
        )
    
    def forward(self, features, positions, edge_index=None):
        """Forward pass with domain-specific processing"""
        # Apply domain-specific transformations
        domain_features = self.domain_layer(features)
        
        # Apply SE(3) equivariant attention
        if edge_index is not None:
            se3_features = self.se3_attention(
                domain_features, positions, edge_index
            )
        else:
            se3_features = domain_features
        
        # Residual connection
        output = features + se3_features
        
        return output

# Integration with GASM core
class CustomGASM(nn.Module):
    """Custom GASM with domain-specific layers"""
    
    def __init__(self, feature_dim=256, domain="manufacturing"):
        super().__init__()
        
        # Base GASM layers
        self.embedding = nn.Linear(6, feature_dim)  # SE(3) pose embedding
        self.custom_layer = CustomSE3Layer(feature_dim, domain)
        self.output_layer = nn.Linear(feature_dim, 6)  # Output SE(3) pose
        
        # Domain-specific parameters
        self.domain_params = nn.ParameterDict({
            'manufacturing': nn.Parameter(torch.ones(feature_dim) * 0.1),
            'surgery': nn.Parameter(torch.ones(feature_dim) * 0.01),
            'assembly': nn.Parameter(torch.ones(feature_dim) * 0.05)
        })
        self.domain = domain
    
    def forward(self, poses, constraints=None, edge_index=None):
        """Forward pass through custom GASM"""
        # Embed poses
        features = self.embedding(poses)
        
        # Apply domain-specific scaling
        if self.domain in self.domain_params:
            features = features * self.domain_params[self.domain]
        
        # Process through custom layer
        processed_features = self.custom_layer(
            features, poses[:, :3], edge_index  # Use position part of poses
        )
        
        # Generate output poses
        output_poses = self.output_layer(processed_features)
        
        return output_poses
```

## Hardware Integration

### Robot Control Integration

```python
from abc import ABC, abstractmethod
import numpy as np

class RobotController(ABC):
    """Abstract base class for robot controllers"""
    
    @abstractmethod
    def move_to_pose(self, pose: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def get_current_pose(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def emergency_stop(self) -> None:
        pass

class GASMRobotIntegration:
    """Integration layer between GASM and robot controllers"""
    
    def __init__(self, bridge, controller: RobotController):
        self.bridge = bridge
        self.controller = controller
        self.safety_checker = SafetyChecker()
    
    async def execute_instruction(self, text: str) -> bool:
        """Execute natural language instruction on robot"""
        try:
            # Process instruction through GASM
            result = self.bridge.process(text)
            
            if not result['success']:
                logger.error(f"GASM processing failed: {result.get('error_message')}")
                return False
            
            # Extract target poses
            target_poses = result['target_poses']
            
            # Execute each pose
            for entity, pose_data in target_poses.items():
                # Safety check
                if not self.safety_checker.validate_pose(pose_data):
                    logger.error(f"Unsafe pose detected for {entity}")
                    self.controller.emergency_stop()
                    return False
                
                # Execute motion
                success = await self.controller.move_to_pose(pose_data)
                if not success:
                    logger.error(f"Failed to move to pose for {entity}")
                    return False
                
                # Verify final pose
                current_pose = await self.controller.get_current_pose()
                if not self._verify_pose_reached(pose_data, current_pose):
                    logger.warning(f"Pose verification failed for {entity}")
            
            return True
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.controller.emergency_stop()
            return False
    
    def _verify_pose_reached(self, target, current, tolerance=0.01):
        """Verify that target pose was reached within tolerance"""
        target_pos = np.array(target['position'])
        current_pos = np.array(current['position'])
        
        position_error = np.linalg.norm(current_pos - target_pos)
        return position_error < tolerance

class SafetyChecker:
    """Safety validation for robot poses and motions"""
    
    def __init__(self):
        # Define workspace bounds
        self.workspace_bounds = {
            'x': [-0.8, 0.8],
            'y': [-0.8, 0.8],
            'z': [0.0, 1.2]
        }
        
        # Define collision objects
        self.collision_objects = []
    
    def validate_pose(self, pose_data: Dict[str, Any]) -> bool:
        """Validate pose for safety"""
        position = pose_data.get('position', [0, 0, 0])
        
        # Check workspace bounds
        if not self._check_workspace_bounds(position):
            return False
        
        # Check collisions
        if not self._check_collisions(position):
            return False
        
        # Check singular configurations
        if not self._check_singularities(pose_data):
            return False
        
        return True
    
    def _check_workspace_bounds(self, position):
        """Check if position is within workspace bounds"""
        x, y, z = position
        bounds = self.workspace_bounds
        
        return (bounds['x'][0] <= x <= bounds['x'][1] and
                bounds['y'][0] <= y <= bounds['y'][1] and
                bounds['z'][0] <= z <= bounds['z'][1])
    
    def _check_collisions(self, position):
        """Check for collisions with known objects"""
        # Implement collision checking logic
        return True  # Placeholder
    
    def _check_singularities(self, pose_data):
        """Check for kinematic singularities"""
        # Implement singularity checking logic
        return True  # Placeholder
```

### Universal Robots (UR) Integration Example

```python
import socket
import struct
from typing import List, Dict, Any

class URRobotController(RobotController):
    """Controller for Universal Robots UR series"""
    
    def __init__(self, robot_ip: str = "192.168.1.100", port: int = 30002):
        self.robot_ip = robot_ip
        self.port = port
        self.socket = None
        self.connect()
    
    def connect(self):
        """Connect to UR robot"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.robot_ip, self.port))
            logger.info(f"Connected to UR robot at {self.robot_ip}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
    
    async def move_to_pose(self, pose: Dict[str, Any]) -> bool:
        """Move robot to specified pose"""
        try:
            # Convert GASM pose to UR format
            ur_pose = self._gasm_to_ur_pose(pose)
            
            # Generate UR script command
            command = f"movel(p{ur_pose}, a=0.1, v=0.05)\n"
            
            # Send command to robot
            self.socket.send(command.encode('utf-8'))
            
            # Wait for completion (simplified)
            await asyncio.sleep(2.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to move robot: {e}")
            return False
    
    async def get_current_pose(self) -> Dict[str, Any]:
        """Get current robot pose"""
        try:
            # Send request for current pose
            self.socket.send(b"get_actual_tcp_pose()\n")
            
            # Receive response (simplified)
            response = self.socket.recv(1024).decode('utf-8')
            
            # Parse UR pose format
            ur_pose = self._parse_ur_pose(response)
            
            # Convert to GASM format
            gasm_pose = self._ur_to_gasm_pose(ur_pose)
            
            return gasm_pose
            
        except Exception as e:
            logger.error(f"Failed to get robot pose: {e}")
            return {}
    
    def emergency_stop(self):
        """Emergency stop the robot"""
        try:
            self.socket.send(b"stop()\n")
            logger.warning("Emergency stop activated")
        except Exception as e:
            logger.error(f"Failed to send emergency stop: {e}")
    
    def _gasm_to_ur_pose(self, pose: Dict[str, Any]) -> List[float]:
        """Convert GASM pose to UR pose format [x, y, z, rx, ry, rz]"""
        position = pose.get('position', [0, 0, 0])
        
        # Convert quaternion to rotation vector (axis-angle)
        if 'quaternion' in pose:
            quaternion = pose['quaternion']
            rotation_vector = self._quaternion_to_rotvec(quaternion)
        else:
            rotation_vector = [0, 0, 0]
        
        return position + rotation_vector
    
    def _ur_to_gasm_pose(self, ur_pose: List[float]) -> Dict[str, Any]:
        """Convert UR pose to GASM pose format"""
        position = ur_pose[:3]
        rotation_vector = ur_pose[3:6]
        
        # Convert rotation vector to quaternion
        quaternion = self._rotvec_to_quaternion(rotation_vector)
        
        return {
            'position': position,
            'quaternion': quaternion,
            'frame_id': 'base',
            'confidence': 1.0
        }
    
    def _quaternion_to_rotvec(self, q: List[float]) -> List[float]:
        """Convert quaternion to rotation vector"""
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat(q)
        return r.as_rotvec().tolist()
    
    def _rotvec_to_quaternion(self, rotvec: List[float]) -> List[float]:
        """Convert rotation vector to quaternion"""
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(rotvec)
        return r.as_quat().tolist()

# Usage example
async def main():
    # Initialize GASM bridge
    bridge = create_bridge()
    
    # Initialize robot controller
    ur_controller = URRobotController("192.168.1.100")
    
    # Create integration
    robot_integration = GASMRobotIntegration(bridge, ur_controller)
    
    # Execute instruction
    success = await robot_integration.execute_instruction(
        "move the tool above the workpiece"
    )
    
    if success:
        print("Instruction executed successfully")
    else:
        print("Execution failed")
```

## Troubleshooting

### Common Integration Issues

#### 1. Import Errors
```python
# Problem: Cannot import GASM modules
try:
    from gasm_bridge import create_bridge
except ImportError as e:
    print(f"GASM import failed: {e}")
    print("Solutions:")
    print("1. Check PYTHONPATH includes GASM directory")
    print("2. Install missing dependencies: pip install -r requirements.txt")
    print("3. Verify GASM package installation")
```

#### 2. Device Compatibility
```python
# Problem: CUDA/GPU issues
import torch

def check_device_compatibility():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Always provide CPU fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    return device

# Use in GASM initialization
device = check_device_compatibility()
bridge = create_bridge({"device": device})
```

#### 3. Memory Issues
```python
# Problem: Out of memory errors
import gc
import torch

def memory_efficient_processing(bridge, instructions):
    """Process instructions with memory management"""
    results = []
    
    for i, instruction in enumerate(instructions):
        try:
            # Process instruction
            result = bridge.process(instruction)
            results.append(result)
            
            # Clear memory periodically
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Memory error on instruction {i}, clearing cache...")
                gc.collect()
                torch.cuda.empty_cache()
                
                # Retry with smaller batch size
                result = bridge.process(instruction, batch_size=1)
                results.append(result)
            else:
                raise e
    
    return results
```

#### 4. Network Connectivity Issues
```python
# Problem: API endpoint not responding
import requests
import time
from urllib.parse import urljoin

class GASMAPIClient:
    def __init__(self, base_url, timeout=30, max_retries=3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def process_instruction(self, text, retry_count=0):
        """Process instruction with retry logic"""
        url = urljoin(self.base_url, '/api/v1/process')
        
        try:
            response = self.session.post(
                url,
                json={'text': text},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Request failed, retrying in {wait_time} seconds... ({retry_count+1}/{self.max_retries})")
                time.sleep(wait_time)
                return self.process_instruction(text, retry_count + 1)
            else:
                print(f"Max retries exceeded. Error: {e}")
                return None
    
    def health_check(self):
        """Check API health"""
        try:
            url = urljoin(self.base_url, '/api/v1/health')
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

# Usage
client = GASMAPIClient('http://localhost:8000')
if not client.health_check():
    print("Warning: GASM API server not responding")
```

## Best Practices

### 1. Error Handling

```python
# Comprehensive error handling
class GASMIntegrationError(Exception):
    pass

class GASMProcessingError(GASMIntegrationError):
    pass

class GASMValidationError(GASMIntegrationError):
    pass

def safe_gasm_processing(bridge, instruction):
    """Safe GASM processing with comprehensive error handling"""
    try:
        # Validate input
        if not instruction or not isinstance(instruction, str):
            raise GASMValidationError("Invalid instruction format")
        
        # Process instruction
        result = bridge.process(instruction)
        
        # Validate result
        if not isinstance(result, dict):
            raise GASMProcessingError("Invalid result format")
        
        if not result.get('success', False):
            error_msg = result.get('error_message', 'Unknown error')
            raise GASMProcessingError(f"Processing failed: {error_msg}")
        
        return result
        
    except GASMIntegrationError:
        # Re-raise GASM-specific errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise GASMProcessingError(f"Unexpected error: {e}") from e
```

### 2. Configuration Management

```python
# Centralized configuration
import yaml
from pathlib import Path

class GASMConfig:
    def __init__(self, config_path=None):
        self.config_path = config_path or Path("gasm_config.yaml")
        self.config = self._load_default_config()
        
        if self.config_path.exists():
            self.load_from_file(self.config_path)
    
    def _load_default_config(self):
        return {
            'system': {
                'device': 'auto',  # 'auto', 'cpu', 'cuda'
                'precision': 'float32',
                'cache_enabled': True,
                'log_level': 'INFO'
            },
            'gasm': {
                'fallback_mode': True,
                'timeout_seconds': 30,
                'confidence_threshold': 0.7
            },
            'planning': {
                'strategy': 'adaptive',
                'max_iterations': 1000,
                'tolerance': 0.01,
                'safety_margin': 0.05
            },
            'robot': {
                'workspace_bounds': {
                    'x': [-1.0, 1.0],
                    'y': [-1.0, 1.0],
                    'z': [0.0, 2.0]
                },
                'max_velocity': 0.1,
                'max_acceleration': 0.05
            }
        }
    
    def load_from_file(self, path):
        with open(path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Deep merge configurations
        self._deep_update(self.config, file_config)
    
    def save_to_file(self, path=None):
        path = path or self.config_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key_path, default=None):
        """Get config value using dot notation (e.g., 'system.device')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def _deep_update(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

# Usage
config = GASMConfig()
device = config.get('system.device', 'cpu')
bridge = create_bridge({'device': device})
```

### 3. Logging and Monitoring

```python
import logging
import time
import json
from pathlib import Path

class GASMLogger:
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger('gasm_integration')
        self.logger.setLevel(log_level)
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"gasm_{int(time.time())}.log"
        )
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics collection
        self.metrics = []
    
    def log_instruction_processing(self, instruction, result, duration):
        """Log instruction processing with metrics"""
        success = result.get('success', False)
        confidence = result.get('confidence', 0.0)
        constraints_count = len(result.get('constraints', []))
        
        self.logger.info(f"Processed instruction: '{instruction}' - "
                        f"Success: {success}, Confidence: {confidence:.2f}, "
                        f"Constraints: {constraints_count}, Duration: {duration:.3f}s")
        
        # Collect metrics
        self.metrics.append({
            'timestamp': time.time(),
            'instruction': instruction,
            'success': success,
            'confidence': confidence,
            'constraints_count': constraints_count,
            'duration': duration
        })
    
    def log_error(self, error, context=None):
        """Log error with context"""
        error_msg = f"Error: {error}"
        if context:
            error_msg += f" - Context: {json.dumps(context)}"
        
        self.logger.error(error_msg)
    
    def get_metrics_summary(self):
        """Get summary of processing metrics"""
        if not self.metrics:
            return {}
        
        total_instructions = len(self.metrics)
        successful_instructions = sum(1 for m in self.metrics if m['success'])
        avg_confidence = sum(m['confidence'] for m in self.metrics) / total_instructions
        avg_duration = sum(m['duration'] for m in self.metrics) / total_instructions
        
        return {
            'total_instructions': total_instructions,
            'success_rate': successful_instructions / total_instructions,
            'average_confidence': avg_confidence,
            'average_duration': avg_duration,
            'total_duration': sum(m['duration'] for m in self.metrics)
        }
    
    def save_metrics(self, filename=None):
        """Save metrics to file"""
        filename = filename or f"gasm_metrics_{int(time.time())}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_metrics_summary(),
                'detailed_metrics': self.metrics
            }, f, indent=2)

# Usage
logger = GASMLogger()

def monitored_processing(bridge, instruction):
    start_time = time.time()
    try:
        result = bridge.process(instruction)
        duration = time.time() - start_time
        logger.log_instruction_processing(instruction, result, duration)
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.log_error(e, {'instruction': instruction, 'duration': duration})
        raise
```

### 4. Testing and Validation

```python
import unittest
import tempfile
from unittest.mock import Mock, patch

class TestGASMIntegration(unittest.TestCase):
    def setUp(self):
        """Setup test fixtures"""
        self.bridge = create_bridge({
            'device': 'cpu',
            'fallback_mode': True
        })
    
    def test_basic_processing(self):
        """Test basic instruction processing"""
        instruction = "place object above surface"
        result = self.bridge.process(instruction)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('constraints', result)
        self.assertIn('confidence', result)
    
    def test_constraint_generation(self):
        """Test constraint generation"""
        instruction = "box above table"
        result = self.bridge.process(instruction)
        
        if result['success']:
            constraints = result['constraints']
            self.assertIsInstance(constraints, list)
            self.assertTrue(len(constraints) > 0)
            
            # Check constraint structure
            for constraint in constraints:
                self.assertIn('type', constraint)
                self.assertIn('subject', constraint)
    
    def test_pose_validation(self):
        """Test pose validation"""
        valid_pose = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'frame_id': 'world',
            'confidence': 1.0
        }
        
        invalid_pose = {
            'position': [0.0, 0.0],  # Invalid: only 2 coordinates
            'orientation': [1.0, 0.0, 0.0, 0.0]
        }
        
        self.assertTrue(self.bridge.validate_pose(valid_pose))
        self.assertFalse(self.bridge.validate_pose(invalid_pose))
    
    def test_error_handling(self):
        """Test error handling"""
        # Test empty instruction
        result = self.bridge.process("")
        self.assertFalse(result.get('success', True))
        
        # Test invalid instruction
        result = self.bridge.process("sdfsdf invalid text 12345")
        # Should not crash, might succeed with fallback
        self.assertIsInstance(result, dict)
    
    def test_supported_constraints(self):
        """Test supported constraints query"""
        constraints = self.bridge.get_supported_constraints()
        self.assertIsInstance(constraints, list)
        self.assertTrue(len(constraints) > 0)
        
        # Check for expected constraint types
        expected_constraints = ['above', 'below', 'near', 'distance']
        for constraint in expected_constraints:
            self.assertIn(constraint, constraints)
    
    @patch('gasm_bridge.GASMBridge.process')
    def test_timeout_handling(self, mock_process):
        """Test timeout handling"""
        # Simulate slow processing
        mock_process.side_effect = lambda x: time.sleep(5) or {'success': True}
        
        # This should timeout and handle gracefully
        # (Implementation depends on actual timeout mechanism)
        result = self.bridge.process("test instruction")
        self.assertIsInstance(result, dict)

class TestGASMAPIIntegration(unittest.TestCase):
    def setUp(self):
        """Setup API test fixtures"""
        from fastapi.testclient import TestClient
        from your_api_module import app
        
        self.client = TestClient(app)
    
    def test_process_endpoint(self):
        """Test /process-instruction endpoint"""
        response = self.client.post(
            "/process-instruction",
            json={"text": "place object above surface"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('success', data)
        self.assertIn('constraints', data)
    
    def test_constraints_endpoint(self):
        """Test /supported-constraints endpoint"""
        response = self.client.get("/supported-constraints")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('constraints', data)
        self.assertIsInstance(data['constraints'], list)

if __name__ == '__main__':
    # Run tests
    unittest.main()
```

This comprehensive integration guide provides the foundation for successfully integrating GASM into various systems and applications. Follow the patterns and examples most relevant to your specific use case, and refer to the troubleshooting section for common issues.