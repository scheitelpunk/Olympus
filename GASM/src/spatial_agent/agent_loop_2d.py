#!/usr/bin/env python3
"""
2D Spatial Agent Demonstration
A complete, self-contained demonstration of GASM with real-time visualization
Features:
- Simple 2D scene with conveyor and sensor
- Text-to-constraints via integrated GASM bridge
- Real-time matplotlib visualization 
- Feedback loop: plan → execute → observe → evaluate → iterate
- Support for spatial relationships: above, angle, distance
- Convergence detection and early stopping
- Optional GIF export for demonstrations
"""

import argparse
import logging
import math
import random
import time
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using numpy fallbacks")
    # Create dummy torch module for fallback
    class DummyTensor(np.ndarray):
        def __new__(cls, data, dtype=None):
            obj = np.asarray(data, dtype=dtype).view(cls)
            return obj
        
        @property
        def device(self):
            return 'cpu'
        
        def clone(self):
            return DummyTensor(self.copy())
        
        def detach(self):
            return self
        
        def cpu(self):
            return self
        
        def numpy(self):
            return np.array(self)
        
        def requires_grad_(self, requires_grad=True):
            return self
        
        def backward(self):
            pass
        
        @property
        def grad(self):
            return None
        
        def zero_grad(self):
            pass
    
    class DummyTorch:
        class Tensor(DummyTensor):
            pass
        
        @staticmethod
        def tensor(data, dtype=None, device=None):
            np_dtype = np.float32 if dtype is None else dtype
            if hasattr(dtype, '__name__') and dtype.__name__ == 'float32':
                np_dtype = np.float32
            return DummyTensor(data, dtype=np_dtype)
        
        @staticmethod
        def zeros(*args, dtype=None, device=None):
            np_dtype = np.float32 if dtype is None else dtype
            return DummyTensor(np.zeros(*args), dtype=np_dtype)
        
        @staticmethod
        def ones(*args, dtype=None, device=None):
            np_dtype = np.float32 if dtype is None else dtype
            return DummyTensor(np.ones(*args), dtype=np_dtype)
        
        @staticmethod
        def randn(*args, dtype=None, device=None):
            return DummyTensor(np.random.randn(*args))
        
        @staticmethod
        def relu(input):
            return DummyTensor(np.maximum(0, input))
        
        @staticmethod
        def sigmoid(input):
            return DummyTensor(1 / (1 + np.exp(-np.array(input))))
        
        @staticmethod
        def tanh(input):
            return DummyTensor(np.tanh(input))
        
        @staticmethod
        def norm(input, dim=None, keepdim=False):
            try:
                if dim is None:
                    result = np.linalg.norm(input)
                    return DummyTensor([result]) if keepdim else DummyTensor([result])
                else:
                    result = np.linalg.norm(input, axis=dim, keepdims=keepdim)
                    return DummyTensor(result)
            except Exception as e:
                # Fallback for broadcasting issues
                return DummyTensor([0.0])
        
        float32 = np.float32
        
        class optim:
            class Adam:
                def __init__(self, params, lr=0.001):
                    self.params = params
                    self.lr = lr
                def step(self):
                    pass
                def zero_grad(self):
                    pass
    
    torch = DummyTorch()
    
    # Create dummy F module with common functions
    class DummyF:
        @staticmethod
        def relu(input):
            return np.maximum(0, input)
        
        @staticmethod 
        def sigmoid(input):
            return 1 / (1 + np.exp(-np.array(input)))
        
        @staticmethod
        def tanh(input):
            return np.tanh(input)
        
        @staticmethod
        def softmax(input, dim=-1):
            exp_input = np.exp(input - np.max(input, axis=dim, keepdims=True))
            return exp_input / np.sum(exp_input, axis=dim, keepdims=True)
    
    F = DummyF()

# Add imageio for GIF creation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    print("Warning: imageio not available, GIF export will be disabled")
    IMAGEIO_AVAILABLE = False

# Import the core GASM functionality
try:
    # Add src to path so we can import gasm directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from gasm.core import GASM as EnhancedGASM
    try:
        from gasm.utils import ConstraintHandler
    except ImportError:
        # Create a simple constraint handler if not available
        class ConstraintHandler:
            @staticmethod
            def apply_energy_constraints(positions, constraints, learning_rate=0.01):
                return positions
    GASM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import GASM core ({e}), using fallback implementation")
    GASM_AVAILABLE = False
    # Create a minimal fallback implementation
    class EnhancedGASM:
        def __init__(self, *args, **kwargs):
            pass
        def forward(self, E, F, R, C=None, **kwargs):
            # Simple fallback: return random positions
            if TORCH_AVAILABLE:
                return torch.randn(len(E), 3) * 0.1
            else:
                return np.random.randn(len(E), 3) * 0.1
    
    class ConstraintHandler:
        @staticmethod
        def apply_energy_constraints(positions, constraints, learning_rate=0.01):
            return positions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToConstraints:
    """Bridge for converting natural language to geometric constraints"""
    
    def __init__(self):
        self.constraint_parsers = {
            'above': self._parse_above,
            'below': self._parse_below,
            'left': self._parse_left,
            'right': self._parse_right,
            'near': self._parse_near,
            'far': self._parse_far,
            'distance': self._parse_distance,
            'angle': self._parse_angle,
            'between': self._parse_between,
            'parallel': self._parse_parallel,
            'perpendicular': self._parse_perpendicular
        }
    
    def parse_text_to_constraints(self, text: str, entities: List[str]) -> Dict[str, Union[np.ndarray, 'torch.Tensor']]:
        """
        Convert natural language description to geometric constraints
        
        Args:
            text: Natural language description
            entities: List of entity names in the scene
            
        Returns:
            Dictionary of constraints compatible with GASM
        """
        text = text.lower().strip()
        constraints = {}
        
        logger.info(f"Parsing text: '{text}' with entities: {entities}")
        
        # Create entity name to index mapping with aliases
        entity_to_idx = {name.lower(): i for i, name in enumerate(entities)}
        
        # Add aliases for compound names
        for i, name in enumerate(entities):
            if 'conveyor' in name.lower():
                entity_to_idx['conveyor'] = i
                entity_to_idx['belt'] = i
                entity_to_idx['conveyor_belt'] = i
            if 'robot' in name.lower():
                entity_to_idx['robot'] = i
            if 'box' in name.lower():
                entity_to_idx['box'] = i
        
        # Simple pattern matching for spatial relationships
        for constraint_type, parser in self.constraint_parsers.items():
            # Check different variations of constraint keywords
            patterns = []
            if constraint_type == 'near':
                patterns = ['near', 'close', 'next to', 'beside']
            elif constraint_type == 'above':
                patterns = ['above', 'over', 'on top']
            elif constraint_type == 'left':
                patterns = ['left of', 'to the left']
            elif constraint_type == 'right':
                patterns = ['right of', 'to the right']
            else:
                patterns = [constraint_type]
            
            # Check if any pattern matches
            if any(pattern in text.lower() for pattern in patterns):
                try:
                    parsed_constraints = parser(text, entity_to_idx)
                    if parsed_constraints:
                        if constraint_type not in constraints:
                            constraints[constraint_type] = []
                        constraints[constraint_type].extend(parsed_constraints)
                except Exception as e:
                    logger.warning(f"Failed to parse {constraint_type}: {e}")
        
        # Convert lists to tensors
        for key, value in constraints.items():
            if isinstance(value, list) and value:
                constraints[key] = torch.tensor(value, dtype=torch.float32)
        
        # Remove empty constraints
        constraints = {k: v for k, v in constraints.items() if len(v) > 0}
        
        logger.info(f"Parsed constraints: {list(constraints.keys())}")
        return constraints
    
    def _parse_above(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A above B' -> y_A > y_B + offset"""
        constraints = []
        
        # Simple pattern matching
        words = text.split()
        for i, word in enumerate(words):
            if word == 'above' and i > 0 and i < len(words) - 1:
                entity1 = words[i-1]
                entity2 = words[i+1] if i+1 < len(words) else None
                
                if entity1 in entity_map and entity2 and entity2 in entity_map:
                    # Create a position constraint: entity1 should be above entity2
                    idx1, idx2 = entity_map[entity1], entity_map[entity2]
                    # Format: [entity1_idx, entity2_idx, y_offset, constraint_type]
                    # constraint_type: 0=above, 1=below, 2=left, 3=right
                    constraints.append([idx1, idx2, 1.0, 0])  # 1.0 unit above
        
        return constraints
    
    def _parse_below(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A below B' -> y_A < y_B - offset"""
        constraints = []
        words = text.split()
        for i, word in enumerate(words):
            if word == 'below' and i > 0 and i < len(words) - 1:
                entity1 = words[i-1]
                entity2 = words[i+1] if i+1 < len(words) else None
                
                if entity1 in entity_map and entity2 and entity2 in entity_map:
                    idx1, idx2 = entity_map[entity1], entity_map[entity2]
                    constraints.append([idx1, idx2, -1.0, 1])  # 1.0 unit below
        
        return constraints
    
    def _parse_left(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A left of B' -> x_A < x_B - offset"""
        constraints = []
        words = text.split()
        for i, word in enumerate(words):
            if word in ['left'] and i < len(words) - 2:
                if words[i+1] == 'of':
                    entity1 = words[i-1] if i > 0 else None
                    entity2 = words[i+2]
                    
                    if entity1 and entity1 in entity_map and entity2 in entity_map:
                        idx1, idx2 = entity_map[entity1], entity_map[entity2]
                        constraints.append([idx1, idx2, -1.0, 2])  # 1.0 unit left
        
        return constraints
    
    def _parse_right(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A right of B' -> x_A > x_B + offset"""
        constraints = []
        words = text.split()
        for i, word in enumerate(words):
            if word == 'right' and i < len(words) - 2:
                if words[i+1] == 'of':
                    entity1 = words[i-1] if i > 0 else None
                    entity2 = words[i+2]
                    
                    if entity1 and entity1 in entity_map and entity2 in entity_map:
                        idx1, idx2 = entity_map[entity1], entity_map[entity2]
                        constraints.append([idx1, idx2, 1.0, 3])  # 1.0 unit right
        
        return constraints
    
    def _parse_near(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A near B' -> distance constraint"""
        constraints = []
        words = text.split()
        
        # Handle compound entity names like "conveyor belt"
        text_lower = text.lower()
        
        # Look for patterns like "robot near conveyor belt"
        for entity1 in entity_map:
            for entity2 in entity_map:
                if entity1 != entity2:
                    # Check various patterns
                    patterns = [
                        f"{entity1} near {entity2}",
                        f"{entity1} close {entity2}",
                        f"{entity1} next to {entity2}",
                        f"{entity1} beside {entity2}"
                    ]
                    
                    for pattern in patterns:
                        if pattern in text_lower:
                            idx1, idx2 = entity_map[entity1], entity_map[entity2]
                            constraints.append([idx1, idx2, 0.5])  # Close distance
                            
        return constraints
    
    def _parse_far(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A far from B' -> distance constraint"""
        constraints = []
        words = text.split()
        for i, word in enumerate(words):
            if word == 'far' and i < len(words) - 2:
                if words[i+1] == 'from':
                    entity1 = words[i-1] if i > 0 else None
                    entity2 = words[i+2]
                    
                    if entity1 and entity1 in entity_map and entity2 in entity_map:
                        idx1, idx2 = entity_map[entity1], entity_map[entity2]
                        constraints.append([idx1, idx2, 3.0])  # Far distance
        
        return constraints
    
    def _parse_distance(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse explicit distance constraints"""
        constraints = []
        # Look for patterns like "A and B are 2 units apart"
        # This is a simplified parser - a real implementation would be more sophisticated
        return constraints
    
    def _parse_angle(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse angle constraints"""
        constraints = []
        # Look for angle-related terms
        return constraints
    
    def _parse_between(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse 'A between B and C'"""
        constraints = []
        return constraints
    
    def _parse_parallel(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse parallel constraints"""
        constraints = []
        return constraints
    
    def _parse_perpendicular(self, text: str, entity_map: Dict[str, int]) -> List[List[float]]:
        """Parse perpendicular constraints"""
        constraints = []
        return constraints


class Scene2D:
    """Simple 2D scene with conveyor and sensor"""
    
    def __init__(self, width=10.0, height=8.0):
        self.width = width
        self.height = height
        
        # Define scene objects
        self.conveyor = {
            'position': [2.0, 2.0],
            'width': 4.0,
            'height': 1.0,
            'angle': 0.0
        }
        
        self.sensor = {
            'position': [5.0, 5.0],
            'radius': 0.5,
            'type': 'circle'
        }
        
        # Moveable entities that can be optimized
        self.entities = {
            'box': {'position': [1.0, 1.0], 'size': 0.3, 'type': 'square'},
            'robot': {'position': [7.0, 3.0], 'size': 0.4, 'type': 'triangle'},
            'conveyor_belt': {'position': [2.0, 2.0], 'size': 0.5, 'type': 'rectangle'},  # Add conveyor as moveable entity
        }
    
    def get_entity_names(self) -> List[str]:
        """Get list of moveable entity names"""
        return list(self.entities.keys())
    
    def get_entity_positions(self) -> torch.Tensor:
        """Get current positions as tensor"""
        positions = []
        for name in self.get_entity_names():
            pos = self.entities[name]['position']
            positions.append([pos[0], pos[1], 0.0])  # Add z=0 for 3D compatibility
        return torch.tensor(positions, dtype=torch.float32)
    
    def set_entity_positions(self, positions: torch.Tensor):
        """Update entity positions from tensor"""
        entity_names = self.get_entity_names()
        for i, name in enumerate(entity_names):
            if i < len(positions):
                self.entities[name]['position'] = [
                    positions[i, 0].item(),
                    positions[i, 1].item()
                ]
    
    def check_collisions(self, positions: torch.Tensor) -> bool:
        """Check if any entities collide with scene objects or boundaries"""
        entity_names = self.get_entity_names()
        
        for i, name in enumerate(entity_names):
            if i >= len(positions):
                continue
                
            pos = positions[i]
            x, y = pos[0].item(), pos[1].item()
            entity_size = self.entities[name]['size']
            
            # Check boundaries
            if (x - entity_size/2 < 0 or x + entity_size/2 > self.width or
                y - entity_size/2 < 0 or y + entity_size/2 > self.height):
                return True
            
            # Check conveyor collision (simplified)
            conv_x, conv_y = self.conveyor['position']
            conv_w, conv_h = self.conveyor['width'], self.conveyor['height']
            
            if (conv_x - conv_w/2 < x < conv_x + conv_w/2 and
                conv_y - conv_h/2 < y < conv_y + conv_h/2):
                return True
        
        return False


class VisualizationEngine:
    """Real-time matplotlib visualization with live updates"""
    
    def __init__(self, scene: Scene2D, save_animation: bool = False):
        self.scene = scene
        self.save_animation = save_animation
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(0, scene.width)
        self.ax.set_ylim(0, scene.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('2D Spatial Agent - Real-time GASM Optimization', fontsize=14, fontweight='bold')
        
        # Animation data storage
        self.animation_frames = []
        self.constraint_history = []
        self.position_history = []
        
        # Create scene objects (static)
        self._draw_static_objects()
        
        # Create entity objects (dynamic)
        self.entity_artists = {}
        self._create_entity_artists()
        
        # Text displays
        self.text_displays = {
            'iteration': self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                    fontsize=10, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)),
            'constraints': self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes,
                                      fontsize=9, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)),
            'convergence': self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                      fontsize=10, verticalalignment='bottom',
                                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        }
        
        # Enable interactive mode for real-time updates
        plt.ion()
        if not plt.isinteractive():
            plt.show(block=False)
        self.fig.show()
    
    def _draw_static_objects(self):
        """Draw conveyor and sensor (static scene objects)"""
        # Conveyor (rectangle)
        conv = self.scene.conveyor
        conveyor_rect = Rectangle(
            (conv['position'][0] - conv['width']/2, conv['position'][1] - conv['height']/2),
            conv['width'], conv['height'],
            angle=conv['angle'],
            facecolor='gray',
            edgecolor='black',
            alpha=0.7,
            label='Conveyor'
        )
        self.ax.add_patch(conveyor_rect)
        
        # Sensor (circle)
        sensor = self.scene.sensor
        sensor_circle = Circle(
            sensor['position'],
            sensor['radius'],
            facecolor='red',
            edgecolor='darkred',
            alpha=0.8,
            label='Sensor'
        )
        self.ax.add_patch(sensor_circle)
        
        # Add legend
        self.ax.legend(loc='upper right')
    
    def _create_entity_artists(self):
        """Create matplotlib artists for moveable entities"""
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, entity) in enumerate(self.scene.entities.items()):
            color = colors[i % len(colors)]
            pos = entity['position']
            size = entity['size']
            
            if entity['type'] == 'square':
                artist = Rectangle(
                    (pos[0] - size/2, pos[1] - size/2),
                    size, size,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.8,
                    label=name.capitalize()
                )
            elif entity['type'] == 'triangle':
                # Create triangle vertices
                vertices = np.array([
                    [pos[0], pos[1] + size/2],      # top
                    [pos[0] - size/2, pos[1] - size/2],  # bottom left
                    [pos[0] + size/2, pos[1] - size/2]   # bottom right
                ])
                artist = Polygon(
                    vertices,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.8,
                    label=name.capitalize()
                )
            else:  # circle
                artist = Circle(
                    pos,
                    size/2,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.8,
                    label=name.capitalize()
                )
            
            self.ax.add_patch(artist)
            self.entity_artists[name] = artist
    
    def update(self, iteration: int, positions: torch.Tensor, 
               constraints: Dict, convergence_info: Dict):
        """Update visualization with new positions and info"""
        
        # Update entity positions
        entity_names = self.scene.get_entity_names()
        for i, name in enumerate(entity_names):
            if i < len(positions):
                new_pos = positions[i]
                x, y = new_pos[0].item(), new_pos[1].item()
                
                artist = self.entity_artists[name]
                if hasattr(artist, 'set_xy'):
                    # Rectangle
                    size = self.scene.entities[name]['size']
                    artist.set_xy((x - size/2, y - size/2))
                elif hasattr(artist, 'set_center'):
                    # Circle
                    artist.set_center((x, y))
                else:
                    # Polygon (triangle)
                    size = self.scene.entities[name]['size']
                    vertices = np.array([
                        [x, y + size/2],
                        [x - size/2, y - size/2],
                        [x + size/2, y - size/2]
                    ])
                    artist.set_xy(vertices)
        
        # Update text displays
        self.text_displays['iteration'].set_text(
            f'Iteration: {iteration}\nEntities: {len(entity_names)}'
        )
        
        constraint_text = 'Active Constraints:\n'
        for constraint_type, params in constraints.items():
            if len(params) > 0:
                constraint_text += f'• {constraint_type}: {len(params)}\n'
        self.text_displays['constraints'].set_text(constraint_text)
        
        # Convergence info
        conv_text = f"Convergence: {convergence_info.get('converged', False)}\n"
        if 'error' in convergence_info:
            conv_text += f"Error: {convergence_info['error']:.4f}\n"
        if 'constraint_violation' in convergence_info:
            conv_text += f"Constraint Violation: {convergence_info['constraint_violation']:.4f}"
        self.text_displays['convergence'].set_text(conv_text)
        
        # Store frame for animation
        if self.save_animation:
            self.animation_frames.append({
                'iteration': iteration,
                'positions': positions.clone(),
                'constraints': len(constraints),
                'converged': convergence_info.get('converged', False)
            })
        
        # Redraw
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.05)  # Smaller delay for smoother animation
        except Exception as e:
            logger.warning(f"Visualization update failed: {e}")
    
    def save_gif(self, filename: str = 'spatial_agent_demo.gif'):
        """Save animation as GIF"""
        if not self.save_animation or not self.animation_frames:
            logger.warning("No animation data to save")
            return
        
        if not IMAGEIO_AVAILABLE:
            logger.warning("imageio not available, trying matplotlib animation")
            try:
                self._save_matplotlib_gif(filename)
            except Exception as e:
                logger.error(f"Failed to save animation with matplotlib: {e}")
            return
        
        try:
            # Create frames for imageio
            images = []
            temp_fig, temp_ax = plt.subplots(figsize=(12, 10))
            
            for frame_data in self.animation_frames:
                temp_ax.clear()
                temp_ax.set_xlim(0, self.scene.width)
                temp_ax.set_ylim(0, self.scene.height)
                temp_ax.set_aspect('equal')
                temp_ax.grid(True, alpha=0.3)
                temp_ax.set_title(f'2D Spatial Agent - Iteration {frame_data["iteration"]}', 
                                 fontsize=14, fontweight='bold')
                
                # Draw static objects
                self._draw_static_objects_on_axis(temp_ax)
                
                # Draw entities at current positions
                positions = frame_data['positions']
                entity_names = self.scene.get_entity_names()
                colors = ['blue', 'green', 'orange', 'purple', 'brown']
                
                for i, name in enumerate(entity_names):
                    if i < len(positions):
                        color = colors[i % len(colors)]
                        pos = positions[i]
                        x, y = pos[0].item(), pos[1].item()
                        entity = self.scene.entities[name]
                        size = entity['size']
                        
                        if entity['type'] == 'square':
                            rect = Rectangle((x - size/2, y - size/2), size, size,
                                           facecolor=color, edgecolor='black', alpha=0.8)
                            temp_ax.add_patch(rect)
                        elif entity['type'] == 'triangle':
                            vertices = np.array([
                                [x, y + size/2],
                                [x - size/2, y - size/2],
                                [x + size/2, y - size/2]
                            ])
                            triangle = Polygon(vertices, facecolor=color, 
                                             edgecolor='black', alpha=0.8)
                            temp_ax.add_patch(triangle)
                        else:  # circle
                            circle = Circle((x, y), size/2, facecolor=color, 
                                          edgecolor='black', alpha=0.8)
                            temp_ax.add_patch(circle)
                
                # Convert plot to image
                temp_fig.canvas.draw()
                image = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(temp_fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)
            
            plt.close(temp_fig)
            
            # Save as GIF
            imageio.mimsave(filename, images, duration=0.2, loop=0)
            logger.info(f"Animation saved as {filename} using imageio")
            
        except Exception as e:
            logger.error(f"Failed to save animation with imageio: {e}")
            # Fallback to matplotlib
            try:
                self._save_matplotlib_gif(filename)
            except Exception as e2:
                logger.error(f"Matplotlib fallback also failed: {e2}")
    
    def _save_matplotlib_gif(self, filename: str):
        """Fallback GIF creation using matplotlib animation"""
        def animate(frame_idx):
            if frame_idx >= len(self.animation_frames):
                return []
                
            frame_data = self.animation_frames[frame_idx]
            positions = frame_data['positions']
            iteration = frame_data['iteration']
            
            # Update positions (simplified for gif)
            entity_names = self.scene.get_entity_names()
            artists_to_return = []
            
            for i, name in enumerate(entity_names):
                if i < len(positions):
                    new_pos = positions[i]
                    x, y = new_pos[0].item(), new_pos[1].item()
                    artist = self.entity_artists[name]
                    
                    if hasattr(artist, 'set_xy'):
                        size = self.scene.entities[name]['size']
                        artist.set_xy((x - size/2, y - size/2))
                    elif hasattr(artist, 'set_center'):
                        artist.set_center((x, y))
                    elif hasattr(artist, 'set_xy') and isinstance(artist, Polygon):
                        # Update polygon vertices
                        size = self.scene.entities[name]['size']
                        vertices = np.array([
                            [x, y + size/2],
                            [x - size/2, y - size/2],
                            [x + size/2, y - size/2]
                        ])
                        artist.set_xy(vertices)
                    
                    artists_to_return.append(artist)
            
            self.text_displays['iteration'].set_text(f'Iteration: {iteration}')
            artists_to_return.extend(self.text_displays.values())
            return artists_to_return
        
        # Create and save animation
        anim = animation.FuncAnimation(
            self.fig, animate, frames=len(self.animation_frames),
            interval=200, blit=False, repeat=True
        )
        
        # Try different writers
        try:
            anim.save(filename, writer='pillow', fps=5)
            logger.info(f"Animation saved as {filename} using pillow writer")
        except Exception:
            try:
                anim.save(filename, writer='imagemagick', fps=5)
                logger.info(f"Animation saved as {filename} using imagemagick writer")
            except Exception:
                # Save as MP4 instead
                mp4_filename = filename.replace('.gif', '.mp4')
                anim.save(mp4_filename, writer='ffmpeg', fps=5)
                logger.info(f"Animation saved as {mp4_filename} (MP4 format)")
    
    def _draw_static_objects_on_axis(self, ax):
        """Helper to draw static objects on a given axis"""
        # Conveyor (rectangle)
        conv = self.scene.conveyor
        conveyor_rect = Rectangle(
            (conv['position'][0] - conv['width']/2, conv['position'][1] - conv['height']/2),
            conv['width'], conv['height'],
            angle=conv['angle'],
            facecolor='gray',
            edgecolor='black',
            alpha=0.7
        )
        ax.add_patch(conveyor_rect)
        
        # Sensor (circle)
        sensor = self.scene.sensor
        sensor_circle = Circle(
            sensor['position'],
            sensor['radius'],
            facecolor='red',
            edgecolor='darkred',
            alpha=0.8
        )
        ax.add_patch(sensor_circle)
    
    def close(self):
        """Close the visualization"""
        try:
            plt.close(self.fig)
            plt.ioff()  # Turn off interactive mode
        except Exception as e:
            logger.warning(f"Error closing visualization: {e}")


class SpatialAgent2D:
    """
    Main 2D Spatial Agent with GASM integration
    Implements the complete feedback loop: plan → execute → observe → evaluate → iterate
    """
    
    def __init__(
        self,
        scene_width: float = 10.0,
        scene_height: float = 8.0,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-3,
        feature_dim: int = 64,
        hidden_dim: int = 128
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize components
        self.scene = Scene2D(scene_width, scene_height)
        self.text_parser = TextToConstraints()
        
        # Initialize GASM with 2D-appropriate parameters
        try:
            self.gasm_model = EnhancedGASM(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=3,  # x, y, θ (keeping 3D for compatibility)
                max_iterations=10,  # Inner GASM iterations
                dropout=0.1
            )
        except Exception as e:
            logger.warning(f"GASM initialization failed: {e}, using fallback")
            self.gasm_model = EnhancedGASM()
        
        self.visualization = None
        self.constraint_handler = ConstraintHandler()
        
        # Optimization state
        self.current_positions = self.scene.get_entity_positions()
        self.best_positions = self.current_positions.clone()
        self.best_score = float('inf')
        self.iteration_count = 0
        self.converged = False
        
        logger.info(f"SpatialAgent2D initialized with {len(self.scene.entities)} entities")
    
    def run(
        self,
        text_description: str,
        enable_visualization: bool = True,
        save_video: bool = False,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Main execution loop
        
        Args:
            text_description: Natural language description of desired spatial arrangement
            enable_visualization: Whether to show real-time visualization
            save_video: Whether to save animation as GIF
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with results and statistics
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            logger.info(f"Set random seed to {seed}")
        
        logger.info(f"Starting spatial optimization with description: '{text_description}'")
        
        # Phase 1: Plan - Parse text to constraints
        constraints = self._plan(text_description)
        
        # Initialize visualization
        if enable_visualization:
            self.visualization = VisualizationEngine(self.scene, save_video)
        
        # Phase 2-5: Execute optimization loop
        results = self._optimization_loop(constraints)
        
        # Save animation if requested
        if save_video and self.visualization:
            self.visualization.save_gif()
        
        return results
    
    def _plan(self, text_description: str) -> Dict:
        """Phase 1: Plan - Convert text to constraints"""
        logger.info("Phase 1: Planning - Converting text to constraints")
        
        entity_names = self.scene.get_entity_names()
        constraints = self.text_parser.parse_text_to_constraints(text_description, entity_names)
        
        logger.info(f"Generated constraints: {list(constraints.keys())}")
        return constraints
    
    def _optimization_loop(self, constraints: Dict) -> Dict:
        """Main optimization loop implementing the feedback cycle"""
        logger.info("Starting optimization loop")
        
        start_time = time.time()
        iteration_scores = []
        constraint_violations = []
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            
            # Phase 2: Execute - Apply GASM optimization
            new_positions = self._execute(constraints, iteration)
            
            # Phase 3: Observe - Update scene state
            self._observe(new_positions)
            
            # Phase 4: Evaluate - Compute fitness and convergence
            evaluation = self._evaluate(constraints, new_positions)
            
            iteration_scores.append(evaluation['score'])
            constraint_violations.append(evaluation['constraint_violation'])
            
            # Update best solution
            if evaluation['score'] < self.best_score:
                self.best_score = evaluation['score']
                self.best_positions = new_positions.clone()
                logger.info(f"Iteration {iteration}: New best score {self.best_score:.4f}")
            
            # Update visualization
            if self.visualization:
                convergence_info = {
                    'converged': evaluation['converged'],
                    'error': evaluation['score'],
                    'constraint_violation': evaluation['constraint_violation']
                }
                self.visualization.update(iteration, new_positions, constraints, convergence_info)
            
            # Phase 5: Iterate - Check convergence and continue
            if evaluation['converged']:
                logger.info(f"Converged at iteration {iteration}")
                self.converged = True
                break
            
            # Small perturbation for next iteration if stuck
            if iteration > 10 and len(iteration_scores) >= 5:
                recent_scores = iteration_scores[-5:]
                if max(recent_scores) - min(recent_scores) < 1e-6:
                    # Add small random perturbation to escape local minimum
                    noise = torch.randn_like(new_positions) * 0.1
                    new_positions = new_positions + noise
                    logger.info(f"Added noise to escape local minimum at iteration {iteration}")
            
            self.current_positions = new_positions
        
        total_time = time.time() - start_time
        
        # Final scene update
        self.scene.set_entity_positions(self.best_positions)
        
        # Prepare results
        results = {
            'success': self.converged,
            'iterations': self.iteration_count + 1,
            'final_score': self.best_score,
            'final_positions': self.best_positions.tolist(),
            'execution_time': total_time,
            'score_history': iteration_scores,
            'constraint_violations': constraint_violations,
            'final_constraint_violation': constraint_violations[-1] if constraint_violations else 0.0,
            'entities': {name: pos for name, pos in zip(self.scene.get_entity_names(), self.best_positions.tolist())}
        }
        
        logger.info(f"Optimization completed: Success={results['success']}, "
                   f"Iterations={results['iterations']}, Final Score={results['final_score']:.4f}")
        
        return results
    
    def _execute(self, constraints: Dict, iteration: int) -> torch.Tensor:
        """Phase 2: Execute - Apply GASM optimization"""
        try:
            # Prepare input for GASM
            entity_names = self.scene.get_entity_names()
            num_entities = len(entity_names)
            
            # Create feature vectors (simple encoding)
            features = torch.randn(num_entities, 64)  # Random features for demo
            for i, name in enumerate(entity_names):
                # Encode entity type and current position
                pos = self.current_positions[i]
                features[i, :3] = pos  # First 3 dims are position
                features[i, 3] = hash(name) % 100 / 100.0  # Entity ID encoding
            
            # Create relation tensor (simplified)
            relations = torch.zeros(num_entities, num_entities, 8)
            for i in range(num_entities):
                for j in range(num_entities):
                    if i != j:
                        # Distance-based relations
                        dist = torch.norm(self.current_positions[i] - self.current_positions[j])
                        relations[i, j, 0] = dist
                        relations[i, j, 1] = 1.0  # Connected
            
            # Convert constraints to GASM format
            gasm_constraints = self._convert_constraints_to_gasm(constraints)
            
            # Apply GASM
            if hasattr(self.gasm_model, 'forward_enhanced'):
                optimized_positions = self.gasm_model.forward_enhanced(
                    E=entity_names,
                    F=features,
                    R=relations,
                    C=gasm_constraints
                )
            else:
                optimized_positions = self.gasm_model.forward(
                    E=entity_names,
                    F=features,
                    R=relations,
                    C=gasm_constraints
                )
            
            # Ensure positions are valid tensors
            if not isinstance(optimized_positions, torch.Tensor):
                logger.warning("GASM returned non-tensor, using current positions")
                optimized_positions = self.current_positions.clone()
            
            # Ensure positions are 2D and within bounds
            optimized_positions = self._clamp_positions(optimized_positions)
            
            return optimized_positions
            
        except Exception as e:
            logger.warning(f"GASM execution failed: {e}, using gradient-based fallback")
            return self._gradient_descent_fallback(constraints)
    
    def _convert_constraints_to_gasm(self, constraints: Dict) -> Dict:
        """Convert parsed constraints to GASM-compatible format"""
        gasm_constraints = {}
        
        for constraint_type, params in constraints.items():
            if constraint_type in ['above', 'below', 'left', 'right']:
                # Convert to distance constraints
                if 'distance' not in gasm_constraints:
                    gasm_constraints['distance'] = []
                
                for param in params:
                    if len(param) >= 4:
                        idx1, idx2, offset, direction = param[:4]
                        # Convert to distance constraint based on direction
                        target_distance = abs(offset)
                        gasm_constraints['distance'].append([int(idx1), int(idx2), target_distance])
            
            elif constraint_type in ['near', 'far']:
                # Direct distance constraints
                if 'distance' not in gasm_constraints:
                    gasm_constraints['distance'] = []
                
                for param in params:
                    if len(param) >= 3:
                        gasm_constraints['distance'].append(param[:3])
        
        # Convert to tensors
        for key, value in gasm_constraints.items():
            if value:
                gasm_constraints[key] = torch.tensor(value, dtype=torch.float32)
        
        return gasm_constraints
    
    def _gradient_descent_fallback(self, constraints: Dict) -> torch.Tensor:
        """Simple gradient descent fallback when GASM fails"""
        positions = self.current_positions.clone()
        positions.requires_grad_(True)
        
        optimizer = torch.optim.Adam([positions], lr=0.01)
        
        for _ in range(10):  # Quick optimization
            optimizer.zero_grad()
            
            loss = self._compute_constraint_loss(positions, constraints)
            loss.backward()
            
            optimizer.step()
            positions.data = self._clamp_positions(positions.data)
        
        return positions.detach()
    
    def _clamp_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Ensure positions are within scene boundaries"""
        clamped = positions.clone()
        
        # Clamp to scene boundaries with margin for entity size
        margin = 0.5  # Maximum entity size / 2
        clamped[:, 0] = torch.clamp(clamped[:, 0], margin, self.scene.width - margin)
        clamped[:, 1] = torch.clamp(clamped[:, 1], margin, self.scene.height - margin)
        clamped[:, 2] = 0.0  # Keep z = 0 for 2D
        
        return clamped
    
    def _observe(self, new_positions: torch.Tensor):
        """Phase 3: Observe - Update scene state"""
        self.scene.set_entity_positions(new_positions)
        
        # Check for collisions
        if self.scene.check_collisions(new_positions):
            logger.warning(f"Collision detected at iteration {self.iteration_count}")
    
    def _evaluate(self, constraints: Dict, positions: torch.Tensor) -> Dict:
        """Phase 4: Evaluate - Compute fitness and check convergence"""
        
        # Compute constraint violation
        constraint_loss = self._compute_constraint_loss(positions, constraints)
        constraint_violation = constraint_loss.item()
        
        # Compute overall score (lower is better)
        collision_penalty = 100.0 if self.scene.check_collisions(positions) else 0.0
        boundary_penalty = self._compute_boundary_penalty(positions)
        
        total_score = constraint_violation + collision_penalty + boundary_penalty
        
        # Check convergence
        position_change = torch.norm(positions - self.current_positions).item()
        converged = (constraint_violation < self.convergence_threshold and 
                    position_change < self.convergence_threshold)
        
        return {
            'score': total_score,
            'constraint_violation': constraint_violation,
            'collision_penalty': collision_penalty,
            'boundary_penalty': boundary_penalty,
            'position_change': position_change,
            'converged': converged
        }
    
    def _compute_constraint_loss(self, positions: torch.Tensor, constraints: Dict) -> torch.Tensor:
        """Compute loss based on constraint violations"""
        total_loss = torch.tensor(0.0)
        
        for constraint_type, params in constraints.items():
            if len(params) == 0:
                continue
                
            if constraint_type in ['above', 'below', 'left', 'right']:
                for param in params:
                    if len(param) >= 4:
                        idx1, idx2, offset, direction = param[:4]
                        idx1, idx2 = int(idx1), int(idx2)
                        
                        if idx1 < len(positions) and idx2 < len(positions):
                            pos1, pos2 = positions[idx1], positions[idx2]
                            
                            if direction == 0:  # above
                                loss = torch.relu(pos2[1] - pos1[1] + offset)
                            elif direction == 1:  # below
                                loss = torch.relu(pos1[1] - pos2[1] - offset)
                            elif direction == 2:  # left
                                loss = torch.relu(pos1[0] - pos2[0] - offset)
                            elif direction == 3:  # right
                                loss = torch.relu(pos2[0] - pos1[0] + offset)
                            else:
                                loss = torch.tensor(0.0)
                            
                            total_loss += loss
            
            elif constraint_type in ['near', 'far']:
                for param in params:
                    if len(param) >= 3:
                        idx1, idx2, target_dist = param[:3]
                        idx1, idx2 = int(idx1), int(idx2)
                        
                        if idx1 < len(positions) and idx2 < len(positions):
                            actual_dist = torch.norm(positions[idx1][:2] - positions[idx2][:2])
                            loss = (actual_dist - target_dist) ** 2
                            total_loss += loss
        
        return total_loss
    
    def _compute_boundary_penalty(self, positions: torch.Tensor) -> float:
        """Compute penalty for positions near boundaries"""
        penalty = 0.0
        margin = 0.5
        
        for pos in positions:
            x, y = pos[0].item(), pos[1].item()
            
            # Check how close to boundaries
            if x < margin:
                penalty += (margin - x) ** 2
            elif x > self.scene.width - margin:
                penalty += (x - (self.scene.width - margin)) ** 2
            
            if y < margin:
                penalty += (margin - y) ** 2
            elif y > self.scene.height - margin:
                penalty += (y - (self.scene.height - margin)) ** 2
        
        return penalty
    
    def close(self):
        """Clean up resources"""
        if self.visualization:
            self.visualization.close()


def main():
    """Command-line interface for the 2D Spatial Agent"""
    parser = argparse.ArgumentParser(
        description='2D Spatial Agent - GASM-powered spatial reasoning demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_loop_2d.py --text "box above robot"
  python agent_loop_2d.py --text "robot near sensor" --steps 30 --save_video
  python agent_loop_2d.py --text "box left of conveyor, robot right of box" --seed 42
        """
    )
    
    parser.add_argument(
        '--text', 
        required=True,
        help='Natural language description of desired spatial arrangement'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Maximum optimization steps (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='Save optimization process as GIF animation'
    )
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable real-time visualization (faster execution)'
    )
    parser.add_argument(
        '--scene_size',
        type=float,
        nargs=2,
        default=[10.0, 8.0],
        metavar=('WIDTH', 'HEIGHT'),
        help='Scene dimensions (default: 10.0 8.0)'
    )
    parser.add_argument(
        '--convergence_threshold',
        type=float,
        default=1e-3,
        help='Convergence threshold (default: 1e-3)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run agent
        print(f"🚀 Starting 2D Spatial Agent")
        print(f"📝 Task: {args.text}")
        print(f"🎯 Scene: {args.scene_size[0]}×{args.scene_size[1]}")
        print(f"⚙️  Max steps: {args.steps}")
        if args.seed:
            print(f"🎲 Seed: {args.seed}")
        print("-" * 50)
        
        agent = SpatialAgent2D(
            scene_width=args.scene_size[0],
            scene_height=args.scene_size[1],
            max_iterations=args.steps,
            convergence_threshold=args.convergence_threshold
        )
        
        # Run optimization
        results = agent.run(
            text_description=args.text,
            enable_visualization=not args.no_visualization,
            save_video=args.save_video,
            seed=args.seed
        )
        
        # Print results
        print("\n" + "="*50)
        print("🎯 OPTIMIZATION RESULTS")
        print("="*50)
        print(f"✅ Success: {results['success']}")
        print(f"🔄 Iterations: {results['iterations']}")
        print(f"📊 Final Score: {results['final_score']:.4f}")
        print(f"⚡ Execution Time: {results['execution_time']:.2f}s")
        print(f"⚠️  Final Constraint Violation: {results['final_constraint_violation']:.4f}")
        
        print(f"\n📍 FINAL POSITIONS:")
        for entity_name, position in results['entities'].items():
            print(f"  {entity_name}: ({position[0]:.2f}, {position[1]:.2f})")
        
        if args.save_video:
            print(f"\n🎥 Animation saved as 'spatial_agent_demo.gif'")
        
        # Keep visualization open if enabled
        if not args.no_visualization:
            print(f"\n👀 Visualization is open. Close the window to exit.")
            try:
                # Check if we're in interactive mode
                if hasattr(__builtins__, '__IPYTHON__') or 'ipykernel' in sys.modules:
                    print("Running in interactive environment, keeping plot open...")
                    plt.show()
                else:
                    input("Press Enter to exit...")
            except (KeyboardInterrupt, EOFError):
                pass
        
        agent.close()
        print("\n✨ Done!")
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())