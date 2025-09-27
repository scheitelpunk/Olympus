"""Mock GASM-Robotics integration for testing.

Provides mock objects for GASM-Robotics components including
material databases, robot configurations, and simulation environments.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock
from pathlib import Path


class MockGASMRobotics:
    """Mock GASM-Robotics system for testing."""
    
    def __init__(self):
        """Initialize mock GASM system."""
        self.materials_database = self._create_materials_database()
        self.robot_configs = self._create_robot_configs()
        self.simulation_config = self._create_simulation_config()
        
    def _create_materials_database(self) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive mock materials database."""
        return {
            'steel': {
                'color': [0.7, 0.7, 0.7, 1.0],
                'friction': 0.8,
                'restitution': 0.2,
                'density': 7850,  # kg/m³
                'young_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'thermal_conductivity': 50.0,  # W/mK
                'specific_heat': 450,  # J/kgK
                'melting_point': 1500,  # °C
                'category': 'metal'
            },
            'aluminum': {
                'color': [0.9, 0.9, 0.9, 1.0],
                'friction': 0.6,
                'restitution': 0.4,
                'density': 2700,
                'young_modulus': 70e9,
                'poisson_ratio': 0.33,
                'thermal_conductivity': 237.0,
                'specific_heat': 900,
                'melting_point': 660,
                'category': 'metal'
            },
            'rubber': {
                'color': [0.2, 0.2, 0.2, 1.0],
                'friction': 1.2,
                'restitution': 0.9,
                'density': 1200,
                'young_modulus': 1e6,
                'poisson_ratio': 0.47,
                'thermal_conductivity': 0.16,
                'specific_heat': 1400,
                'melting_point': 150,
                'category': 'polymer'
            },
            'glass': {
                'color': [0.9, 0.9, 1.0, 0.3],
                'friction': 0.2,
                'restitution': 0.1,
                'density': 2500,
                'young_modulus': 70e9,
                'poisson_ratio': 0.22,
                'thermal_conductivity': 1.4,
                'specific_heat': 840,
                'melting_point': 1700,
                'category': 'ceramic'
            },
            'plastic': {
                'color': [0.1, 0.5, 0.8, 1.0],
                'friction': 0.4,
                'restitution': 0.6,
                'density': 1200,
                'young_modulus': 2e9,
                'poisson_ratio': 0.35,
                'thermal_conductivity': 0.2,
                'specific_heat': 1200,
                'melting_point': 200,
                'category': 'polymer'
            }
        }
    
    def _create_robot_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create mock robot configurations."""
        return {
            'franka_panda': {
                'name': 'Franka Emika Panda',
                'dof': 7,
                'end_effector': 'parallel_gripper',
                'workspace': {
                    'x_range': [-0.8, 0.8],
                    'y_range': [-0.8, 0.8],
                    'z_range': [0.0, 1.2]
                },
                'joint_limits': {
                    'position': [
                        [-2.8973, 2.8973],
                        [-1.7628, 1.7628],
                        [-2.8973, 2.8973],
                        [-3.0718, -0.0698],
                        [-2.8973, 2.8973],
                        [-0.0175, 3.7525],
                        [-2.8973, 2.8973]
                    ],
                    'velocity': [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
                    'effort': [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
                },
                'tactile_sensors': {
                    'fingertip_left': {
                        'link_id': 10,
                        'sensor_type': 'force_torque',
                        'resolution': 0.1,  # N
                        'range': [-50, 50]  # N
                    },
                    'fingertip_right': {
                        'link_id': 11,
                        'sensor_type': 'force_torque',
                        'resolution': 0.1,
                        'range': [-50, 50]
                    },
                    'palm': {
                        'link_id': 8,
                        'sensor_type': 'pressure_array',
                        'resolution': 0.01,  # N/cm²
                        'array_size': [16, 16]
                    }
                }
            },
            'ur5e': {
                'name': 'Universal Robots UR5e',
                'dof': 6,
                'end_effector': 'robotiq_2f',
                'workspace': {
                    'x_range': [-0.85, 0.85],
                    'y_range': [-0.85, 0.85],
                    'z_range': [0.0, 1.0]
                },
                'joint_limits': {
                    'position': [
                        [-2*np.pi, 2*np.pi],
                        [-2*np.pi, 2*np.pi],
                        [-2*np.pi, 2*np.pi],
                        [-2*np.pi, 2*np.pi],
                        [-2*np.pi, 2*np.pi],
                        [-2*np.pi, 2*np.pi]
                    ],
                    'velocity': [3.15, 3.15, 3.15, 3.2, 3.2, 3.2],
                    'effort': [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
                },
                'tactile_sensors': {
                    'gripper_tip': {
                        'link_id': 8,
                        'sensor_type': 'force_torque',
                        'resolution': 0.5,
                        'range': [-100, 100]
                    }
                }
            }
        }
    
    def _create_simulation_config(self) -> Dict[str, Any]:
        """Create mock simulation configuration."""
        return {
            'physics_engine': 'bullet',
            'time_step': 1.0/240.0,
            'solver_iterations': 50,
            'gravity': [0, 0, -9.81],
            'real_time_factor': 1.0,
            'collision_margin': 0.001,
            'contact_breaking_threshold': 0.02,
            'visualization': {
                'enable_gui': False,
                'camera_distance': 1.5,
                'camera_yaw': 45,
                'camera_pitch': -30,
                'camera_target': [0, 0, 0.5]
            },
            'rendering': {
                'width': 640,
                'height': 480,
                'fov': 60,
                'near_plane': 0.1,
                'far_plane': 10.0
            }
        }
    
    def get_material_properties(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Get material properties by name."""
        return self.materials_database.get(material_name)
    
    def list_available_materials(self) -> List[str]:
        """List all available materials."""
        return list(self.materials_database.keys())
    
    def get_robot_config(self, robot_name: str) -> Optional[Dict[str, Any]]:
        """Get robot configuration by name."""
        return self.robot_configs.get(robot_name)
    
    def list_available_robots(self) -> List[str]:
        """List all available robots."""
        return list(self.robot_configs.keys())
    
    def compute_material_interaction(self, mat1: str, mat2: str) -> Dict[str, Any]:
        """Compute interaction properties between two materials."""
        props1 = self.get_material_properties(mat1)
        props2 = self.get_material_properties(mat2)
        
        if not props1 or not props2:
            return {}
        
        # Compute combined properties
        combined_friction = np.sqrt(props1['friction'] * props2['friction'])
        combined_restitution = min(props1['restitution'], props2['restitution'])
        
        # Effective modulus (harmonic mean)
        eff_modulus = (2 * props1['young_modulus'] * props2['young_modulus']) / \
                      (props1['young_modulus'] + props2['young_modulus'])
        
        return {
            'materials': [mat1, mat2],
            'combined_friction': combined_friction,
            'combined_restitution': combined_restitution,
            'effective_modulus': eff_modulus,
            'thermal_contact_resistance': abs(props1['thermal_conductivity'] - props2['thermal_conductivity']),
            'contact_stiffness': eff_modulus * 0.001,  # Simplified
            'expected_sound_frequency': np.sqrt(eff_modulus / ((props1['density'] + props2['density']) / 2)) / 100
        }


class MockMaterialDatabase:
    """Mock material database with search and filtering capabilities."""
    
    def __init__(self, gasm_system: MockGASMRobotics):
        """Initialize with GASM system reference."""
        self.gasm = gasm_system
        self.search_history = []
    
    def search_by_property(self, property_name: str, min_val: float, max_val: float) -> List[str]:
        """Search materials by property range."""
        results = []
        
        for mat_name, properties in self.gasm.materials_database.items():
            if property_name in properties:
                value = properties[property_name]
                if min_val <= value <= max_val:
                    results.append(mat_name)
        
        self.search_history.append({
            'property': property_name,
            'range': [min_val, max_val],
            'results': results
        })
        
        return results
    
    def search_by_category(self, category: str) -> List[str]:
        """Search materials by category."""
        results = []
        
        for mat_name, properties in self.gasm.materials_database.items():
            if properties.get('category') == category:
                results.append(mat_name)
        
        return results
    
    def get_similar_materials(self, reference_material: str, property_weights: Dict[str, float] = None) -> List[str]:
        """Find materials similar to reference material."""
        ref_props = self.gasm.get_material_properties(reference_material)
        if not ref_props:
            return []
        
        if property_weights is None:
            property_weights = {
                'friction': 1.0,
                'restitution': 1.0,
                'density': 0.5,
                'young_modulus': 0.8
            }
        
        similarities = []
        
        for mat_name, properties in self.gasm.materials_database.items():
            if mat_name == reference_material:
                continue
            
            similarity_score = 0.0
            total_weight = 0.0
            
            for prop, weight in property_weights.items():
                if prop in ref_props and prop in properties:
                    # Normalized difference
                    ref_val = ref_props[prop]
                    mat_val = properties[prop]
                    
                    if ref_val != 0:
                        diff = abs(ref_val - mat_val) / ref_val
                        similarity = 1.0 - min(diff, 1.0)  # Cap at 1.0
                    else:
                        similarity = 1.0 if mat_val == 0 else 0.0
                    
                    similarity_score += similarity * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_similarity = similarity_score / total_weight
                similarities.append((mat_name, avg_similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [mat_name for mat_name, score in similarities[:5]]  # Top 5


class MockRobotController:
    """Mock robot controller for testing robot operations."""
    
    def __init__(self, robot_config: Dict[str, Any]):
        """Initialize mock robot controller."""
        self.config = robot_config
        self.current_joint_positions = [0.0] * robot_config['dof']
        self.current_joint_velocities = [0.0] * robot_config['dof']
        self.current_joint_efforts = [0.0] * robot_config['dof']
        self.is_connected = False
        self.tactile_readings = {}
        
    def connect(self) -> bool:
        """Mock connection to robot."""
        self.is_connected = True
        self.tactile_readings = self._initialize_tactile_sensors()
        return True
    
    def disconnect(self):
        """Mock disconnection from robot."""
        self.is_connected = False
        self.tactile_readings = {}
    
    def _initialize_tactile_sensors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock tactile sensor readings."""
        readings = {}
        
        for sensor_name, sensor_config in self.config['tactile_sensors'].items():
            if sensor_config['sensor_type'] == 'force_torque':
                readings[sensor_name] = {
                    'force': [0.0, 0.0, 0.0],
                    'torque': [0.0, 0.0, 0.0],
                    'timestamp': 0.0
                }
            elif sensor_config['sensor_type'] == 'pressure_array':
                array_size = sensor_config['array_size']
                readings[sensor_name] = {
                    'pressure_array': np.zeros(array_size),
                    'total_force': 0.0,
                    'center_of_pressure': [0.0, 0.0],
                    'timestamp': 0.0
                }
        
        return readings
    
    def get_joint_states(self) -> Dict[str, List[float]]:
        """Get current joint states."""
        return {
            'positions': self.current_joint_positions.copy(),
            'velocities': self.current_joint_velocities.copy(),
            'efforts': self.current_joint_efforts.copy()
        }
    
    def move_to_joint_positions(self, target_positions: List[float], duration: float = 3.0) -> bool:
        """Mock move to joint positions."""
        if len(target_positions) != self.config['dof']:
            return False
        
        # Check joint limits
        for i, (pos, limits) in enumerate(zip(target_positions, self.config['joint_limits']['position'])):
            if not (limits[0] <= pos <= limits[1]):
                return False
        
        # Simulate movement
        self.current_joint_positions = target_positions.copy()
        self.current_joint_velocities = [0.0] * self.config['dof']
        
        return True
    
    def get_end_effector_pose(self) -> Dict[str, List[float]]:
        """Get end effector pose (mock forward kinematics)."""
        # Simplified forward kinematics for testing
        x = np.sum(np.cos(self.current_joint_positions[:3])) * 0.3
        y = np.sum(np.sin(self.current_joint_positions[:3])) * 0.3
        z = 0.5 + np.sum(self.current_joint_positions[3:]) * 0.1
        
        return {
            'position': [x, y, z],
            'orientation': [0, 0, 0, 1]  # Quaternion
        }
    
    def get_tactile_readings(self) -> Dict[str, Dict[str, Any]]:
        """Get current tactile sensor readings."""
        # Add some random noise to simulate real sensors
        readings = {}
        
        for sensor_name, reading in self.tactile_readings.items():
            noisy_reading = reading.copy()
            
            if 'force' in reading:
                noisy_reading['force'] = [
                    f + np.random.normal(0, 0.1) for f in reading['force']
                ]
                noisy_reading['torque'] = [
                    t + np.random.normal(0, 0.01) for t in reading['torque']
                ]
            
            if 'pressure_array' in reading:
                noise = np.random.normal(0, 0.001, reading['pressure_array'].shape)
                noisy_reading['pressure_array'] = reading['pressure_array'] + noise
                noisy_reading['total_force'] = np.sum(noisy_reading['pressure_array'])
            
            noisy_reading['timestamp'] = reading['timestamp'] + np.random.uniform(0, 0.001)
            readings[sensor_name] = noisy_reading
        
        return readings
    
    def simulate_contact(self, sensor_name: str, contact_force: List[float], contact_position: List[float] = None):
        """Simulate contact on a specific sensor."""
        if sensor_name not in self.tactile_readings:
            return
        
        sensor_config = self.config['tactile_sensors'][sensor_name]
        
        if sensor_config['sensor_type'] == 'force_torque':
            self.tactile_readings[sensor_name]['force'] = contact_force[:3]
            if len(contact_force) > 3:
                self.tactile_readings[sensor_name]['torque'] = contact_force[3:6]
        
        elif sensor_config['sensor_type'] == 'pressure_array':
            if contact_position and len(contact_position) >= 2:
                array_size = sensor_config['array_size']
                x_idx = int(contact_position[0] * array_size[0])
                y_idx = int(contact_position[1] * array_size[1])
                
                x_idx = np.clip(x_idx, 0, array_size[0] - 1)
                y_idx = np.clip(y_idx, 0, array_size[1] - 1)
                
                self.tactile_readings[sensor_name]['pressure_array'][y_idx, x_idx] = contact_force[0]
                self.tactile_readings[sensor_name]['center_of_pressure'] = [
                    contact_position[0], contact_position[1]
                ]
            
            self.tactile_readings[sensor_name]['total_force'] = sum(contact_force[:1])


def create_mock_gasm_environment(robot_name: str = 'franka_panda'):
    """Create complete mock GASM environment."""
    gasm_system = MockGASMRobotics()
    material_db = MockMaterialDatabase(gasm_system)
    
    robot_config = gasm_system.get_robot_config(robot_name)
    robot_controller = MockRobotController(robot_config) if robot_config else None
    
    return {
        'gasm_system': gasm_system,
        'material_database': material_db,
        'robot_controller': robot_controller,
        'simulation_config': gasm_system.simulation_config
    }