#!/usr/bin/env python3
"""
Procedural URDF Asset Generation System
Generates URDF files with configurable parameters for various robotic objects.
"""

import os
import sys
import argparse
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple, Optional, Any
import math
import json


class URDFGenerator:
    """Main class for generating URDF files procedurally."""
    
    def __init__(self, config_path: str = None):
        """Initialize URDF generator with configuration."""
        self.config_path = config_path or "assets/configs/simulation_params.yaml"
        self.config = self.load_config()
        self.materials = self.config.get('materials', {})
        self.physics = self.config.get('physics', {})
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file doesn't exist."""
        return {
            'materials': {
                'steel': {'color': [0.7, 0.7, 0.7, 1.0], 'friction': 0.8},
                'plastic': {'color': [0.2, 0.2, 0.8, 1.0], 'friction': 0.6},
                'rubber': {'color': [0.1, 0.1, 0.1, 1.0], 'friction': 1.2},
                'glass': {'color': [0.9, 0.9, 0.9, 0.3], 'friction': 0.1},
            },
            'physics': {
                'gravity': [0, 0, -9.81],
                'time_step': 0.001,
                'solver_iterations': 10,
            },
            'variants': {
                'sizes': ['small', 'medium', 'large'],
                'colors': ['red', 'green', 'blue', 'yellow', 'white'],
            }
        }
    
    def calculate_inertia_box(self, mass: float, dimensions: Tuple[float, float, float]) -> Dict[str, float]:
        """Calculate inertia tensor for a box."""
        x, y, z = dimensions
        ixx = (mass / 12.0) * (y*y + z*z)
        iyy = (mass / 12.0) * (x*x + z*z)
        izz = (mass / 12.0) * (x*x + y*y)
        return {'ixx': ixx, 'iyy': iyy, 'izz': izz, 'ixy': 0, 'ixz': 0, 'iyz': 0}
    
    def calculate_inertia_cylinder(self, mass: float, radius: float, height: float) -> Dict[str, float]:
        """Calculate inertia tensor for a cylinder."""
        ixx = iyy = (mass / 12.0) * (3 * radius*radius + height*height)
        izz = (mass / 2.0) * radius*radius
        return {'ixx': ixx, 'iyy': iyy, 'izz': izz, 'ixy': 0, 'ixz': 0, 'iyz': 0}
    
    def calculate_inertia_sphere(self, mass: float, radius: float) -> Dict[str, float]:
        """Calculate inertia tensor for a sphere."""
        inertia = (2.0 / 5.0) * mass * radius*radius
        return {'ixx': inertia, 'iyy': inertia, 'izz': inertia, 'ixy': 0, 'ixz': 0, 'iyz': 0}
    
    def create_link(self, name: str, mass: float, inertia: Dict[str, float], 
                   visual_geometry: Dict[str, Any], collision_geometry: Dict[str, Any] = None,
                   material: str = 'steel', origin: Tuple[float, float, float] = (0, 0, 0)) -> ET.Element:
        """Create a URDF link element."""
        link = ET.Element('link', name=name)
        
        # Inertial properties
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=f"{origin[0]} {origin[1]} {origin[2]}", rpy="0 0 0")
        ET.SubElement(inertial, 'mass', value=str(mass))
        inertia_elem = ET.SubElement(inertial, 'inertia')
        for key, value in inertia.items():
            inertia_elem.set(key, f"{value:.6f}")
        
        # Visual properties
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(visual, 'origin', xyz=f"{origin[0]} {origin[1]} {origin[2]}", rpy="0 0 0")
        visual_geom = ET.SubElement(visual, 'geometry')
        self.add_geometry(visual_geom, visual_geometry)
        
        # Material
        material_elem = ET.SubElement(visual, 'material', name=material)
        if material in self.materials:
            color = self.materials[material]['color']
            ET.SubElement(material_elem, 'color', rgba=f"{color[0]} {color[1]} {color[2]} {color[3]}")
        
        # Collision properties
        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', xyz=f"{origin[0]} {origin[1]} {origin[2]}", rpy="0 0 0")
        collision_geom = ET.SubElement(collision, 'geometry')
        collision_geometry = collision_geometry or visual_geometry
        self.add_geometry(collision_geom, collision_geometry)
        
        # Surface properties
        if material in self.materials and 'friction' in self.materials[material]:
            surface = ET.SubElement(collision, 'surface')
            friction_elem = ET.SubElement(surface, 'friction')
            ode = ET.SubElement(friction_elem, 'ode')
            friction_val = self.materials[material]['friction']
            ode.set('mu', str(friction_val))
            ode.set('mu2', str(friction_val))
        
        return link
    
    def add_geometry(self, parent: ET.Element, geometry: Dict[str, Any]):
        """Add geometry to a parent element."""
        geom_type = geometry['type']
        
        if geom_type == 'box':
            size = geometry['size']
            ET.SubElement(parent, 'box', size=f"{size[0]} {size[1]} {size[2]}")
        elif geom_type == 'cylinder':
            ET.SubElement(parent, 'cylinder', radius=str(geometry['radius']), length=str(geometry['length']))
        elif geom_type == 'sphere':
            ET.SubElement(parent, 'sphere', radius=str(geometry['radius']))
        elif geom_type == 'mesh':
            ET.SubElement(parent, 'mesh', filename=geometry['filename'], scale=geometry.get('scale', '1 1 1'))
    
    def create_joint(self, name: str, joint_type: str, parent: str, child: str,
                    origin: Tuple[float, float, float] = (0, 0, 0),
                    rpy: Tuple[float, float, float] = (0, 0, 0),
                    axis: Tuple[float, float, float] = None,
                    limits: Dict[str, float] = None) -> ET.Element:
        """Create a URDF joint element."""
        joint = ET.Element('joint', name=name, type=joint_type)
        
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', 
                     xyz=f"{origin[0]} {origin[1]} {origin[2]}", 
                     rpy=f"{rpy[0]} {rpy[1]} {rpy[2]}")
        
        if axis:
            ET.SubElement(joint, 'axis', xyz=f"{axis[0]} {axis[1]} {axis[2]}")
        
        if limits and joint_type in ['revolute', 'prismatic']:
            limit_elem = ET.SubElement(joint, 'limit')
            for key, value in limits.items():
                limit_elem.set(key, str(value))
            
            # Add dynamics
            dynamics = ET.SubElement(joint, 'dynamics')
            dynamics.set('damping', str(limits.get('damping', 0.1)))
            dynamics.set('friction', str(limits.get('friction', 0.01)))
        
        return joint
    
    def generate_conveyor(self, length: float = 2.0, width: float = 0.5, height: float = 0.1,
                         belt_speed: float = 0.1, variant: str = 'standard') -> ET.Element:
        """Generate a conveyor belt URDF."""
        robot = ET.Element('robot', name=f'conveyor_{variant}')
        
        # Base frame
        base_mass = 50.0 * (length * width * height / (2.0 * 0.5 * 0.1))  # Scale mass with size
        base_inertia = self.calculate_inertia_box(base_mass, (length, width, height))
        base_geometry = {'type': 'box', 'size': (length, width, height)}
        
        base_link = self.create_link('base_link', base_mass, base_inertia, 
                                   base_geometry, material='steel', origin=(0, 0, height/2))
        robot.append(base_link)
        
        # Belt surface
        belt_mass = 10.0 * (length * width / (2.0 * 0.5))
        belt_thickness = 0.02
        belt_inertia = self.calculate_inertia_box(belt_mass, (length, width, belt_thickness))
        belt_geometry = {'type': 'box', 'size': (length, width, belt_thickness)}
        
        belt_link = self.create_link('belt_surface', belt_mass, belt_inertia,
                                   belt_geometry, material='rubber')
        robot.append(belt_link)
        
        # Joint between base and belt
        belt_joint = self.create_joint('base_to_belt', 'fixed', 'base_link', 'belt_surface',
                                     origin=(0, 0, height + belt_thickness/2))
        robot.append(belt_joint)
        
        # Support legs
        leg_positions = [(length*0.4, width*0.3), (length*0.4, -width*0.3), 
                        (-length*0.4, width*0.3), (-length*0.4, -width*0.3)]
        
        for i, (x, y) in enumerate(leg_positions):
            leg_name = f'leg_{i+1}'
            leg_height = 0.8
            leg_radius = 0.05
            leg_mass = 5.0
            
            leg_inertia = self.calculate_inertia_cylinder(leg_mass, leg_radius, leg_height)
            leg_geometry = {'type': 'cylinder', 'radius': leg_radius, 'length': leg_height}
            
            leg_link = self.create_link(leg_name, leg_mass, leg_inertia,
                                      leg_geometry, material='steel', origin=(0, 0, -leg_height/2))
            robot.append(leg_link)
            
            leg_joint = self.create_joint(f'base_to_{leg_name}', 'fixed', 'base_link', leg_name,
                                        origin=(x, y, 0))
            robot.append(leg_joint)
        
        return robot
    
    def generate_sensor(self, sensor_type: str = 'camera', variant: str = 'standard',
                       movable: bool = True) -> ET.Element:
        """Generate a sensor URDF."""
        robot = ET.Element('robot', name=f'{sensor_type}_sensor_{variant}')
        
        # Sensor housing
        housing_size = (0.2, 0.2, 0.15)
        housing_mass = 2.0
        housing_inertia = self.calculate_inertia_box(housing_mass, housing_size)
        housing_geometry = {'type': 'box', 'size': housing_size}
        
        housing_link = self.create_link('sensor_base', housing_mass, housing_inertia,
                                      housing_geometry, material='plastic')
        robot.append(housing_link)
        
        # Sensor lens
        lens_radius = 0.04
        lens_length = 0.06
        lens_mass = 0.5
        lens_inertia = self.calculate_inertia_cylinder(lens_mass, lens_radius, lens_length)
        lens_geometry = {'type': 'cylinder', 'radius': lens_radius, 'length': lens_length}
        
        lens_link = self.create_link('sensor_lens', lens_mass, lens_inertia,
                                   lens_geometry, material='glass')
        robot.append(lens_link)
        
        if movable:
            # Mounting bracket
            bracket_size = (0.25, 0.05, 0.3)
            bracket_mass = 1.0
            bracket_inertia = self.calculate_inertia_box(bracket_mass, bracket_size)
            bracket_geometry = {'type': 'box', 'size': bracket_size}
            
            bracket_link = self.create_link('mounting_bracket', bracket_mass, bracket_inertia,
                                          bracket_geometry, material='steel')
            robot.append(bracket_link)
            
            # Pan joint (rotation around Z-axis)
            pan_joint = self.create_joint('bracket_to_sensor', 'revolute', 'mounting_bracket', 'sensor_base',
                                        origin=(0, 0, 0.1), axis=(0, 0, 1),
                                        limits={'lower': -math.pi, 'upper': math.pi, 
                                               'effort': 10.0, 'velocity': 1.0, 'damping': 0.1})
            robot.append(pan_joint)
            
            # Tilt joint (rotation around X-axis)
            tilt_joint = self.create_joint('sensor_tilt', 'revolute', 'sensor_base', 'sensor_lens',
                                         origin=(0, 0.12, 0), rpy=(math.pi/2, 0, 0), axis=(1, 0, 0),
                                         limits={'lower': -math.pi/2, 'upper': math.pi/2,
                                                'effort': 5.0, 'velocity': 1.0, 'damping': 0.05})
            robot.append(tilt_joint)
        else:
            # Fixed lens attachment
            lens_joint = self.create_joint('base_to_lens', 'fixed', 'sensor_base', 'sensor_lens',
                                         origin=(0, 0.12, 0), rpy=(math.pi/2, 0, 0))
            robot.append(lens_joint)
        
        return robot
    
    def generate_object_variants(self, object_type: str, base_params: Dict[str, Any]) -> List[ET.Element]:
        """Generate multiple variants of an object with different sizes and colors."""
        variants = []
        sizes = self.config.get('variants', {}).get('sizes', ['small', 'medium', 'large'])
        
        size_multipliers = {'small': 0.7, 'medium': 1.0, 'large': 1.3}
        
        for size in sizes:
            multiplier = size_multipliers.get(size, 1.0)
            scaled_params = base_params.copy()
            
            # Scale dimensional parameters
            for key in ['length', 'width', 'height', 'radius']:
                if key in scaled_params:
                    scaled_params[key] *= multiplier
            
            # Scale mass proportionally to volume
            if 'mass' in scaled_params:
                scaled_params['mass'] *= multiplier ** 3
            
            scaled_params['variant'] = size
            
            if object_type == 'conveyor':
                variant = self.generate_conveyor(**scaled_params)
            elif object_type == 'sensor':
                variant = self.generate_sensor(**scaled_params)
            else:
                continue
                
            variants.append(variant)
        
        return variants
    
    def save_urdf(self, robot: ET.Element, filename: str, output_dir: str = "assets/urdf/"):
        """Save URDF to file with proper formatting."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # Format XML nicely
        rough_string = ET.tostring(robot, 'unicode')
        reparsed = minidom.parseString(rough_string)
        formatted = reparsed.toprettyxml(indent="    ")
        
        # Remove empty lines
        lines = [line for line in formatted.split('\n') if line.strip()]
        formatted = '\n'.join(lines)
        
        with open(filepath, 'w') as f:
            f.write(formatted)
        
        print(f"Generated URDF: {filepath}")
    
    def generate_all_assets(self):
        """Generate all standard assets."""
        print("Generating URDF assets...")
        
        # Generate standard conveyor
        conveyor = self.generate_conveyor()
        self.save_urdf(conveyor, 'conveyor.urdf')
        
        # Generate standard sensor
        sensor = self.generate_sensor()
        self.save_urdf(sensor, 'sensor.urdf')
        
        # Generate conveyor variants
        conveyor_variants = self.generate_object_variants('conveyor', {
            'length': 2.0, 'width': 0.5, 'height': 0.1, 'belt_speed': 0.1
        })
        sizes = ['small', 'medium', 'large']
        for i, variant in enumerate(conveyor_variants):
            if i < len(sizes):
                size = sizes[i]
                self.save_urdf(variant, f'conveyor_{size}.urdf')
        
        # Generate sensor variants
        sensor_variants = self.generate_object_variants('sensor', {
            'sensor_type': 'camera', 'movable': True
        })
        sizes = ['small', 'medium', 'large']
        for i, variant in enumerate(sensor_variants):
            if i < len(sizes):
                size = sizes[i]
                self.save_urdf(variant, f'sensor_{size}.urdf')
        
        # Generate fixed sensor
        fixed_sensor = self.generate_sensor(movable=False, variant='fixed')
        self.save_urdf(fixed_sensor, 'sensor_fixed.urdf')
        
        print("Asset generation complete!")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate URDF assets procedurally')
    parser.add_argument('--config', help='Path to configuration YAML file')
    parser.add_argument('--object', choices=['conveyor', 'sensor', 'all'], default='all',
                       help='Type of object to generate')
    parser.add_argument('--output', default='assets/urdf/', help='Output directory')
    parser.add_argument('--variants', action='store_true', help='Generate size variants')
    
    args = parser.parse_args()
    
    generator = URDFGenerator(args.config)
    
    if args.object == 'all':
        generator.generate_all_assets()
    elif args.object == 'conveyor':
        conveyor = generator.generate_conveyor()
        generator.save_urdf(conveyor, 'conveyor.urdf', args.output)
        if args.variants:
            variants = generator.generate_object_variants('conveyor', {
                'length': 2.0, 'width': 0.5, 'height': 0.1
            })
            for i, variant in enumerate(variants):
                size = ['small', 'medium', 'large'][i]
                generator.save_urdf(variant, f'conveyor_{size}.urdf', args.output)
    elif args.object == 'sensor':
        sensor = generator.generate_sensor()
        generator.save_urdf(sensor, 'sensor.urdf', args.output)
        if args.variants:
            variants = generator.generate_object_variants('sensor', {
                'sensor_type': 'camera', 'movable': True
            })
            for i, variant in enumerate(variants):
                size = ['small', 'medium', 'large'][i]
                generator.save_urdf(variant, f'sensor_{size}.urdf', args.output)


if __name__ == '__main__':
    main()