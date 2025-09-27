#!/usr/bin/env python3
"""
URDF Validation Script
Validates URDF files for correctness and PyBullet compatibility.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import math
import yaml
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


class URDFValidator:
    """Validates URDF files for structural correctness and physics compatibility."""
    
    def __init__(self, config_path: str = None):
        """Initialize validator with configuration."""
        self.config_path = config_path or "assets/configs/simulation_params.yaml"
        self.config = self.load_config()
        self.errors = []
        self.warnings = []
        self.validation_settings = self.config.get('validation', {})
        
    def load_config(self) -> Dict[str, Any]:
        """Load validation configuration."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}
    
    def reset_results(self):
        """Reset validation results."""
        self.errors.clear()
        self.warnings.clear()
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(f"ERROR: {message}")
        
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(f"WARNING: {message}")
    
    def validate_file(self, urdf_path: str) -> bool:
        """Validate a URDF file and return True if valid."""
        self.reset_results()
        
        if not os.path.exists(urdf_path):
            self.add_error(f"URDF file not found: {urdf_path}")
            return False
        
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except ET.ParseError as e:
            self.add_error(f"XML parsing error: {e}")
            return False
        
        if root.tag != 'robot':
            self.add_error("Root element must be 'robot'")
            return False
        
        robot_name = root.get('name')
        if not robot_name:
            self.add_error("Robot must have a 'name' attribute")
            return False
        
        # Validate structure
        self.validate_links(root)
        self.validate_joints(root)
        self.validate_materials(root)
        self.validate_physics_properties(root)
        self.validate_geometry(root)
        self.validate_joint_tree_structure(root)
        
        return len(self.errors) == 0
    
    def validate_links(self, root: ET.Element):
        """Validate all links in the URDF."""
        links = root.findall('link')
        
        if not links:
            self.add_error("URDF must contain at least one link")
            return
        
        link_names = set()
        
        for link in links:
            name = link.get('name')
            if not name:
                self.add_error("Link missing 'name' attribute")
                continue
            
            if name in link_names:
                self.add_error(f"Duplicate link name: {name}")
            link_names.add(name)
            
            self.validate_link_structure(link, name)
    
    def validate_link_structure(self, link: ET.Element, link_name: str):
        """Validate individual link structure."""
        # Check for required elements
        visual = link.find('visual')
        collision = link.find('collision')
        inertial = link.find('inertial')
        
        if visual is None:
            self.add_warning(f"Link '{link_name}' missing visual element")
        
        if collision is None:
            self.add_warning(f"Link '{link_name}' missing collision element")
        
        if inertial is None:
            self.add_warning(f"Link '{link_name}' missing inertial properties")
        else:
            self.validate_inertial_element(inertial, link_name)
    
    def validate_inertial_element(self, inertial: ET.Element, link_name: str):
        """Validate inertial properties of a link."""
        mass_elem = inertial.find('mass')
        inertia_elem = inertial.find('inertia')
        
        if mass_elem is None:
            self.add_error(f"Link '{link_name}' missing mass element")
            return
        
        try:
            mass = float(mass_elem.get('value', 0))
            mass_limits = self.validation_settings.get('mass_limits', {})
            min_mass = mass_limits.get('minimum', 0.001)
            max_mass = mass_limits.get('maximum', 10000)
            
            if mass <= 0:
                self.add_error(f"Link '{link_name}' has non-positive mass: {mass}")
            elif mass < min_mass:
                self.add_warning(f"Link '{link_name}' has very small mass: {mass}")
            elif mass > max_mass:
                self.add_warning(f"Link '{link_name}' has very large mass: {mass}")
        except (ValueError, TypeError):
            self.add_error(f"Link '{link_name}' has invalid mass value")
        
        if inertia_elem is None:
            self.add_error(f"Link '{link_name}' missing inertia element")
            return
        
        # Validate inertia matrix
        inertia_attrs = ['ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz']
        inertia_values = {}
        
        for attr in inertia_attrs:
            value_str = inertia_elem.get(attr, '0')
            try:
                inertia_values[attr] = float(value_str)
            except (ValueError, TypeError):
                self.add_error(f"Link '{link_name}' has invalid inertia {attr}: {value_str}")
        
        if len(inertia_values) == 6:
            self.validate_inertia_matrix(inertia_values, link_name)
    
    def validate_inertia_matrix(self, inertia: Dict[str, float], link_name: str):
        """Validate that inertia tensor is physically valid."""
        ixx, iyy, izz = inertia['ixx'], inertia['iyy'], inertia['izz']
        ixy, ixz, iyz = inertia['ixy'], inertia['ixz'], inertia['iyz']
        
        # Check diagonal elements are positive
        if ixx <= 0 or iyy <= 0 or izz <= 0:
            self.add_error(f"Link '{link_name}' has non-positive diagonal inertia elements")
        
        # Check triangle inequality for inertia tensor
        if ixx + iyy <= izz or iyy + izz <= ixx or izz + ixx <= iyy:
            self.add_warning(f"Link '{link_name}' inertia tensor may violate triangle inequality")
        
        # Create inertia matrix and check positive definiteness
        I = np.array([[ixx, ixy, ixz],
                     [ixy, iyy, iyz],
                     [ixz, iyz, izz]])
        
        try:
            eigenvalues = np.linalg.eigvals(I)
            if np.any(eigenvalues <= 0):
                self.add_error(f"Link '{link_name}' has non-positive definite inertia tensor")
        except np.linalg.LinAlgError:
            self.add_error(f"Link '{link_name}' has singular inertia tensor")
    
    def validate_joints(self, root: ET.Element):
        """Validate all joints in the URDF."""
        joints = root.findall('joint')
        joint_names = set()
        
        for joint in joints:
            name = joint.get('name')
            if not name:
                self.add_error("Joint missing 'name' attribute")
                continue
            
            if name in joint_names:
                self.add_error(f"Duplicate joint name: {name}")
            joint_names.add(name)
            
            self.validate_joint_structure(joint, name)
    
    def validate_joint_structure(self, joint: ET.Element, joint_name: str):
        """Validate individual joint structure."""
        joint_type = joint.get('type')
        if not joint_type:
            self.add_error(f"Joint '{joint_name}' missing 'type' attribute")
            return
        
        valid_types = ['fixed', 'revolute', 'prismatic', 'continuous', 'floating', 'planar']
        if joint_type not in valid_types:
            self.add_error(f"Joint '{joint_name}' has invalid type: {joint_type}")
        
        # Check required child elements
        parent = joint.find('parent')
        child = joint.find('child')
        
        if parent is None:
            self.add_error(f"Joint '{joint_name}' missing parent link")
        elif not parent.get('link'):
            self.add_error(f"Joint '{joint_name}' parent missing 'link' attribute")
        
        if child is None:
            self.add_error(f"Joint '{joint_name}' missing child link")
        elif not child.get('link'):
            self.add_error(f"Joint '{joint_name}' child missing 'link' attribute")
        
        # Validate joint limits for revolute and prismatic joints
        if joint_type in ['revolute', 'prismatic']:
            limit = joint.find('limit')
            if limit is None:
                self.add_error(f"Joint '{joint_name}' of type '{joint_type}' missing limit element")
            else:
                self.validate_joint_limits(limit, joint_name, joint_type)
        
        # Validate axis for revolute, prismatic, and continuous joints
        if joint_type in ['revolute', 'prismatic', 'continuous']:
            axis = joint.find('axis')
            if axis is None:
                self.add_warning(f"Joint '{joint_name}' missing axis (will default to [1 0 0])")
            else:
                xyz = axis.get('xyz', '1 0 0')
                try:
                    axis_vec = [float(x) for x in xyz.split()]
                    if len(axis_vec) != 3:
                        self.add_error(f"Joint '{joint_name}' axis must have 3 components")
                    elif all(abs(x) < 1e-6 for x in axis_vec):
                        self.add_error(f"Joint '{joint_name}' axis cannot be zero vector")
                except ValueError:
                    self.add_error(f"Joint '{joint_name}' has invalid axis values: {xyz}")
    
    def validate_joint_limits(self, limit: ET.Element, joint_name: str, joint_type: str):
        """Validate joint limit parameters."""
        required_attrs = ['lower', 'upper', 'effort', 'velocity']
        
        for attr in required_attrs:
            value_str = limit.get(attr)
            if not value_str:
                self.add_error(f"Joint '{joint_name}' limit missing '{attr}' attribute")
                continue
            
            try:
                value = float(value_str)
                if attr == 'effort' and value <= 0:
                    self.add_warning(f"Joint '{joint_name}' has non-positive effort limit: {value}")
                elif attr == 'velocity' and value <= 0:
                    self.add_warning(f"Joint '{joint_name}' has non-positive velocity limit: {value}")
            except ValueError:
                self.add_error(f"Joint '{joint_name}' has invalid {attr} value: {value_str}")
        
        # Check that lower <= upper for position limits
        try:
            lower = float(limit.get('lower', 0))
            upper = float(limit.get('upper', 0))
            if lower > upper:
                self.add_error(f"Joint '{joint_name}' lower limit ({lower}) > upper limit ({upper})")
        except (ValueError, TypeError):
            pass  # Already reported above
    
    def validate_materials(self, root: ET.Element):
        """Validate material definitions."""
        materials = root.findall('material')
        material_names = set()
        
        for material in materials:
            name = material.get('name')
            if not name:
                self.add_error("Material missing 'name' attribute")
                continue
            
            if name in material_names:
                self.add_warning(f"Duplicate material name: {name}")
            material_names.add(name)
            
            # Check color definition
            color = material.find('color')
            texture = material.find('texture')
            
            if color is None and texture is None:
                self.add_warning(f"Material '{name}' has neither color nor texture")
            
            if color is not None:
                rgba = color.get('rgba')
                if rgba:
                    try:
                        color_values = [float(x) for x in rgba.split()]
                        if len(color_values) != 4:
                            self.add_error(f"Material '{name}' color must have 4 RGBA values")
                        elif any(v < 0 or v > 1 for v in color_values):
                            self.add_warning(f"Material '{name}' color values should be between 0 and 1")
                    except ValueError:
                        self.add_error(f"Material '{name}' has invalid color values: {rgba}")
            
            if texture is not None:
                filename = texture.get('filename')
                if filename and self.validation_settings.get('warn_on_missing_textures', True):
                    # Check if texture file exists (relative to URDF location)
                    if not filename.startswith('package://'):
                        self.add_warning(f"Texture file may not exist: {filename}")
    
    def validate_physics_properties(self, root: ET.Element):
        """Validate physics-related properties."""
        links = root.findall('link')
        
        for link in links:
            name = link.get('name')
            collision = link.find('collision')
            
            if collision is not None:
                # Check surface properties
                surface = collision.find('surface')
                if surface is not None:
                    friction = surface.find('friction')
                    if friction is not None:
                        ode = friction.find('ode')
                        if ode is not None:
                            mu = ode.get('mu')
                            mu2 = ode.get('mu2')
                            
                            if mu:
                                try:
                                    mu_val = float(mu)
                                    if mu_val < 0:
                                        self.add_warning(f"Link '{name}' has negative friction coefficient: {mu_val}")
                                except ValueError:
                                    self.add_error(f"Link '{name}' has invalid friction mu: {mu}")
                            
                            if mu2:
                                try:
                                    mu2_val = float(mu2)
                                    if mu2_val < 0:
                                        self.add_warning(f"Link '{name}' has negative friction coefficient mu2: {mu2_val}")
                                except ValueError:
                                    self.add_error(f"Link '{name}' has invalid friction mu2: {mu2}")
    
    def validate_geometry(self, root: ET.Element):
        """Validate geometry definitions."""
        links = root.findall('link')
        dimension_limits = self.validation_settings.get('dimension_limits', {})
        min_dim = dimension_limits.get('minimum', 0.001)
        max_dim = dimension_limits.get('maximum', 100)
        
        for link in links:
            name = link.get('name')
            
            # Check visual geometry
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    self.validate_geometry_element(geometry, f"Link '{name}' visual", min_dim, max_dim)
            
            # Check collision geometry
            collision = link.find('collision')
            if collision is not None:
                geometry = collision.find('geometry')
                if geometry is not None:
                    self.validate_geometry_element(geometry, f"Link '{name}' collision", min_dim, max_dim)
    
    def validate_geometry_element(self, geometry: ET.Element, context: str, min_dim: float, max_dim: float):
        """Validate individual geometry element."""
        # Check box geometry
        box = geometry.find('box')
        if box is not None:
            size = box.get('size')
            if size:
                try:
                    dimensions = [float(x) for x in size.split()]
                    if len(dimensions) != 3:
                        self.add_error(f"{context} box must have 3 size dimensions")
                    elif any(d <= 0 for d in dimensions):
                        self.add_error(f"{context} box dimensions must be positive: {dimensions}")
                    elif any(d < min_dim for d in dimensions):
                        self.add_warning(f"{context} box has very small dimensions: {dimensions}")
                    elif any(d > max_dim for d in dimensions):
                        self.add_warning(f"{context} box has very large dimensions: {dimensions}")
                except ValueError:
                    self.add_error(f"{context} box has invalid size values: {size}")
        
        # Check cylinder geometry
        cylinder = geometry.find('cylinder')
        if cylinder is not None:
            radius = cylinder.get('radius')
            length = cylinder.get('length')
            
            if radius:
                try:
                    r = float(radius)
                    if r <= 0:
                        self.add_error(f"{context} cylinder radius must be positive: {r}")
                    elif r < min_dim:
                        self.add_warning(f"{context} cylinder has very small radius: {r}")
                    elif r > max_dim:
                        self.add_warning(f"{context} cylinder has very large radius: {r}")
                except ValueError:
                    self.add_error(f"{context} cylinder has invalid radius: {radius}")
            
            if length:
                try:
                    l = float(length)
                    if l <= 0:
                        self.add_error(f"{context} cylinder length must be positive: {l}")
                    elif l < min_dim:
                        self.add_warning(f"{context} cylinder has very small length: {l}")
                    elif l > max_dim:
                        self.add_warning(f"{context} cylinder has very large length: {l}")
                except ValueError:
                    self.add_error(f"{context} cylinder has invalid length: {length}")
        
        # Check sphere geometry
        sphere = geometry.find('sphere')
        if sphere is not None:
            radius = sphere.get('radius')
            if radius:
                try:
                    r = float(radius)
                    if r <= 0:
                        self.add_error(f"{context} sphere radius must be positive: {r}")
                    elif r < min_dim:
                        self.add_warning(f"{context} sphere has very small radius: {r}")
                    elif r > max_dim:
                        self.add_warning(f"{context} sphere has very large radius: {r}")
                except ValueError:
                    self.add_error(f"{context} sphere has invalid radius: {radius}")
        
        # Check mesh geometry
        mesh = geometry.find('mesh')
        if mesh is not None:
            filename = mesh.get('filename')
            if not filename:
                self.add_error(f"{context} mesh missing filename attribute")
            elif not filename.startswith('package://'):
                self.add_warning(f"{context} mesh filename should use package:// format: {filename}")
    
    def validate_joint_tree_structure(self, root: ET.Element):
        """Validate that joints form a proper tree structure."""
        links = {link.get('name') for link in root.findall('link')}
        joints = root.findall('joint')
        
        # Build parent-child relationships
        children = set()
        parents = set()
        parent_child_map = {}
        
        for joint in joints:
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')
            
            if parent_elem is not None and child_elem is not None:
                parent_name = parent_elem.get('link')
                child_name = child_elem.get('link')
                
                if parent_name and child_name:
                    if parent_name not in links:
                        self.add_error(f"Joint '{joint.get('name')}' references unknown parent link: {parent_name}")
                    if child_name not in links:
                        self.add_error(f"Joint '{joint.get('name')}' references unknown child link: {child_name}")
                    
                    if child_name in children:
                        self.add_error(f"Link '{child_name}' has multiple parents (not a tree structure)")
                    
                    children.add(child_name)
                    parents.add(parent_name)
                    parent_child_map[child_name] = parent_name
        
        # Find root links (links that are parents but not children)
        root_links = parents - children
        
        if len(root_links) == 0:
            self.add_error("No root link found - joint structure may contain cycles")
        elif len(root_links) > 1:
            self.add_warning(f"Multiple root links found: {root_links}")
        
        # Check for unreachable links
        all_connected = parents | children
        disconnected = links - all_connected
        
        if disconnected:
            self.add_warning(f"Links not connected by any joints: {disconnected}")
    
    def print_results(self):
        """Print validation results."""
        if self.errors:
            print("VALIDATION ERRORS:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print("VALIDATION WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.errors and not self.warnings:
            print("VALIDATION PASSED: No issues found")
        elif not self.errors:
            print(f"VALIDATION PASSED: No errors, {len(self.warnings)} warnings")
        else:
            print(f"VALIDATION FAILED: {len(self.errors)} errors, {len(self.warnings)} warnings")
    
    def validate_pybullet_compatibility(self, urdf_path: str) -> bool:
        """Test if URDF can be loaded by PyBullet."""
        try:
            import pybullet as p
            
            # Create temporary physics client
            physics_client = p.connect(p.DIRECT)
            
            try:
                # Try to load URDF
                robot_id = p.loadURDF(urdf_path)
                
                # Get basic info
                num_joints = p.getNumJoints(robot_id)
                base_info = p.getBasePositionAndOrientation(robot_id)
                
                print(f"PyBullet compatibility: PASSED")
                print(f"  Robot loaded with {num_joints} joints")
                print(f"  Base position: {base_info[0]}")
                
                return True
                
            except Exception as e:
                self.add_error(f"PyBullet loading failed: {e}")
                return False
                
            finally:
                p.disconnect(physics_client)
                
        except ImportError:
            self.add_warning("PyBullet not available for compatibility testing")
            return True


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Validate URDF files')
    parser.add_argument('urdf_files', nargs='+', help='URDF files to validate')
    parser.add_argument('--config', help='Path to configuration YAML file')
    parser.add_argument('--pybullet-test', action='store_true', 
                       help='Test PyBullet compatibility')
    parser.add_argument('--quiet', action='store_true',
                       help='Only show errors and warnings count')
    
    args = parser.parse_args()
    
    validator = URDFValidator(args.config)
    
    total_files = len(args.urdf_files)
    passed_files = 0
    
    for urdf_file in args.urdf_files:
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"Validating: {urdf_file}")
            print(f"{'='*60}")
        
        is_valid = validator.validate_file(urdf_file)
        
        if args.pybullet_test:
            pybullet_ok = validator.validate_pybullet_compatibility(urdf_file)
            is_valid = is_valid and pybullet_ok
        
        if not args.quiet:
            validator.print_results()
        
        if is_valid:
            passed_files += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed_files}/{total_files} files passed validation")
    
    if passed_files == total_files:
        print("All URDF files are valid!")
        sys.exit(0)
    else:
        print(f"{total_files - passed_files} files failed validation")
        sys.exit(1)


if __name__ == '__main__':
    main()