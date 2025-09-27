"""
Tests for Action Filter System
"""

import pytest
import numpy as np
from datetime import datetime
from src.olympus.safety_layer.action_filter import (
    ActionFilter, FilterResult, FilterStatus, FilterLayer,
    PhysicsLimits, SpatialLimits
)


class TestActionFilter:
    """Test cases for ActionFilter"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.physics_limits = PhysicsLimits(
            max_force=20.0,
            max_speed=1.0,
            max_acceleration=2.0,
            max_jerk=10.0,
            max_torque=5.0
        )
        
        self.spatial_limits = SpatialLimits(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
            min_obstacle_distance=0.1,
            max_reach_distance=1.5
        )
        
        self.filter = ActionFilter(
            physics_limits=self.physics_limits,
            spatial_limits=self.spatial_limits,
            strict_mode=True
        )
    
    def test_safe_action_approved(self):
        """Test that safe actions are approved"""
        safe_action = {
            'force': [5.0, 0.0, 0.0],
            'velocity': [0.3, 0.0, 0.0],
            'target_position': [0.5, 0.5, 1.0],
            'current_position': [0.0, 0.0, 1.0]
        }
        
        result = self.filter.filter_action(safe_action)
        
        assert result.status == FilterStatus.APPROVED
        assert result.risk_score < 0.5
    
    def test_excessive_force_blocked(self):
        """Test that excessive force is blocked in strict mode"""
        dangerous_action = {
            'force': [25.0, 0.0, 0.0],  # Exceeds 20N limit
            'velocity': [0.3, 0.0, 0.0]
        }
        
        result = self.filter.filter_action(dangerous_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.PHYSICS
        assert "Force" in result.reason
    
    def test_excessive_speed_blocked(self):
        """Test that excessive speed is blocked in strict mode"""
        dangerous_action = {
            'force': [5.0, 0.0, 0.0],
            'velocity': [1.5, 0.0, 0.0]  # Exceeds 1.0 m/s limit
        }
        
        result = self.filter.filter_action(dangerous_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.PHYSICS
        assert "Speed" in result.reason
    
    def test_non_strict_mode_modification(self):
        """Test that non-strict mode modifies rather than blocks"""
        non_strict_filter = ActionFilter(
            physics_limits=self.physics_limits,
            spatial_limits=self.spatial_limits,
            strict_mode=False
        )
        
        dangerous_action = {
            'force': [25.0, 0.0, 0.0],  # Exceeds 20N limit
            'velocity': [1.5, 0.0, 0.0]  # Exceeds 1.0 m/s limit
        }
        
        result = non_strict_filter.filter_action(dangerous_action)
        
        assert result.status == FilterStatus.MODIFIED
        assert result.filtered_action is not None
        
        # Check that force was scaled down
        modified_force = np.linalg.norm(result.filtered_action['force'])
        assert modified_force <= self.physics_limits.max_force
        
        # Check that velocity was scaled down
        modified_speed = np.linalg.norm(result.filtered_action['velocity'])
        assert modified_speed <= self.physics_limits.max_speed
    
    def test_workspace_boundary_violation(self):
        """Test workspace boundary violations are blocked"""
        out_of_bounds_action = {
            'target_position': [2.0, 0.0, 1.0],  # Outside x bounds (-1.0, 1.0)
            'current_position': [0.0, 0.0, 1.0]
        }
        
        result = self.filter.filter_action(out_of_bounds_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.SPATIAL
        assert "workspace bounds" in result.reason
    
    def test_dangerous_tool_requires_confirmation(self):
        """Test that dangerous tools require confirmation"""
        dangerous_tool_action = {
            'tool': 'cutter',
            'force': [5.0, 0.0, 0.0],
            'velocity': [0.3, 0.0, 0.0]
        }
        
        result = self.filter.filter_action(dangerous_tool_action)
        
        assert result.status == FilterStatus.REQUIRES_CONFIRMATION
        assert result.layer == FilterLayer.INTENTION
        assert "Dangerous tool" in result.reason
    
    def test_human_proximity_blocks_action(self):
        """Test that human proximity blocks actions"""
        human_proximity_action = {
            'force': [5.0, 0.0, 0.0],
            'velocity': [0.3, 0.0, 0.0],
            'humans_detected': [
                {
                    'distance': 0.5,  # Below minimum safe distance of 1.0m
                    'min_safe_distance': 1.0
                }
            ]
        }
        
        result = self.filter.filter_action(human_proximity_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.HUMAN_SAFETY
        assert "minimum safe distance" in result.reason
    
    def test_human_warning_zone(self):
        """Test human warning zone requires confirmation"""
        warning_zone_action = {
            'force': [5.0, 0.0, 0.0],
            'velocity': [0.3, 0.0, 0.0],
            'humans_detected': [
                {
                    'distance': 1.2,  # In warning zone (1.5 * min_safe_distance)
                    'min_safe_distance': 1.0
                }
            ]
        }
        
        result = self.filter.filter_action(warning_zone_action)
        
        assert result.status == FilterStatus.REQUIRES_CONFIRMATION
        assert result.layer == FilterLayer.HUMAN_SAFETY
        assert "warning zone" in result.reason
    
    def test_environmental_hazards(self):
        """Test environmental hazard detection"""
        hazardous_action = {
            'force': [5.0, 0.0, 0.0],
            'environment': {
                'lighting': 10,  # Very poor lighting
                'temperature': 60,  # High temperature
                'hazardous_materials': True
            },
            'system_status': {
                'sensors_operational': False  # Critical sensor failure
            }
        }
        
        result = self.filter.filter_action(hazardous_action)
        
        # Should be blocked due to non-operational sensors
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.CONTEXT
    
    def test_obstacle_collision_risk(self):
        """Test obstacle collision detection"""
        collision_risk_action = {
            'trajectory': [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
            'obstacles': [
                {
                    'position': [0.15, 0.0, 1.0],
                    'radius': 0.05
                }
            ]
        }
        
        result = self.filter.filter_action(collision_risk_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.SPATIAL
        assert "clearance" in result.reason
    
    def test_multiple_filter_layers(self):
        """Test that actions pass through all filter layers"""
        complex_action = {
            'force': [10.0, 0.0, 0.0],
            'velocity': [0.5, 0.0, 0.0],
            'target_position': [0.5, 0.5, 1.0],
            'current_position': [0.0, 0.0, 1.0],
            'tool': 'gripper',
            'environment': {
                'lighting': 80,
                'temperature': 22
            }
        }
        
        result = self.filter.filter_action(complex_action)
        
        # Should be approved as it passes all checks
        assert result.status == FilterStatus.APPROVED
        assert result.layer == FilterLayer.HUMAN_SAFETY  # Final layer
    
    def test_filter_configuration_update(self):
        """Test updating filter configuration"""
        new_physics_limits = PhysicsLimits(max_force=15.0, max_speed=0.8)
        
        self.filter.update_limits(physics_limits=new_physics_limits)
        
        # Test with force that was previously acceptable
        action = {
            'force': [18.0, 0.0, 0.0]  # Would be OK with old limits, not with new
        }
        
        result = self.filter.filter_action(action)
        
        assert result.status == FilterStatus.BLOCKED
        assert "Force" in result.reason
    
    def test_filter_status_reporting(self):
        """Test filter status reporting"""
        status = self.filter.get_filter_status()
        
        assert 'strict_mode' in status
        assert 'physics_limits' in status
        assert 'spatial_limits' in status
        assert 'active_layers' in status
        
        assert status['strict_mode'] == True
        assert status['physics_limits']['max_force'] == 20.0
        assert len(status['active_layers']) == 5  # All filter layers


class TestFilterResult:
    """Test cases for FilterResult"""
    
    def test_filter_result_creation(self):
        """Test FilterResult creation and properties"""
        action = {'test': 'action'}
        
        result = FilterResult(
            status=FilterStatus.BLOCKED,
            layer=FilterLayer.PHYSICS,
            original_action=action,
            reason="Test reason",
            risk_score=0.8
        )
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.PHYSICS
        assert result.original_action == action
        assert result.reason == "Test reason"
        assert result.risk_score == 0.8
        assert isinstance(result.timestamp, datetime)


class TestPhysicsLimits:
    """Test cases for PhysicsLimits"""
    
    def test_default_limits(self):
        """Test default physics limits"""
        limits = PhysicsLimits()
        
        assert limits.max_force == 20.0
        assert limits.max_speed == 1.0
        assert limits.max_acceleration == 2.0
        assert limits.max_jerk == 10.0
        assert limits.max_torque == 5.0
    
    def test_custom_limits(self):
        """Test custom physics limits"""
        limits = PhysicsLimits(
            max_force=15.0,
            max_speed=0.5,
            max_acceleration=1.0
        )
        
        assert limits.max_force == 15.0
        assert limits.max_speed == 0.5
        assert limits.max_acceleration == 1.0


class TestSpatialLimits:
    """Test cases for SpatialLimits"""
    
    def test_workspace_bounds(self):
        """Test workspace boundary definitions"""
        limits = SpatialLimits(
            workspace_bounds=((-2.0, 2.0), (-1.5, 1.5), (0.0, 3.0))
        )
        
        assert limits.workspace_bounds[0] == (-2.0, 2.0)  # X bounds
        assert limits.workspace_bounds[1] == (-1.5, 1.5)  # Y bounds
        assert limits.workspace_bounds[2] == (0.0, 3.0)   # Z bounds
    
    def test_safety_distances(self):
        """Test safety distance parameters"""
        limits = SpatialLimits(
            min_obstacle_distance=0.2,
            max_reach_distance=2.0
        )
        
        assert limits.min_obstacle_distance == 0.2
        assert limits.max_reach_distance == 2.0


if __name__ == "__main__":
    pytest.main([__file__])