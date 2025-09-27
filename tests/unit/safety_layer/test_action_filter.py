"""
Comprehensive test suite for Safety Layer Action Filter

This test suite validates the multi-layer action filtering system that provides
the final safety checks before any action is executed. All filtering layers
must be thoroughly tested to ensure human safety.

Filter layers tested:
1. Physics validation (force, speed, acceleration limits)
2. Spatial validation (workspace boundaries, collision avoidance)  
3. Intention validation (analyzing action purpose and safety)
4. Context validation (environmental conditions)
5. Human safety validation (proximity and interaction safety)
"""

import numpy as np
import pytest
import time
from unittest.mock import Mock, patch

from olympus.safety_layer.action_filter import (
    ActionFilter,
    FilterLayer,
    FilterStatus,
    FilterResult,
    PhysicsLimits,
    SpatialLimits
)


@pytest.mark.safety
class TestActionFilterInitialization:
    """Test ActionFilter initialization and configuration."""
    
    def test_default_initialization(self):
        """Test filter initializes with default parameters."""
        filter_system = ActionFilter()
        
        assert filter_system.strict_mode is True
        assert filter_system.physics_limits is not None
        assert filter_system.spatial_limits is not None
        assert len(filter_system.filter_layers) == 5
        
    def test_custom_limits_initialization(self):
        """Test filter initialization with custom limits."""
        physics_limits = PhysicsLimits(
            max_force=15.0,
            max_speed=0.8,
            max_acceleration=1.5,
            max_jerk=8.0,
            max_torque=4.0
        )
        
        spatial_limits = SpatialLimits(
            workspace_bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
            min_obstacle_distance=0.2,
            max_reach_distance=1.0
        )
        
        filter_system = ActionFilter(
            physics_limits=physics_limits,
            spatial_limits=spatial_limits,
            strict_mode=False
        )
        
        assert filter_system.physics_limits.max_force == 15.0
        assert filter_system.spatial_limits.min_obstacle_distance == 0.2
        assert not filter_system.strict_mode
    
    def test_filter_status_reporting(self):
        """Test filter status reporting functionality."""
        filter_system = ActionFilter()
        
        status = filter_system.get_filter_status()
        
        required_keys = ['strict_mode', 'physics_limits', 'spatial_limits', 'active_layers']
        for key in required_keys:
            assert key in status
        
        assert status['strict_mode'] is True
        assert len(status['active_layers']) == 5


@pytest.mark.safety 
class TestPhysicsFilterLayer:
    """Test physics-based filtering (Layer 1)."""
    
    def test_force_limit_enforcement_strict_mode(self):
        """Test force limits are enforced in strict mode."""
        filter_system = ActionFilter(strict_mode=True)
        
        excessive_force_action = {
            'force': [25.0, 0.0, 0.0],  # Exceeds 20N limit
            'action_type': 'push'
        }
        
        result = filter_system.filter_action(excessive_force_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.PHYSICS
        assert "force" in result.reason.lower()
        assert result.risk_score > 0.5
    
    def test_force_limit_scaling_non_strict_mode(self):
        """Test force limits are scaled in non-strict mode."""
        filter_system = ActionFilter(strict_mode=False)
        
        excessive_force_action = {
            'force': [30.0, 0.0, 0.0],  # Exceeds 20N limit
            'action_type': 'push'
        }
        
        result = filter_system.filter_action(excessive_force_action)
        
        assert result.status == FilterStatus.MODIFIED
        assert result.layer == FilterLayer.PHYSICS
        assert result.filtered_action is not None
        
        # Force should be scaled down
        scaled_force = np.linalg.norm(result.filtered_action['force'])
        assert scaled_force <= filter_system.physics_limits.max_force
    
    def test_velocity_limit_enforcement(self):
        """Test velocity limits are enforced."""
        filter_system = ActionFilter(strict_mode=True)
        
        high_speed_action = {
            'velocity': [2.0, 0.0, 0.0],  # Exceeds 1.0 m/s limit
            'action_type': 'move'
        }
        
        result = filter_system.filter_action(high_speed_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.PHYSICS
        assert "speed" in result.reason.lower()
    
    def test_acceleration_limit_enforcement(self):
        """Test acceleration limits are enforced."""
        filter_system = ActionFilter(strict_mode=True)
        
        high_accel_action = {
            'acceleration': [5.0, 0.0, 0.0],  # Exceeds 2.0 m/s² limit
            'action_type': 'accelerate'
        }
        
        result = filter_system.filter_action(high_accel_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.PHYSICS
        assert "acceleration" in result.reason.lower()
    
    def test_safe_physics_parameters_pass(self):
        """Test actions with safe physics parameters pass."""
        filter_system = ActionFilter()
        
        safe_action = {
            'force': [10.0, 0.0, 0.0],       # Within 20N limit
            'velocity': [0.5, 0.0, 0.0],     # Within 1.0 m/s limit
            'acceleration': [1.0, 0.0, 0.0], # Within 2.0 m/s² limit
            'action_type': 'move'
        }
        
        # Test just physics layer by mocking other layers
        with patch.object(filter_system, '_spatial_filter') as mock_spatial, \
             patch.object(filter_system, '_intention_filter') as mock_intention, \
             patch.object(filter_system, '_context_filter') as mock_context, \
             patch.object(filter_system, '_human_safety_filter') as mock_human:
            
            # Mock other layers to approve
            mock_spatial.return_value = FilterResult(
                status=FilterStatus.APPROVED, layer=FilterLayer.SPATIAL, 
                original_action=safe_action, reason="Spatial OK"
            )
            mock_intention.return_value = FilterResult(
                status=FilterStatus.APPROVED, layer=FilterLayer.INTENTION,
                original_action=safe_action, reason="Intention OK"
            )
            mock_context.return_value = FilterResult(
                status=FilterStatus.APPROVED, layer=FilterLayer.CONTEXT,
                original_action=safe_action, reason="Context OK"
            )
            mock_human.return_value = FilterResult(
                status=FilterStatus.APPROVED, layer=FilterLayer.HUMAN_SAFETY,
                original_action=safe_action, reason="Human safety OK"
            )
            
            result = filter_system.filter_action(safe_action)
            
            assert result.status == FilterStatus.APPROVED


@pytest.mark.safety
class TestSpatialFilterLayer:
    """Test spatial boundary and collision filtering (Layer 2)."""
    
    def test_workspace_boundary_enforcement(self):
        """Test workspace boundaries are enforced."""
        spatial_limits = SpatialLimits(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
            min_obstacle_distance=0.1,
            max_reach_distance=1.5
        )
        filter_system = ActionFilter(spatial_limits=spatial_limits)
        
        out_of_bounds_action = {
            'target_position': [2.0, 0.0, 0.5],  # x=2.0 exceeds bounds [-1,1]
            'current_position': [0.0, 0.0, 0.5],
            'action_type': 'move'
        }
        
        result = filter_system._spatial_filter(out_of_bounds_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.SPATIAL
        assert "workspace bounds" in result.reason
        assert "x=" in result.reason
    
    def test_reach_distance_limit_enforcement(self):
        """Test maximum reach distance is enforced."""
        filter_system = ActionFilter()
        
        far_reach_action = {
            'target_position': [0.0, 0.0, 2.0],
            'current_position': [0.0, 0.0, 0.0],  # Distance = 2.0m > 1.5m limit
            'action_type': 'reach'
        }
        
        result = filter_system._spatial_filter(far_reach_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.SPATIAL
        assert "reach" in result.reason.lower()
    
    def test_obstacle_clearance_enforcement(self):
        """Test minimum obstacle clearance is enforced."""
        filter_system = ActionFilter()
        
        collision_action = {
            'trajectory': [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.2, 0.0, 0.5]],
            'obstacles': [
                {'position': [0.15, 0.0, 0.5], 'radius': 0.05}
            ],
            'action_type': 'move'
        }
        
        result = filter_system._spatial_filter(collision_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.SPATIAL
        assert "clearance" in result.reason.lower()
    
    def test_safe_spatial_parameters_pass(self):
        """Test actions with safe spatial parameters pass."""
        filter_system = ActionFilter()
        
        safe_spatial_action = {
            'target_position': [0.5, 0.3, 0.8],   # Within bounds
            'current_position': [0.0, 0.0, 0.5],  # Distance < 1.5m
            'action_type': 'move'
        }
        
        result = filter_system._spatial_filter(safe_spatial_action)
        
        assert result.status == FilterStatus.APPROVED
        assert result.layer == FilterLayer.SPATIAL


@pytest.mark.safety
class TestIntentionFilterLayer:
    """Test intention analysis filtering (Layer 3)."""
    
    def test_high_speed_intention_detection(self):
        """Test detection of high-speed movement intentions."""
        filter_system = ActionFilter()
        
        high_speed_action = {
            'velocity': [0.8, 0.0, 0.0],  # High speed but within physics limits
            'action_type': 'rapid_movement'
        }
        
        result = filter_system._intention_filter(high_speed_action)
        
        # Should detect risk but not block (unless very high)
        assert result.risk_score > 0.1
    
    def test_dangerous_tool_detection(self):
        """Test detection and handling of dangerous tools."""
        filter_system = ActionFilter()
        
        dangerous_tool_action = {
            'tool': 'plasma',  # Dangerous tool
            'action_type': 'cut'
        }
        
        result = filter_system._intention_filter(dangerous_tool_action)
        
        assert result.status == FilterStatus.REQUIRES_CONFIRMATION
        assert result.layer == FilterLayer.INTENTION
        assert "dangerous tool" in result.reason.lower()
        assert result.risk_score > 0.4
    
    def test_high_force_intention_detection(self):
        """Test detection of high-force operations."""
        filter_system = ActionFilter()
        
        high_force_action = {
            'force': [15.0, 0.0, 0.0],  # High force but within physics limits
            'action_type': 'press'
        }
        
        result = filter_system._intention_filter(high_force_action)
        
        assert result.risk_score > 0.2
        assert "high force" in result.reason.lower()
    
    def test_high_repetition_detection(self):
        """Test detection of high repetition operations."""
        filter_system = ActionFilter()
        
        repetitive_action = {
            'repetitions': 150,  # High repetition count
            'action_type': 'repeat'
        }
        
        result = filter_system._intention_filter(repetitive_action)
        
        assert result.risk_score > 0.1
        assert "repetition" in result.reason.lower()
    
    def test_low_risk_intentions_pass(self):
        """Test low-risk intentions pass without issues."""
        filter_system = ActionFilter()
        
        safe_action = {
            'velocity': [0.1, 0.0, 0.0],  # Low speed
            'force': [2.0, 0.0, 0.0],     # Low force
            'tool': 'gripper',             # Safe tool
            'repetitions': 1,              # Single operation
            'action_type': 'gentle_move'
        }
        
        result = filter_system._intention_filter(safe_action)
        
        assert result.status == FilterStatus.APPROVED
        assert result.risk_score <= 0.6


@pytest.mark.safety
class TestContextFilterLayer:
    """Test environmental context filtering (Layer 4)."""
    
    def test_poor_lighting_detection(self):
        """Test detection of poor lighting conditions."""
        filter_system = ActionFilter()
        
        poor_lighting_action = {
            'environment': {
                'lighting': 10,  # Too dark (< 20)
                'temperature': 20,
                'vibration_level': 1
            },
            'action_type': 'precision_work'
        }
        
        result = filter_system._context_filter(poor_lighting_action)
        
        assert result.risk_score > 0.2
        assert "lighting" in result.reason.lower()
    
    def test_extreme_temperature_detection(self):
        """Test detection of extreme temperatures."""
        filter_system = ActionFilter()
        
        extreme_temp_action = {
            'environment': {
                'lighting': 80,
                'temperature': 60,  # Too hot (> 50)
                'vibration_level': 1
            },
            'action_type': 'work'
        }
        
        result = filter_system._context_filter(extreme_temp_action)
        
        assert result.risk_score > 0.1
        assert "temperature" in result.reason.lower()
    
    def test_high_vibration_detection(self):
        """Test detection of high vibration levels."""
        filter_system = ActionFilter()
        
        high_vibration_action = {
            'environment': {
                'lighting': 80,
                'temperature': 20,
                'vibration_level': 8  # High vibration (> 5)
            },
            'action_type': 'precision_work'
        }
        
        result = filter_system._context_filter(high_vibration_action)
        
        assert result.risk_score > 0.1
        assert "vibration" in result.reason.lower()
    
    def test_hazardous_materials_detection(self):
        """Test detection of hazardous materials."""
        filter_system = ActionFilter()
        
        hazardous_action = {
            'environment': {
                'lighting': 80,
                'temperature': 20,
                'vibration_level': 1,
                'hazardous_materials': True
            },
            'action_type': 'work'
        }
        
        result = filter_system._context_filter(hazardous_action)
        
        assert result.risk_score > 0.3
        assert "hazardous materials" in result.reason.lower()
    
    def test_low_battery_detection(self):
        """Test detection of low battery conditions."""
        filter_system = ActionFilter()
        
        low_battery_action = {
            'system_status': {
                'battery_level': 15,  # Low battery (< 20)
                'error_count': 0,
                'sensors_operational': True
            },
            'action_type': 'work'
        }
        
        result = filter_system._context_filter(low_battery_action)
        
        assert result.risk_score > 0.2
        assert "battery" in result.reason.lower()
    
    def test_sensor_failure_blocking(self):
        """Test that sensor failures block operations."""
        filter_system = ActionFilter()
        
        sensor_failure_action = {
            'system_status': {
                'battery_level': 80,
                'error_count': 2,
                'sensors_operational': False  # Critical sensor failure
            },
            'action_type': 'work'
        }
        
        result = filter_system._context_filter(sensor_failure_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.CONTEXT
        assert "sensors" in result.reason.lower()
        assert result.risk_score == 1.0
    
    def test_multiple_system_errors(self):
        """Test handling of multiple system errors."""
        filter_system = ActionFilter()
        
        error_prone_action = {
            'system_status': {
                'battery_level': 80,
                'error_count': 8,  # Multiple errors (> 5)
                'sensors_operational': True
            },
            'action_type': 'work'
        }
        
        result = filter_system._context_filter(error_prone_action)
        
        assert result.risk_score > 0.3
        assert "error" in result.reason.lower()


@pytest.mark.safety
class TestHumanSafetyFilterLayer:
    """Test human safety filtering (Layer 5 - Final)."""
    
    def test_human_too_close_blocking(self):
        """Test that humans too close block operations."""
        filter_system = ActionFilter()
        
        human_too_close_action = {
            'humans_detected': [
                {
                    'distance': 0.5,  # Too close
                    'min_safe_distance': 1.0
                }
            ],
            'action_type': 'work'
        }
        
        result = filter_system._human_safety_filter(human_too_close_action)
        
        assert result.status == FilterStatus.BLOCKED
        assert result.layer == FilterLayer.HUMAN_SAFETY
        assert "safe distance" in result.reason.lower()
        assert result.risk_score == 1.0
    
    def test_human_in_warning_zone(self):
        """Test human in warning zone requires confirmation."""
        filter_system = ActionFilter()
        
        human_warning_zone_action = {
            'humans_detected': [
                {
                    'distance': 1.2,  # In warning zone (< 1.5 * min_safe_distance)
                    'min_safe_distance': 1.0
                }
            ],
            'action_type': 'work'
        }
        
        result = filter_system._human_safety_filter(human_warning_zone_action)
        
        assert result.status == FilterStatus.REQUIRES_CONFIRMATION
        assert result.layer == FilterLayer.HUMAN_SAFETY
        assert "warning zone" in result.reason.lower()
        assert result.risk_score > 0.5
    
    def test_direct_human_interaction_confirmation(self):
        """Test direct human interaction requires confirmation."""
        filter_system = ActionFilter()
        
        direct_interaction_action = {
            'interaction_type': 'direct_human',
            'action_type': 'handshake'
        }
        
        result = filter_system._human_safety_filter(direct_interaction_action)
        
        assert result.status == FilterStatus.REQUIRES_CONFIRMATION
        assert result.layer == FilterLayer.HUMAN_SAFETY
        assert "direct human interaction" in result.reason.lower()
        assert result.risk_score > 0.5
    
    def test_no_humans_detected_passes(self):
        """Test actions with no humans detected pass safely."""
        filter_system = ActionFilter()
        
        no_humans_action = {
            'humans_detected': [],
            'action_type': 'automated_work'
        }
        
        result = filter_system._human_safety_filter(no_humans_action)
        
        assert result.status == FilterStatus.APPROVED
        assert result.layer == FilterLayer.HUMAN_SAFETY


@pytest.mark.safety
class TestIntegratedFiltering:
    """Test complete integrated filtering through all layers."""
    
    def test_completely_safe_action_passes_all_layers(self, valid_action_dict):
        """Test that completely safe actions pass all filter layers."""
        filter_system = ActionFilter()
        
        result = filter_system.filter_action(valid_action_dict)
        
        assert result.status == FilterStatus.APPROVED
        assert result.layer == FilterLayer.HUMAN_SAFETY  # Last layer
        assert result.risk_score < 0.5
    
    def test_dangerous_action_blocked_early(self, dangerous_action_dict):
        """Test that dangerous actions are blocked at appropriate layers."""
        filter_system = ActionFilter(strict_mode=True)
        
        result = filter_system.filter_action(dangerous_action_dict)
        
        # Should be blocked at one of the layers
        assert result.status == FilterStatus.BLOCKED
        assert result.layer in [FilterLayer.PHYSICS, FilterLayer.SPATIAL, 
                               FilterLayer.CONTEXT, FilterLayer.HUMAN_SAFETY]
        assert result.risk_score > 0.5
    
    def test_filter_modification_through_layers(self):
        """Test action modification through multiple layers."""
        filter_system = ActionFilter(strict_mode=False)  # Allow modifications
        
        modifiable_action = {
            'force': [25.0, 0.0, 0.0],      # Will be scaled down
            'velocity': [1.5, 0.0, 0.0],    # Will be scaled down
            'target_position': [0.5, 0.3, 0.8],  # Safe position
            'current_position': [0.0, 0.0, 0.5],
            'action_type': 'controlled_move',
            'environment': {
                'lighting': 80,
                'temperature': 22,
                'vibration_level': 1,
                'hazardous_materials': False
            },
            'system_status': {
                'battery_level': 85,
                'error_count': 0,
                'sensors_operational': True
            },
            'humans_detected': []
        }
        
        result = filter_system.filter_action(modifiable_action)
        
        assert result.status == FilterStatus.MODIFIED
        assert result.filtered_action is not None
        
        # Verify modifications were applied
        filtered_force = np.linalg.norm(result.filtered_action['force'])
        filtered_velocity = np.linalg.norm(result.filtered_action['velocity'])
        
        assert filtered_force <= filter_system.physics_limits.max_force
        assert filtered_velocity <= filter_system.physics_limits.max_speed
    
    def test_filter_layer_exception_handling(self):
        """Test that exceptions in filter layers are handled gracefully."""
        filter_system = ActionFilter()
        
        # Mock exception in spatial filter
        with patch.object(filter_system, '_spatial_filter', side_effect=Exception("Test error")):
            test_action = {
                'action_type': 'test',
                'force': [5.0, 0.0, 0.0]  # Safe force
            }
            
            result = filter_system.filter_action(test_action)
            
            assert result.status == FilterStatus.BLOCKED
            assert "error" in result.reason.lower()
            assert result.risk_score == 1.0


@pytest.mark.safety
class TestRiskCalculation:
    """Test risk score calculation functionality."""
    
    def test_overall_risk_calculation(self):
        """Test overall risk calculation includes all factors."""
        filter_system = ActionFilter()
        
        high_risk_action = {
            'force': [18.0, 0.0, 0.0],     # High force
            'velocity': [0.9, 0.0, 0.0],   # High velocity  
            'tool': 'plasma',               # Dangerous tool
            'humans_detected': [
                {'distance': 2.0, 'min_safe_distance': 1.0}  # Humans present
            ]
        }
        
        risk_score = filter_system._calculate_overall_risk(high_risk_action)
        
        # Should be high risk due to multiple factors
        assert risk_score > 0.5
        assert risk_score <= 1.0
    
    def test_low_risk_calculation(self):
        """Test low risk calculation for safe actions."""
        filter_system = ActionFilter()
        
        low_risk_action = {
            'force': [2.0, 0.0, 0.0],      # Low force
            'velocity': [0.1, 0.0, 0.0],   # Low velocity
            'tool': 'gripper',              # Safe tool
            'humans_detected': []           # No humans
        }
        
        risk_score = filter_system._calculate_overall_risk(low_risk_action)
        
        assert risk_score < 0.5
        assert risk_score >= 0.1  # Minimum base risk


@pytest.mark.safety
class TestFilterConfiguration:
    """Test filter configuration and limit updates."""
    
    def test_update_physics_limits(self):
        """Test updating physics limits."""
        filter_system = ActionFilter()
        
        new_physics_limits = PhysicsLimits(
            max_force=10.0,  # Reduced from default 20.0
            max_speed=0.5,   # Reduced from default 1.0
            max_acceleration=1.0
        )
        
        filter_system.update_limits(physics_limits=new_physics_limits)
        
        assert filter_system.physics_limits.max_force == 10.0
        assert filter_system.physics_limits.max_speed == 0.5
    
    def test_update_spatial_limits(self):
        """Test updating spatial limits."""
        filter_system = ActionFilter()
        
        new_spatial_limits = SpatialLimits(
            workspace_bounds=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)),
            min_obstacle_distance=0.2,
            max_reach_distance=1.0
        )
        
        filter_system.update_limits(spatial_limits=new_spatial_limits)
        
        assert filter_system.spatial_limits.min_obstacle_distance == 0.2
        assert filter_system.spatial_limits.max_reach_distance == 1.0


@pytest.mark.performance
class TestActionFilterPerformance:
    """Performance tests for action filtering."""
    
    def test_filter_performance_benchmark(self, valid_action_dict):
        """Benchmark filtering performance."""
        filter_system = ActionFilter()
        
        # Measure filtering time
        start_time = time.time()
        num_filters = 100
        
        for _ in range(num_filters):
            filter_system.filter_action(valid_action_dict)
        
        end_time = time.time()
        avg_filter_time = (end_time - start_time) / num_filters
        
        # Filtering should be fast (< 10ms per action)
        assert avg_filter_time < 0.01
    
    def test_complex_action_filtering_performance(self):
        """Test performance with complex actions."""
        filter_system = ActionFilter()
        
        complex_action = {
            'force': [15.0, 5.0, 2.0],
            'velocity': [0.8, 0.2, 0.1],
            'acceleration': [1.5, 0.5, 0.3],
            'target_position': [0.7, 0.4, 1.2],
            'current_position': [0.1, 0.1, 0.8],
            'trajectory': [[0.1, 0.1, 0.8], [0.4, 0.25, 1.0], [0.7, 0.4, 1.2]],
            'obstacles': [
                {'position': [0.3, 0.3, 0.9], 'radius': 0.05},
                {'position': [0.6, 0.1, 1.1], 'radius': 0.08}
            ],
            'tool': 'precision_gripper',
            'environment': {
                'lighting': 75,
                'temperature': 23,
                'vibration_level': 2,
                'hazardous_materials': False
            },
            'system_status': {
                'battery_level': 70,
                'error_count': 1,
                'sensors_operational': True
            },
            'humans_detected': [
                {'distance': 2.5, 'min_safe_distance': 1.0}
            ],
            'repetitions': 1
        }
        
        start_time = time.time()
        result = filter_system.filter_action(complex_action)
        filter_time = time.time() - start_time
        
        # Even complex actions should be filtered quickly
        assert filter_time < 0.02
        assert result is not None