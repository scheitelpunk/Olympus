"""
Tests for Human Protection System
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.olympus.safety_layer.human_protection import (
    HumanProtection, Human, ProximityAlert, AlertLevel, HumanZone,
    SafetyZoneConfig
)


class TestHuman:
    """Test cases for Human class"""
    
    def test_human_creation(self):
        """Test Human object creation"""
        human = Human(
            id="human_001",
            position=(1.0, 0.5, 1.8),
            velocity=(0.1, 0.0, 0.0),
            size_estimate=0.6,
            confidence=0.9
        )
        
        assert human.id == "human_001"
        assert human.position == (1.0, 0.5, 1.8)
        assert human.velocity == (0.1, 0.0, 0.0)
        assert human.size_estimate == 0.6
        assert human.confidence == 0.9
        assert isinstance(human.last_seen, datetime)
    
    def test_distance_calculation(self):
        """Test distance calculation from robot origin"""
        human = Human(
            id="human_001",
            position=(3.0, 4.0, 0.0)  # 5 meters from origin
        )
        
        assert abs(human.distance_from_robot - 5.0) < 0.001
    
    def test_position_prediction(self):
        """Test position prediction based on velocity"""
        human = Human(
            id="human_001",
            position=(0.0, 0.0, 0.0),
            velocity=(1.0, 0.5, 0.0)
        )
        
        predicted = human.predicted_position(2.0)  # 2 seconds ahead
        
        assert predicted == (2.0, 1.0, 0.0)
    
    def test_position_prediction_no_velocity(self):
        """Test position prediction when velocity is None"""
        human = Human(
            id="human_001",
            position=(1.0, 2.0, 3.0),
            velocity=None
        )
        
        predicted = human.predicted_position(5.0)
        
        assert predicted == (1.0, 2.0, 3.0)  # Should remain the same


class TestSafetyZoneConfig:
    """Test cases for SafetyZoneConfig"""
    
    def test_default_configuration(self):
        """Test default safety zone configuration"""
        config = SafetyZoneConfig()
        
        assert config.critical_distance == 0.3
        assert config.safety_distance == 1.0
        assert config.warning_distance == 1.5
        assert config.monitoring_distance == 3.0
        
        assert config.critical_speed_factor == 0.0
        assert config.safety_speed_factor == 0.1
        assert config.warning_speed_factor == 0.3
        assert config.monitoring_speed_factor == 0.7
    
    def test_custom_configuration(self):
        """Test custom safety zone configuration"""
        config = SafetyZoneConfig(
            critical_distance=0.2,
            safety_distance=0.8,
            warning_distance=1.2,
            monitoring_distance=2.5
        )
        
        assert config.critical_distance == 0.2
        assert config.safety_distance == 0.8
        assert config.warning_distance == 1.2
        assert config.monitoring_distance == 2.5


class TestHumanProtection:
    """Test cases for HumanProtection system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.safety_config = SafetyZoneConfig(
            critical_distance=0.3,
            safety_distance=1.0,
            warning_distance=1.5,
            monitoring_distance=3.0
        )
        
        self.protection = HumanProtection(
            safety_config=self.safety_config,
            prediction_horizon=2.0,
            human_timeout=5.0
        )
    
    def test_no_humans_detected(self):
        """Test behavior when no humans are detected"""
        constraints = self.protection.get_safety_constraints({})
        
        assert constraints['speed_limit_factor'] == 1.0
        assert constraints['force_limit_factor'] == 1.0
        assert constraints['emergency_stop_required'] == False
        assert constraints['recommended_action'] == 'proceed'
        assert len(constraints['active_constraints']) == 0
    
    def test_human_in_monitoring_zone(self):
        """Test human detection in monitoring zone"""
        detections = [
            {
                'id': 'human_001',
                'position': [2.5, 0.0, 1.8],  # Within monitoring distance
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        
        alerts = self.protection.update_human_detections(detections)
        constraints = self.protection.get_safety_constraints({})
        
        assert len(alerts) == 1
        assert alerts[0].alert_level == AlertLevel.CAUTION
        assert alerts[0].zone == HumanZone.MONITORING
        
        assert constraints['speed_limit_factor'] == 0.7
        assert constraints['force_limit_factor'] == 0.7
        assert constraints['emergency_stop_required'] == False
    
    def test_human_in_warning_zone(self):
        """Test human detection in warning zone"""
        detections = [
            {
                'id': 'human_001',
                'position': [1.2, 0.0, 1.8],  # Within warning distance
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        
        alerts = self.protection.update_human_detections(detections)
        constraints = self.protection.get_safety_constraints({})
        
        assert len(alerts) == 1
        assert alerts[0].alert_level == AlertLevel.WARNING
        assert alerts[0].zone == HumanZone.WARNING
        
        assert constraints['speed_limit_factor'] == 0.3
        assert constraints['force_limit_factor'] == 0.3
        assert constraints['emergency_stop_required'] == False
    
    def test_human_in_safety_zone(self):
        """Test human detection in safety zone"""
        detections = [
            {
                'id': 'human_001',
                'position': [0.8, 0.0, 1.8],  # Within safety distance
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        
        alerts = self.protection.update_human_detections(detections)
        constraints = self.protection.get_safety_constraints({})
        
        assert len(alerts) == 1
        assert alerts[0].alert_level == AlertLevel.DANGER
        assert alerts[0].zone == HumanZone.SAFETY
        
        assert constraints['speed_limit_factor'] == 0.1
        assert constraints['force_limit_factor'] == 0.1
        assert constraints['emergency_stop_required'] == False
    
    def test_human_in_critical_zone(self):
        """Test human detection in critical zone - emergency stop required"""
        detections = [
            {
                'id': 'human_001',
                'position': [0.2, 0.0, 1.8],  # Within critical distance
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        
        alerts = self.protection.update_human_detections(detections)
        constraints = self.protection.get_safety_constraints({})
        
        assert len(alerts) == 1
        assert alerts[0].alert_level == AlertLevel.EMERGENCY
        assert alerts[0].zone == HumanZone.CRITICAL
        
        assert constraints['speed_limit_factor'] == 0.0
        assert constraints['force_limit_factor'] == 0.0
        assert constraints['emergency_stop_required'] == True
        assert constraints['recommended_action'] == 'emergency_stop'
    
    def test_multiple_humans(self):
        """Test multiple human detections with different zones"""
        detections = [
            {
                'id': 'human_001',
                'position': [2.0, 0.0, 1.8],  # Monitoring zone
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            },
            {
                'id': 'human_002',
                'position': [1.2, 1.0, 1.8],  # Warning zone
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.8
            }
        ]
        
        alerts = self.protection.update_human_detections(detections)
        constraints = self.protection.get_safety_constraints({})
        
        assert len(alerts) == 2
        
        # Most restrictive constraints should apply (warning zone)
        assert constraints['speed_limit_factor'] == 0.3
        assert constraints['force_limit_factor'] == 0.3
        assert constraints['most_critical_zone'] == 'warning'
        assert constraints['human_count'] == 2
    
    def test_collision_prediction_safe(self):
        """Test collision prediction for safe trajectory"""
        # Add human detection
        detections = [
            {
                'id': 'human_001',
                'position': [2.0, 2.0, 1.8],
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        # Robot trajectory moving away from human
        robot_trajectory = [
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.5, 0.0, 1.0)
        ]
        time_steps = [0.0, 0.5, 1.0, 1.5]
        
        collision_risk = self.protection.predict_collision_risk(robot_trajectory, time_steps)
        
        assert collision_risk['collision_risk'] < 0.5
        assert 'No significant collision risk' in collision_risk['recommendations'][0]
    
    def test_collision_prediction_dangerous(self):
        """Test collision prediction for dangerous trajectory"""
        # Add human detection
        detections = [
            {
                'id': 'human_001',
                'position': [1.0, 0.0, 1.8],
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        # Robot trajectory moving toward human
        robot_trajectory = [
            (0.0, 0.0, 1.0),
            (0.3, 0.0, 1.0),
            (0.7, 0.0, 1.0),
            (1.0, 0.0, 1.0)  # Same position as human
        ]
        time_steps = [0.0, 0.5, 1.0, 1.5]
        
        collision_risk = self.protection.predict_collision_risk(robot_trajectory, time_steps)
        
        assert collision_risk['collision_risk'] > 0.8
        assert 'CRITICAL' in collision_risk['recommendations'][0]
        assert len(collision_risk['risk_points']) > 0
    
    def test_moving_human_prediction(self):
        """Test collision prediction with moving human"""
        # Add moving human detection
        detections = [
            {
                'id': 'human_001',
                'position': [2.0, 0.0, 1.8],
                'velocity': [-1.0, 0.0, 0.0],  # Moving toward robot
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        # Robot trajectory
        robot_trajectory = [
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 1.0)
        ]
        time_steps = [0.0, 1.0, 2.0]
        
        collision_risk = self.protection.predict_collision_risk(robot_trajectory, time_steps)
        
        # Should detect high risk due to converging paths
        assert collision_risk['collision_risk'] > 0.5
        assert len(collision_risk['risk_points']) > 0
    
    def test_emergency_conditions_detection(self):
        """Test emergency condition detection"""
        # Add human in critical zone
        detections = [
            {
                'id': 'human_001',
                'position': [0.15, 0.0, 1.8],  # Critical distance
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        emergency_status = self.protection.check_emergency_conditions()
        
        assert emergency_status['emergency_required'] == True
        assert emergency_status['risk_level'] == 4
        assert len(emergency_status['conditions']) > 0
        assert 'EMERGENCY STOP' in emergency_status['immediate_actions']
        assert emergency_status['human_count_in_danger'] == 1
    
    def test_rapid_approach_detection(self):
        """Test detection of rapidly approaching human"""
        # Add rapidly approaching human
        detections = [
            {
                'id': 'human_001',
                'position': [1.4, 0.0, 1.8],  # In warning zone
                'velocity': [-2.0, 0.0, 0.0],  # Rapidly approaching
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        emergency_status = self.protection.check_emergency_conditions()
        
        assert emergency_status['risk_level'] == 3
        assert any('rapidly' in condition for condition in emergency_status['conditions'])
        assert 'REDUCE SPEED' in ' '.join(emergency_status['immediate_actions'])
    
    def test_human_timeout(self):
        """Test human detection timeout"""
        # Add human detection
        detections = [
            {
                'id': 'human_001',
                'position': [1.0, 0.0, 1.8],
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        # Verify human is detected
        status = self.protection.get_human_status()
        assert status['humans_detected'] == 1
        
        # Simulate timeout by manually setting old timestamp
        human = list(self.protection.detected_humans.values())[0]
        human.last_seen = datetime.utcnow() - timedelta(seconds=10)  # Older than timeout
        
        # Update with empty detections to trigger cleanup
        self.protection.update_human_detections([])
        
        # Verify human was removed
        status = self.protection.get_human_status()
        assert status['humans_detected'] == 0
    
    def test_protection_callbacks(self):
        """Test protection callback system"""
        callback_called = []
        
        def test_callback(event_data):
            callback_called.append(event_data)
        
        self.protection.register_protection_callback('test', test_callback)
        
        # Trigger danger level alert
        detections = [
            {
                'id': 'human_001',
                'position': [0.8, 0.0, 1.8],  # Safety zone
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        # Callback should have been called
        assert len(callback_called) == 1
        assert callback_called[0]['alert_level'] == 'danger'
        assert callback_called[0]['human_id'] == 'human_001'
    
    def test_safety_config_update(self):
        """Test updating safety configuration"""
        new_config = SafetyZoneConfig(
            safety_distance=2.0,  # Increased from 1.0
            warning_distance=3.0   # Increased from 1.5
        )
        
        self.protection.update_safety_config(new_config)
        
        # Test with human at distance that would now be in safety zone
        detections = [
            {
                'id': 'human_001',
                'position': [1.5, 0.0, 1.8],  # Would be warning zone with old config
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        
        alerts = self.protection.update_human_detections(detections)
        
        # Should now be classified as safety zone with new config
        assert alerts[0].zone == HumanZone.SAFETY
        assert alerts[0].alert_level == AlertLevel.DANGER
    
    def test_manual_emergency_mode(self):
        """Test manual emergency mode activation"""
        callback_called = []
        
        def emergency_callback(event_data):
            callback_called.append(event_data)
        
        self.protection.register_protection_callback('emergency', emergency_callback)
        
        self.protection.force_emergency_mode("Manual safety test")
        
        # Callback should have been triggered
        assert len(callback_called) == 1
        assert callback_called[0]['alert_level'] == 'emergency'
        assert 'Manual safety test' in callback_called[0]['recommended_action']
    
    def test_system_reset(self):
        """Test protection system reset"""
        # Add some detections and history
        detections = [
            {
                'id': 'human_001',
                'position': [1.0, 0.0, 1.8],
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        # Verify system has data
        status = self.protection.get_human_status()
        assert status['humans_detected'] == 1
        
        # Reset system
        self.protection.reset_protection_system()
        
        # Verify system is clean
        status = self.protection.get_human_status()
        assert status['humans_detected'] == 0
        assert len(self.protection.alert_history) == 0
    
    def test_protection_statistics(self):
        """Test protection statistics collection"""
        # Generate some activity
        detections = [
            {
                'id': 'human_001',
                'position': [0.8, 0.0, 1.8],  # Safety zone - triggers callback
                'velocity': [0.0, 0.0, 0.0],
                'confidence': 0.9
            }
        ]
        self.protection.update_human_detections(detections)
        
        stats = self.protection.get_protection_statistics()
        
        assert 'runtime_stats' in stats
        assert 'recent_alerts_count' in stats
        assert 'alert_distribution' in stats
        assert 'configuration' in stats
        
        assert stats['runtime_stats']['total_detections'] == 1
        assert stats['runtime_stats']['protection_activations'] >= 1


if __name__ == "__main__":
    pytest.main([__file__])