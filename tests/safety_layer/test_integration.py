"""
Integration Tests for Complete Safety Layer System

Tests the integration and interaction between all safety layer components:
- Action Filter + Human Protection integration
- Risk Assessment + Fail-Safe coordination
- Audit Logger capturing all safety events
- End-to-end safety pipeline validation
"""

import pytest
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.olympus.safety_layer import (
    ActionFilter, PhysicsLimits, SpatialLimits,
    IntentionAnalyzer, RiskAssessment, 
    HumanProtection, SafetyZoneConfig,
    FailSafeManager, FailSafeMechanism, FailSafeType, FailSafePriority,
    AuditLogger, SafetyEventType, EventSeverity, AuditConfiguration
)


class TestSafetyLayerIntegration:
    """Integration tests for complete safety layer system"""
    
    def setup_method(self):
        """Setup integrated safety system"""
        # Action Filter
        self.physics_limits = PhysicsLimits(max_force=20.0, max_speed=1.0)
        self.spatial_limits = SpatialLimits(
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        )
        self.action_filter = ActionFilter(
            physics_limits=self.physics_limits,
            spatial_limits=self.spatial_limits,
            strict_mode=True
        )
        
        # Human Protection
        self.safety_config = SafetyZoneConfig()
        self.human_protection = HumanProtection(safety_config=self.safety_config)
        
        # Risk Assessment
        self.risk_assessment = RiskAssessment()
        
        # Intention Analyzer
        self.intention_analyzer = IntentionAnalyzer()
        
        # Fail-Safe Manager
        self.failsafe_manager = FailSafeManager()
        
        # Audit Logger (in-memory for testing)
        audit_config = AuditConfiguration(
            database_path=":memory:",  # In-memory SQLite
            real_time_alerts=False     # Disable for testing
        )
        self.audit_logger = AuditLogger(audit_config)
        self.audit_logger.start()
        
        # Event tracking for tests
        self.safety_events = []
        self.failsafe_events = []
    
    def teardown_method(self):
        """Cleanup after tests"""
        if hasattr(self, 'audit_logger'):
            self.audit_logger.stop()
        if hasattr(self, 'failsafe_manager'):
            self.failsafe_manager.stop_monitoring()
    
    def test_safe_action_end_to_end(self):
        """Test safe action processing through all safety layers"""
        
        safe_action = {
            'force': [5.0, 0.0, 0.0],
            'velocity': [0.3, 0.0, 0.0],
            'target_position': [0.5, 0.5, 1.0],
            'current_position': [0.0, 0.0, 1.0]
        }
        
        # Process through action filter
        filter_result = self.action_filter.filter_action(safe_action)
        assert filter_result.status.value == "approved"
        
        # Analyze intention
        intention_result = self.intention_analyzer.analyze_intention(safe_action)
        assert intention_result.risk_category.value in ["low", "medium"]
        
        # Assess risk
        risk_result = self.risk_assessment.assess_risk(safe_action)
        assert risk_result.risk_level.label in ["minimal", "low", "moderate"]
        
        # Log the complete pipeline
        self.audit_logger.log_action_filtered(safe_action, {
            'filter_status': filter_result.status.value,
            'intention_type': intention_result.intention_type.value,
            'risk_level': risk_result.risk_level.label,
            'pipeline_result': 'approved'
        })
        
        # Verify audit log
        events = self.audit_logger.query_events(limit=5)
        assert len(events) >= 1
        assert events[0]['event_type'] == 'action_filtered'
        assert 'approved' in str(events[0]['data'])
    
    def test_dangerous_action_blocked_with_logging(self):
        """Test dangerous action blocked and logged appropriately"""
        
        dangerous_action = {
            'force': [30.0, 0.0, 0.0],  # Exceeds limit
            'velocity': [1.5, 0.0, 0.0],  # Exceeds limit
            'tool': 'plasma_cutter'
        }
        
        # Should be blocked by action filter
        filter_result = self.action_filter.filter_action(dangerous_action)
        assert filter_result.status.value == "blocked"
        
        # Still analyze intention (for learning)
        intention_result = self.intention_analyzer.analyze_intention(dangerous_action)
        assert intention_result.risk_category.value in ["high", "critical"]
        
        # Risk assessment should also show high risk
        risk_result = self.risk_assessment.assess_risk(dangerous_action)
        assert risk_result.risk_level.label in ["high", "critical"]
        
        # Log safety violation
        self.audit_logger.log_event(
            event_type=SafetyEventType.SAFETY_VIOLATION,
            severity=EventSeverity.ERROR,
            component="integration_test",
            description="Dangerous action blocked by safety system",
            data={
                'action': dangerous_action,
                'filter_result': filter_result.reason,
                'risk_score': risk_result.overall_risk_score
            }
        )
        
        # Verify comprehensive logging
        events = self.audit_logger.query_events(limit=5)
        violation_events = [e for e in events if e['event_type'] == 'safety_violation']
        assert len(violation_events) >= 1
        assert 'blocked' in str(violation_events[0]['data'])
    
    def test_human_proximity_integrated_response(self):
        """Test integrated response to human proximity detection"""
        
        # Human detection in critical zone
        human_detections = [{
            'id': 'human_001',
            'position': [0.2, 0.0, 1.8],  # Critical zone
            'velocity': [0.0, 0.0, 0.0],
            'confidence': 0.95
        }]
        
        # Process human detection
        alerts = self.human_protection.update_human_detections(human_detections)
        assert len(alerts) == 1
        assert alerts[0].alert_level.value[0] == "emergency"
        
        # Get safety constraints
        constraints = self.human_protection.get_safety_constraints({})
        assert constraints['emergency_stop_required'] == True
        assert constraints['speed_limit_factor'] == 0.0
        
        # Action should be modified/blocked based on human proximity
        test_action = {
            'force': [10.0, 0.0, 0.0],
            'velocity': [0.5, 0.0, 0.0],
            'humans_detected': human_detections
        }
        
        filter_result = self.action_filter.filter_action(test_action)
        assert filter_result.status.value in ["blocked", "requires_confirmation"]
        
        # Log human proximity event
        self.audit_logger.log_human_proximity(human_detections[0], alerts[0].__dict__)
        
        # Should trigger fail-safe if integrated
        # (Would need fail-safe mechanism connected to human protection)
        
        # Verify audit trail
        events = self.audit_logger.query_events(limit=5)
        proximity_events = [e for e in events if e['event_type'] == 'proximity_alert']
        assert len(proximity_events) >= 1
    
    def test_failsafe_integration_with_audit(self):
        """Test fail-safe mechanism integration with audit logging"""
        
        # Create test fail-safe mechanism
        trigger_condition = False
        def test_check():
            return trigger_condition
        
        test_mechanism = FailSafeMechanism(
            mechanism_id="integration_test",
            fail_safe_type=FailSafeType.FORCE_LIMITS,
            priority=FailSafePriority.HIGH,
            check_function=test_check
        )
        
        # Register mechanism and start monitoring
        self.failsafe_manager.register_mechanism(test_mechanism)
        self.failsafe_manager.start_monitoring()
        
        # Register event handler to capture fail-safe events
        def capture_event(event):
            self.failsafe_events.append(event)
            # Log to audit system
            self.audit_logger.log_failsafe_trigger(
                event.mechanism_id, event.__dict__
            )
        
        self.failsafe_manager.register_event_handler(
            FailSafePriority.HIGH, capture_event
        )
        
        # Trigger fail-safe manually
        success = self.failsafe_manager.trigger_manual_failsafe(
            "integration_test", "Test trigger for integration"
        )
        assert success == True
        
        # Give time for processing
        time.sleep(0.2)
        
        # Verify fail-safe was captured
        assert len(self.failsafe_events) >= 1
        
        # Verify audit logging
        events = self.audit_logger.query_events(limit=10)
        failsafe_events = [e for e in events if e['event_type'] == 'failsafe_triggered']
        assert len(failsafe_events) >= 1
        
        self.failsafe_manager.stop_monitoring()
    
    def test_cumulative_risk_assessment(self):
        """Test cumulative risk assessment across multiple actions"""
        
        # Series of progressively riskier actions
        action_series = [
            {'force': [5.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]},    # Low risk
            {'force': [10.0, 0.0, 0.0], 'velocity': [0.5, 0.0, 0.0]},   # Medium risk
            {'force': [15.0, 0.0, 0.0], 'velocity': [0.8, 0.0, 0.0]},   # Higher risk
            {'force': [18.0, 0.0, 0.0], 'velocity': [0.9, 0.0, 0.0], 'tool': 'cutter'}  # High risk
        ]
        
        risk_scores = []
        
        for i, action in enumerate(action_series):
            # Process each action
            filter_result = self.action_filter.filter_action(action)
            risk_result = self.risk_assessment.assess_risk(action)
            
            risk_scores.append(risk_result.overall_risk_score)
            
            # Log each action
            self.audit_logger.log_action_filtered(action, {
                'sequence_number': i + 1,
                'filter_status': filter_result.status.value,
                'cumulative_risk': risk_result.cumulative_risk,
                'predicted_risk': risk_result.predicted_risk
            })
            
            time.sleep(0.1)  # Small delay to separate actions
        
        # Risk should generally increase
        assert risk_scores[-1] > risk_scores[0], "Risk should increase with more dangerous actions"
        
        # Verify complete audit trail
        events = self.audit_logger.query_events(limit=10)
        action_events = [e for e in events if e['event_type'] == 'action_filtered']
        assert len(action_events) >= 4  # All actions logged
    
    def test_emergency_scenario_coordination(self):
        """Test coordinated response to emergency scenario"""
        
        # Simulate emergency: human in critical zone + high-risk action
        emergency_action = {
            'force': [25.0, 0.0, 0.0],  # Excessive force
            'velocity': [1.2, 0.0, 0.0],  # High speed
            'tool': 'plasma_cutter',
            'humans_detected': [{
                'id': 'emergency_human',
                'position': [0.15, 0.0, 1.8],  # Critical zone
                'distance': 0.15,
                'min_safe_distance': 1.0
            }]
        }
        
        # Process through all systems
        filter_result = self.action_filter.filter_action(emergency_action)
        assert filter_result.status.value == "blocked", "Emergency action should be blocked"
        
        # Human protection should trigger emergency
        human_detections = emergency_action['humans_detected']
        alerts = self.human_protection.update_human_detections(human_detections)
        emergency_status = self.human_protection.check_emergency_conditions()
        assert emergency_status['emergency_required'] == True
        
        # Risk assessment should show critical risk
        risk_result = self.risk_assessment.assess_risk(emergency_action)
        assert risk_result.risk_level.label in ["high", "critical"]
        
        # Log emergency stop
        self.audit_logger.log_emergency_stop(
            "Human in critical zone with dangerous action attempted",
            {
                'action': emergency_action,
                'human_distance': 0.15,
                'risk_score': risk_result.overall_risk_score,
                'filter_blocked': True
            }
        )
        
        # Verify comprehensive emergency logging
        events = self.audit_logger.query_events(limit=10)
        emergency_events = [e for e in events if e['event_type'] == 'emergency_stop']
        assert len(emergency_events) >= 1
        
        # Check event severity
        assert emergency_events[0]['severity'] == 'critical'
    
    def test_configuration_change_impact(self):
        """Test impact of configuration changes across safety systems"""
        
        # Record initial configuration
        initial_config = self.action_filter.get_filter_status()
        
        # Test action that would be approved with current limits
        test_action = {
            'force': [18.0, 0.0, 0.0],  # Below 20N limit
            'velocity': [0.9, 0.0, 0.0]  # Below 1.0 m/s limit
        }
        
        initial_result = self.action_filter.filter_action(test_action)
        assert initial_result.status.value == "approved"
        
        # Change physics limits to be more restrictive
        new_physics_limits = PhysicsLimits(max_force=15.0, max_speed=0.8)
        self.action_filter.update_limits(physics_limits=new_physics_limits)
        
        # Log configuration change
        self.audit_logger.log_configuration_change(
            component="action_filter",
            old_config={"max_force": 20.0, "max_speed": 1.0},
            new_config={"max_force": 15.0, "max_speed": 0.8},
            user_id="integration_test"
        )
        
        # Same action should now be blocked or modified
        new_result = self.action_filter.filter_action(test_action)
        assert new_result.status.value in ["blocked", "modified"]
        assert new_result.status != initial_result.status
        
        # Verify configuration change was logged
        events = self.audit_logger.query_events(limit=5)
        config_events = [e for e in events if e['event_type'] == 'configuration_changed']
        assert len(config_events) >= 1
        assert 'max_force' in str(config_events[0]['data'])
    
    def test_system_performance_under_load(self):
        """Test safety system performance under load"""
        
        import concurrent.futures
        import threading
        
        # Metrics tracking
        processing_times = []
        results = []
        errors = []
        
        def process_action(action_data):
            """Process a single action through safety pipeline"""
            start_time = time.time()
            try:
                action_id, action = action_data
                
                # Process through safety pipeline
                filter_result = self.action_filter.filter_action(action)
                intention_result = self.intention_analyzer.analyze_intention(action)
                risk_result = self.risk_assessment.assess_risk(action)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                results.append({
                    'action_id': action_id,
                    'filter_status': filter_result.status.value,
                    'risk_level': risk_result.risk_level.label,
                    'processing_time': processing_time
                })
                
                # Log to audit (this tests concurrent logging)
                self.audit_logger.log_action_filtered(action, {
                    'action_id': action_id,
                    'processing_time': processing_time,
                    'load_test': True
                })
                
            except Exception as e:
                errors.append(str(e))
        
        # Generate test actions
        test_actions = []
        for i in range(20):  # 20 concurrent actions
            action = {
                'force': [np.random.uniform(1, 25), 0.0, 0.0],
                'velocity': [np.random.uniform(0.1, 1.5), 0.0, 0.0],
                'target_position': [
                    np.random.uniform(-1.5, 1.5),
                    np.random.uniform(-1.5, 1.5), 
                    np.random.uniform(0.5, 2.5)
                ]
            }
            test_actions.append((f"action_{i:03d}", action))
        
        # Process actions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_action, action_data) for action_data in test_actions]
            concurrent.futures.wait(futures)
        
        # Analyze results
        assert len(errors) == 0, f"Errors during concurrent processing: {errors}"
        assert len(results) == 20, "Not all actions were processed"
        
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        # Performance assertions (adjust based on requirements)
        assert avg_processing_time < 0.1, f"Average processing time too slow: {avg_processing_time:.3f}s"
        assert max_processing_time < 0.5, f"Max processing time too slow: {max_processing_time:.3f}s"
        
        # Verify all actions were logged
        time.sleep(0.5)  # Allow time for async logging
        events = self.audit_logger.query_events(limit=25)
        load_test_events = [e for e in events if e['data'].get('load_test')]
        assert len(load_test_events) >= 15, "Not all load test actions were logged"
        
        print(f"Performance Test Results:")
        print(f"  Actions processed: {len(results)}")
        print(f"  Average time: {avg_processing_time:.3f}s")
        print(f"  Max time: {max_processing_time:.3f}s")
        print(f"  Errors: {len(errors)}")
        print(f"  Events logged: {len(load_test_events)}")
    
    def test_audit_integrity_verification(self):
        """Test audit log integrity verification"""
        
        # Generate various events
        test_events = [
            (SafetyEventType.ACTION_FILTERED, "Action filtering test"),
            (SafetyEventType.HUMAN_DETECTED, "Human detection test"),
            (SafetyEventType.SAFETY_VIOLATION, "Safety violation test"),
            (SafetyEventType.EMERGENCY_STOP, "Emergency stop test")
        ]
        
        for event_type, description in test_events:
            self.audit_logger.log_event(
                event_type=event_type,
                severity=EventSeverity.INFO,
                component="integrity_test",
                description=description,
                data={"test": True, "timestamp": datetime.utcnow().isoformat()}
            )
        
        # Allow time for processing
        time.sleep(0.2)
        
        # Verify integrity of logged events
        integrity_result = self.audit_logger.verify_integrity(limit=10)
        
        assert integrity_result['total_checked'] >= len(test_events)
        assert integrity_result['integrity_failures'] == 0, "Integrity check should pass for valid logs"
        assert integrity_result['success_rate'] == 1.0, "All events should pass integrity check"
        
        print(f"Integrity Check Results:")
        print(f"  Events checked: {integrity_result['total_checked']}")
        print(f"  Failures: {integrity_result['integrity_failures']}")
        print(f"  Success rate: {integrity_result['success_rate']:.2%}")


class TestSafetySystemResilience:
    """Test safety system resilience and error handling"""
    
    def setup_method(self):
        """Setup minimal safety system for resilience testing"""
        self.action_filter = ActionFilter()
        self.human_protection = HumanProtection()
        
        audit_config = AuditConfiguration(database_path=":memory:")
        self.audit_logger = AuditLogger(audit_config)
        self.audit_logger.start()
    
    def teardown_method(self):
        """Cleanup"""
        if hasattr(self, 'audit_logger'):
            self.audit_logger.stop()
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        
        # Test invalid action data
        invalid_actions = [
            {},  # Empty action
            {"force": "invalid"},  # Wrong data type
            {"force": [float('inf'), 0, 0]},  # Infinite values
            {"force": [float('nan'), 0, 0]},  # NaN values
            {"velocity": [1e10, 0, 0]},  # Extreme values
        ]
        
        for invalid_action in invalid_actions:
            try:
                result = self.action_filter.filter_action(invalid_action)
                # Should either handle gracefully or provide meaningful error
                assert hasattr(result, 'status'), "Filter should return valid result object"
            except Exception as e:
                # Exceptions should be meaningful, not system crashes
                assert isinstance(e, (ValueError, TypeError)), f"Unexpected exception type: {type(e)}"
    
    def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access"""
        
        import threading
        errors = []
        results = []
        
        def worker():
            try:
                for i in range(10):
                    action = {
                        'force': [5.0, 0.0, 0.0],
                        'velocity': [0.3, 0.0, 0.0]
                    }
                    result = self.action_filter.filter_action(action)
                    results.append(result.status.value)
                    
                    # Log event
                    self.audit_logger.log_event(
                        event_type=SafetyEventType.ACTION_FILTERED,
                        severity=EventSeverity.INFO,
                        component="concurrency_test",
                        description=f"Thread {threading.current_thread().ident} action {i}",
                        data={"thread_id": threading.current_thread().ident, "action_id": i}
                    )
                    
            except Exception as e:
                errors.append(f"Thread {threading.current_thread().ident}: {str(e)}")
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors and consistent results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 50, "All actions should be processed"
        assert all(result == "approved" for result in results), "All safe actions should be approved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])