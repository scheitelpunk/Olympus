"""
Safety Layer Demonstration

Complete demonstration of the OLYMPUS Safety Layer capabilities:
- Action filtering with multiple layers
- Human protection system
- Risk assessment and mitigation
- Fail-safe mechanisms
- Comprehensive audit logging
"""

import time
import numpy as np
from datetime import datetime, timedelta
import logging
import threading

from src.olympus.safety_layer import (
    ActionFilter, PhysicsLimits, SpatialLimits,
    IntentionAnalyzer, RiskAssessment,
    HumanProtection, SafetyZoneConfig,
    FailSafeManager, FailSafeMechanism, FailSafeType, FailSafePriority,
    AuditLogger, SafetyEventType, EventSeverity, AuditConfiguration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafetyLayerDemo:
    """Comprehensive demonstration of safety layer capabilities"""
    
    def __init__(self):
        """Initialize all safety components"""
        print("🛡️  Initializing OLYMPUS Safety Layer Demo")
        print("=" * 60)
        
        # Initialize components
        self._setup_action_filter()
        self._setup_intention_analyzer()
        self._setup_risk_assessment()
        self._setup_human_protection()
        self._setup_failsafe_manager()
        self._setup_audit_logger()
        
        print("✅ All safety components initialized successfully\n")
    
    def _setup_action_filter(self):
        """Setup action filtering system"""
        print("🔧 Setting up Action Filter...")
        
        self.physics_limits = PhysicsLimits(
            max_force=20.0,      # 20 Newtons max
            max_speed=1.0,       # 1.0 m/s max
            max_acceleration=2.0, # 2.0 m/s² max
            max_jerk=10.0,       # 10.0 m/s³ max
            max_torque=5.0       # 5.0 N⋅m max
        )
        
        self.spatial_limits = SpatialLimits(
            workspace_bounds=((-2.0, 2.0), (-2.0, 2.0), (0.0, 3.0)),
            min_obstacle_distance=0.1,  # 10cm clearance
            max_reach_distance=2.5      # 2.5m max reach
        )
        
        self.action_filter = ActionFilter(
            physics_limits=self.physics_limits,
            spatial_limits=self.spatial_limits,
            strict_mode=True
        )
        print("   ✓ Action Filter configured with safety limits")
    
    def _setup_intention_analyzer(self):
        """Setup intention analysis system"""
        print("🧠 Setting up Intention Analyzer...")
        
        self.intention_analyzer = IntentionAnalyzer(history_size=50)
        print("   ✓ Intention Analyzer ready for pattern recognition")
    
    def _setup_risk_assessment(self):
        """Setup risk assessment system"""
        print("📊 Setting up Risk Assessment...")
        
        self.risk_assessment = RiskAssessment(
            history_window=30,
            risk_decay_factor=0.95,
            prediction_horizon=5
        )
        print("   ✓ Risk Assessment system configured")
    
    def _setup_human_protection(self):
        """Setup human protection system"""
        print("👥 Setting up Human Protection...")
        
        self.safety_config = SafetyZoneConfig(
            critical_distance=0.3,    # 30cm - emergency stop
            safety_distance=1.0,      # 1m - minimum safe distance
            warning_distance=1.5,     # 1.5m - warning zone
            monitoring_distance=3.0   # 3m - monitoring range
        )
        
        self.human_protection = HumanProtection(
            safety_config=self.safety_config,
            prediction_horizon=2.0,
            human_timeout=5.0
        )
        
        # Register protection callback
        self.human_protection.register_protection_callback(
            'demo_callback', self._human_protection_callback
        )
        print("   ✓ Human Protection system active with 4-zone safety model")
    
    def _setup_failsafe_manager(self):
        """Setup fail-safe management system"""
        print("🚨 Setting up Fail-Safe Manager...")
        
        self.failsafe_manager = FailSafeManager(check_interval=0.1)
        
        # Create sample fail-safe mechanisms
        self._create_sample_failsafes()
        
        # Register event handlers
        self.failsafe_manager.register_event_handler(
            FailSafePriority.CRITICAL, self._critical_failsafe_handler
        )
        self.failsafe_manager.register_event_handler(
            FailSafePriority.HIGH, self._high_failsafe_handler
        )
        
        print("   ✓ Fail-Safe Manager configured with sample mechanisms")
    
    def _setup_audit_logger(self):
        """Setup comprehensive audit logging"""
        print("📝 Setting up Audit Logger...")
        
        audit_config = AuditConfiguration(
            log_directory="logs/safety_demo",
            database_path="logs/safety_demo/demo_audit.db",
            max_log_size_mb=50,
            log_retention_days=30,
            real_time_alerts=True,
            integrity_checking=True
        )
        
        self.audit_logger = AuditLogger(audit_config)
        
        # Register alert handlers
        self.audit_logger.register_alert_handler(
            EventSeverity.CRITICAL, self._critical_alert_handler
        )
        self.audit_logger.register_alert_handler(
            EventSeverity.ERROR, self._error_alert_handler
        )
        
        print("   ✓ Audit Logger ready for comprehensive event tracking")
    
    def _create_sample_failsafes(self):
        """Create sample fail-safe mechanisms for demonstration"""
        
        # Emergency stop mechanism
        def check_emergency_stop():
            # Simulate hardware e-stop check
            return False  # Not triggered for demo
        
        estop_mechanism = FailSafeMechanism(
            mechanism_id="emergency_stop_demo",
            fail_safe_type=FailSafeType.HARDWARE_ESTOP,
            priority=FailSafePriority.CRITICAL,
            check_function=check_emergency_stop
        )
        
        # Force limit mechanism
        force_exceeded = False
        def check_force_limits():
            return force_exceeded
        
        force_mechanism = FailSafeMechanism(
            mechanism_id="force_limit_demo",
            fail_safe_type=FailSafeType.FORCE_LIMITS,
            priority=FailSafePriority.HIGH,
            check_function=check_force_limits
        )
        
        # Human safety mechanism
        human_too_close = False
        def check_human_safety():
            return human_too_close
        
        human_mechanism = FailSafeMechanism(
            mechanism_id="human_safety_demo",
            fail_safe_type=FailSafeType.HUMAN_SAFETY,
            priority=FailSafePriority.CRITICAL,
            check_function=check_human_safety
        )
        
        # Register mechanisms
        self.failsafe_manager.register_mechanism(estop_mechanism)
        self.failsafe_manager.register_mechanism(force_mechanism)
        self.failsafe_manager.register_mechanism(human_mechanism)
        
        # Store references for demo control
        self.demo_checks = {
            'force_exceeded': lambda val: globals().update({'force_exceeded': val}),
            'human_too_close': lambda val: globals().update({'human_too_close': val})
        }
    
    def run_comprehensive_demo(self):
        """Run comprehensive safety layer demonstration"""
        print("🚀 Starting Comprehensive Safety Layer Demo")
        print("=" * 60)
        
        try:
            # Start systems
            self._start_systems()
            
            # Run demonstration scenarios
            self._demo_safe_actions()
            self._demo_dangerous_actions()
            self._demo_human_protection()
            self._demo_risk_assessment()
            self._demo_failsafe_triggers()
            self._demo_audit_logging()
            
            # Show final statistics
            self._show_final_statistics()
            
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            logger.exception("Demo failed")
        finally:
            self._cleanup_systems()
    
    def _start_systems(self):
        """Start all safety systems"""
        print("\n🔄 Starting safety systems...")
        
        self.audit_logger.start()
        self.failsafe_manager.start_monitoring()
        
        # Log system startup
        self.audit_logger.log_event(
            event_type=SafetyEventType.SYSTEM_START,
            severity=EventSeverity.INFO,
            component="safety_demo",
            description="Safety layer demonstration started",
            data={"demo_version": "1.0", "timestamp": datetime.utcnow().isoformat()}
        )
        
        print("   ✅ All systems operational")
        time.sleep(1)
    
    def _demo_safe_actions(self):
        """Demonstrate safe action processing"""
        print("\n✅ SCENARIO 1: Safe Action Processing")
        print("-" * 40)
        
        safe_actions = [
            {
                "name": "Gentle Movement",
                "action": {
                    'force': [5.0, 0.0, 0.0],
                    'velocity': [0.3, 0.0, 0.0],
                    'target_position': [1.0, 0.5, 1.5],
                    'current_position': [0.0, 0.0, 1.0]
                }
            },
            {
                "name": "Precision Positioning",
                "action": {
                    'force': [2.0, 0.0, 0.0],
                    'velocity': [0.1, 0.0, 0.0],
                    'target_position': [0.5, 0.2, 1.2],
                    'current_position': [0.3, 0.1, 1.0],
                    'tool': 'precision_gripper'
                }
            }
        ]
        
        for scenario in safe_actions:
            print(f"\n🔍 Testing: {scenario['name']}")
            self._process_action_scenario(scenario['action'], scenario['name'])
            time.sleep(0.5)
    
    def _demo_dangerous_actions(self):
        """Demonstrate dangerous action filtering"""
        print("\n⚠️  SCENARIO 2: Dangerous Action Filtering")
        print("-" * 40)
        
        dangerous_actions = [
            {
                "name": "Excessive Force",
                "action": {
                    'force': [35.0, 0.0, 0.0],  # Exceeds 20N limit
                    'velocity': [0.3, 0.0, 0.0]
                }
            },
            {
                "name": "High Speed Operation",
                "action": {
                    'force': [10.0, 0.0, 0.0],
                    'velocity': [2.5, 0.0, 0.0]  # Exceeds 1.0 m/s limit
                }
            },
            {
                "name": "Out of Workspace",
                "action": {
                    'target_position': [3.0, 0.0, 1.0],  # Outside bounds
                    'current_position': [0.0, 0.0, 1.0]
                }
            },
            {
                "name": "Dangerous Tool",
                "action": {
                    'tool': 'plasma_cutter',
                    'force': [15.0, 0.0, 0.0],
                    'velocity': [0.8, 0.0, 0.0]
                }
            }
        ]
        
        for scenario in dangerous_actions:
            print(f"\n🔍 Testing: {scenario['name']}")
            self._process_action_scenario(scenario['action'], scenario['name'])
            time.sleep(0.5)
    
    def _demo_human_protection(self):
        """Demonstrate human protection system"""
        print("\n👥 SCENARIO 3: Human Protection System")
        print("-" * 40)
        
        human_scenarios = [
            {
                "name": "Human in Monitoring Zone",
                "detection": {
                    'id': 'human_001',
                    'position': [2.5, 0.0, 1.8],
                    'velocity': [0.0, 0.0, 0.0],
                    'confidence': 0.9
                }
            },
            {
                "name": "Human in Warning Zone",
                "detection": {
                    'id': 'human_002',
                    'position': [1.3, 0.0, 1.8],
                    'velocity': [0.0, 0.0, 0.0],
                    'confidence': 0.95
                }
            },
            {
                "name": "Human Approaching Rapidly",
                "detection": {
                    'id': 'human_003',
                    'position': [1.4, 0.0, 1.8],
                    'velocity': [-1.2, 0.0, 0.0],  # Moving toward robot
                    'confidence': 0.85
                }
            },
            {
                "name": "CRITICAL: Human Too Close",
                "detection": {
                    'id': 'human_004',
                    'position': [0.2, 0.0, 1.8],  # Critical zone
                    'velocity': [0.0, 0.0, 0.0],
                    'confidence': 0.99
                }
            }
        ]
        
        for scenario in human_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            self._process_human_scenario([scenario['detection']], scenario['name'])
            time.sleep(1)
    
    def _demo_risk_assessment(self):
        """Demonstrate risk assessment system"""
        print("\n📊 SCENARIO 4: Risk Assessment & Prediction")
        print("-" * 40)
        
        risk_scenarios = [
            {
                "name": "Low Risk Operation",
                "action": {'force': [3.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]},
                "environment": {'temperature': 22, 'lighting_level': 90},
                "system": {'battery_level': 80, 'error_count': 0}
            },
            {
                "name": "Moderate Risk - Environmental",
                "action": {'force': [12.0, 0.0, 0.0], 'velocity': [0.6, 0.0, 0.0]},
                "environment": {'temperature': 35, 'lighting_level': 30, 'vibration_level': 4},
                "system": {'battery_level': 60, 'error_count': 2}
            },
            {
                "name": "High Risk - System Issues",
                "action": {'force': [18.0, 0.0, 0.0], 'velocity': [0.9, 0.0, 0.0], 'tool': 'cutter'},
                "environment": {'temperature': 45, 'lighting_level': 20},
                "system": {'battery_level': 15, 'error_count': 8, 'maintenance_overdue': True}
            }
        ]
        
        for scenario in risk_scenarios:
            print(f"\n🔍 Assessing: {scenario['name']}")
            self._process_risk_scenario(scenario)
            time.sleep(0.5)
    
    def _demo_failsafe_triggers(self):
        """Demonstrate fail-safe mechanism triggers"""
        print("\n🚨 SCENARIO 5: Fail-Safe Mechanism Triggers")
        print("-" * 40)
        
        # Demonstrate force limit trigger
        print("\n🔍 Triggering Force Limit Fail-Safe")
        self.failsafe_manager.trigger_manual_failsafe(
            "force_limit_demo", "Demonstration of force limit exceeded"
        )
        time.sleep(1)
        
        # Show recovery attempt
        print("   🔄 Attempting recovery...")
        success = self.failsafe_manager.attempt_recovery("force_limit_demo")
        print(f"   {'✅' if success else '❌'} Recovery {'succeeded' if success else 'failed'}")
        time.sleep(1)
        
        # Demonstrate human safety trigger
        print("\n🔍 Triggering Human Safety Fail-Safe")
        self.failsafe_manager.trigger_manual_failsafe(
            "human_safety_demo", "Human detected in critical zone"
        )
        time.sleep(1)
    
    def _demo_audit_logging(self):
        """Demonstrate comprehensive audit logging"""
        print("\n📝 SCENARIO 6: Comprehensive Audit Logging")
        print("-" * 40)
        
        # Generate various types of audit events
        print("\n🔍 Generating audit events...")
        
        # Configuration change
        self.audit_logger.log_configuration_change(
            component="action_filter",
            old_config={"max_force": 20.0},
            new_config={"max_force": 18.0},
            user_id="demo_user"
        )
        
        # Safety violation
        self.audit_logger.log_event(
            event_type=SafetyEventType.SAFETY_VIOLATION,
            severity=EventSeverity.ERROR,
            component="demo_system",
            description="Simulated safety violation for demonstration",
            data={"violation_type": "speed_limit", "measured_value": 1.2, "limit": 1.0}
        )
        
        # Emergency event
        self.audit_logger.log_emergency_stop(
            trigger_reason="Demonstration emergency stop",
            system_state={"demo_mode": True, "timestamp": datetime.utcnow().isoformat()}
        )
        
        print("   ✅ Audit events logged successfully")
        
        # Query recent events
        print("\n🔍 Querying recent audit events...")
        recent_events = self.audit_logger.query_events(
            start_time=datetime.utcnow() - timedelta(minutes=5),
            limit=10
        )
        
        print(f"   📊 Found {len(recent_events)} recent events")
        for event in recent_events[-3:]:  # Show last 3
            print(f"   • {event['event_type']}: {event['description']}")
    
    def _process_action_scenario(self, action, scenario_name):
        """Process an action through all safety layers"""
        
        # Action filtering
        filter_result = self.action_filter.filter_action(action)
        print(f"   🔍 Action Filter: {filter_result.status.value}")
        if filter_result.reason:
            print(f"      Reason: {filter_result.reason}")
        
        # Intention analysis
        intention_result = self.intention_analyzer.analyze_intention(action)
        print(f"   🧠 Intention: {intention_result.intention_type.value} "
              f"(risk: {intention_result.risk_category.value})")
        
        # Risk assessment
        risk_result = self.risk_assessment.assess_risk(action)
        print(f"   📊 Risk Level: {risk_result.risk_level.label} "
              f"(score: {risk_result.overall_risk_score:.3f})")
        
        # Log to audit
        self.audit_logger.log_action_filtered(action, {
            'status': filter_result.status.value,
            'layer': filter_result.layer.value,
            'reason': filter_result.reason,
            'risk_score': filter_result.risk_score
        })
        
        # Show recommendations if any
        if risk_result.recommendations:
            print(f"   💡 Recommendations: {', '.join(risk_result.recommendations[:2])}")
    
    def _process_human_scenario(self, detections, scenario_name):
        """Process human detection scenario"""
        
        alerts = self.human_protection.update_human_detections(detections)
        constraints = self.human_protection.get_safety_constraints({})
        emergency = self.human_protection.check_emergency_conditions()
        
        for alert in alerts:
            print(f"   🚨 Alert: {alert.alert_level.value[0].upper()} - {alert.zone.value}")
            print(f"      Distance: {alert.distance:.2f}m")
            print(f"      Action: {alert.recommended_action}")
        
        print(f"   🎛️  Speed Limit: {constraints['speed_limit_factor']:.1%}")
        print(f"   🎛️  Force Limit: {constraints['force_limit_factor']:.1%}")
        
        if emergency['emergency_required']:
            print(f"   🚨 EMERGENCY REQUIRED: Risk Level {emergency['risk_level']}")
        
        # Log human proximity event
        if alerts:
            self.audit_logger.log_human_proximity(
                detections[0], alerts[0].__dict__
            )
    
    def _process_risk_scenario(self, scenario):
        """Process risk assessment scenario"""
        
        risk_result = self.risk_assessment.assess_risk(
            action=scenario['action'],
            environment=scenario['environment'],
            system_state=scenario['system']
        )
        
        print(f"   📊 Risk Level: {risk_result.risk_level.label}")
        print(f"   📈 Risk Score: {risk_result.overall_risk_score:.3f}")
        print(f"   🔮 Predicted Risk: {risk_result.predicted_risk:.3f}")
        print(f"   📚 Risk Factors: {len(risk_result.risk_factors)}")
        
        if risk_result.recommendations:
            print(f"   💡 Top Recommendations:")
            for rec in risk_result.recommendations[:3]:
                print(f"      • {rec}")
    
    def _show_final_statistics(self):
        """Show final system statistics"""
        print("\n📈 FINAL SYSTEM STATISTICS")
        print("=" * 60)
        
        # Action filter status
        filter_status = self.action_filter.get_filter_status()
        print(f"🔧 Action Filter: {len(filter_status['active_layers'])} layers active")
        
        # Human protection stats
        protection_stats = self.human_protection.get_protection_statistics()
        print(f"👥 Human Protection: {protection_stats['runtime_stats']['total_detections']} detections processed")
        
        # Fail-safe system status
        failsafe_status = self.failsafe_manager.get_system_status()
        print(f"🚨 Fail-Safe: {failsafe_status['total_mechanisms']} mechanisms, "
              f"{failsafe_status['triggered_mechanisms']} triggered")
        
        # Audit logger statistics
        audit_stats = self.audit_logger.get_statistics()
        print(f"📝 Audit Logger: {audit_stats['runtime_stats']['events_logged']} events logged")
        
        # Risk assessment trends (if available)
        try:
            risk_trends = self.risk_assessment.get_risk_trends(hours=1)
            print(f"📊 Risk Assessment: {risk_trends['assessment_count']} assessments, "
                  f"avg risk: {risk_trends['average_risk']:.3f}")
        except:
            pass
    
    def _cleanup_systems(self):
        """Clean up and stop all systems"""
        print("\n🔄 Shutting down safety systems...")
        
        self.failsafe_manager.stop_monitoring()
        self.audit_logger.stop()
        
        print("✅ All systems shut down safely")
    
    def _human_protection_callback(self, event_data):
        """Callback for human protection events"""
        alert_level = event_data['alert_level']
        if alert_level in ['danger', 'emergency']:
            print(f"   🚨 PROTECTION CALLBACK: {alert_level.upper()} - {event_data['recommended_action']}")
    
    def _critical_failsafe_handler(self, event):
        """Handler for critical fail-safe events"""
        print(f"   🚨 CRITICAL FAIL-SAFE: {event.mechanism_id} - {event.description}")
    
    def _high_failsafe_handler(self, event):
        """Handler for high priority fail-safe events"""
        print(f"   ⚠️  HIGH FAIL-SAFE: {event.mechanism_id} - {event.description}")
    
    def _critical_alert_handler(self, event):
        """Handler for critical audit alerts"""
        print(f"   🚨 CRITICAL AUDIT ALERT: {event.description}")
    
    def _error_alert_handler(self, event):
        """Handler for error audit alerts"""
        print(f"   ❌ ERROR AUDIT ALERT: {event.description}")


def main():
    """Main demonstration function"""
    print("🛡️  OLYMPUS Safety Layer Comprehensive Demo")
    print("=" * 60)
    print("This demonstration showcases all safety layer capabilities:")
    print("• Multi-layer action filtering")
    print("• Human proximity protection")
    print("• Risk assessment and prediction")  
    print("• Fail-safe mechanisms")
    print("• Comprehensive audit logging")
    print("=" * 60)
    
    input("Press Enter to start the demonstration...")
    
    try:
        demo = SafetyLayerDemo()
        demo.run_comprehensive_demo()
        
        print("\n🎉 Safety Layer Demo Completed Successfully!")
        print("Check the logs/safety_demo directory for audit logs.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        logger.exception("Demo failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())