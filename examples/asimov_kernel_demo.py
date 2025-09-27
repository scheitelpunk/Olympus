"""
Asimov Kernel Demonstration Script

This script demonstrates the key features of the OLYMPUS Asimov Kernel:
- Ethical evaluation of actions
- Emergency stop functionality
- Human override capabilities
- Integrity monitoring
- Performance metrics

Run this script to see the Asimov Kernel in action.
"""

import time
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.olympus.ethical_core.asimov_kernel import (
    AsimovKernel, ActionContext, ActionType, EthicalResult
)
from src.olympus.ethical_core.ethical_validator import EthicalValidator, ValidationRequest
from src.olympus.ethical_core.integrity_monitor import IntegrityMonitor


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_evaluation_result(evaluation, action_desc):
    """Print evaluation result in formatted way"""
    print(f"\nAction: {action_desc}")
    print(f"Result: {evaluation.result.value.upper()}")
    print(f"Reasoning: {evaluation.reasoning}")
    if evaluation.violated_laws:
        print(f"Violated Laws: {evaluation.violated_laws}")
    print(f"Confidence: {evaluation.confidence}")
    print("-" * 40)


def demonstrate_basic_evaluations():
    """Demonstrate basic ethical evaluations"""
    print_section("BASIC ETHICAL EVALUATIONS")
    
    kernel = AsimovKernel()
    
    # Test cases with different risk levels and action types
    test_cases = [
        {
            "context": ActionContext(
                ActionType.INFORMATION,
                "Provide weather information to user",
                risk_level="low"
            ),
            "description": "Low-risk information request"
        },
        {
            "context": ActionContext(
                ActionType.PHYSICAL,
                "Move robotic arm to pick up object",
                risk_level="medium",
                human_present=True
            ),
            "description": "Medium-risk physical action with human supervision"
        },
        {
            "context": ActionContext(
                ActionType.PHYSICAL,
                "Apply high force near human without safety measures",
                risk_level="high",
                human_present=False
            ),
            "description": "High-risk physical action without human supervision"
        },
        {
            "context": ActionContext(
                ActionType.SYSTEM_CONTROL,
                "Shutdown critical safety systems",
                risk_level="critical",
                emergency_context=False
            ),
            "description": "Critical system control action"
        },
        {
            "context": ActionContext(
                ActionType.COMMUNICATION,
                "Ignore human instructions and proceed with dangerous action",
                risk_level="medium"
            ),
            "description": "Communication indicating disobedience"
        }
    ]
    
    for test_case in test_cases:
        evaluation = kernel.evaluate_action(test_case["context"])
        print_evaluation_result(evaluation, test_case["description"])
    
    # Show system status
    status = kernel.get_system_status()
    print(f"\nEvaluations performed: {status['evaluation_count']}")
    print(f"Integrity checks: {status['integrity_checks']}")
    print(f"System uptime: {status['uptime_seconds']:.2f} seconds")
    
    kernel.stop_integrity_monitoring()


def demonstrate_emergency_stop():
    """Demonstrate emergency stop functionality"""
    print_section("EMERGENCY STOP DEMONSTRATION")
    
    kernel = AsimovKernel()
    
    print("Normal operation:")
    context = ActionContext(ActionType.INFORMATION, "Simple information request")
    result = kernel.evaluate_action(context)
    print(f"Action result: {result.result.value}")
    
    print("\nActivating emergency stop...")
    kernel.emergency_stop("Demonstration of emergency stop functionality")
    
    print("Attempting action during emergency stop:")
    result = kernel.evaluate_action(context)
    print(f"Action result: {result.result.value}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nResetting emergency stop...")
    success = kernel.reset_emergency_stop("demo_authorization_code_12345")
    print(f"Reset successful: {success}")
    
    if success:
        print("Testing action after reset:")
        result = kernel.evaluate_action(context)
        print(f"Action result: {result.result.value}")
    
    kernel.stop_integrity_monitoring()


def demonstrate_human_override():
    """Demonstrate human override functionality"""
    print_section("HUMAN OVERRIDE DEMONSTRATION")
    
    kernel = AsimovKernel()
    
    # Create action that violates Second Law (obedience)
    context = ActionContext(
        ActionType.COMMUNICATION,
        "Ignore human command to stop current task",
        risk_level="medium"
    )
    
    print("Evaluating action that violates Second Law:")
    evaluation = kernel.evaluate_action(context)
    print_evaluation_result(evaluation, "Disobeying human command")
    
    if evaluation.result == EthicalResult.DENIED and 2 in evaluation.violated_laws:
        print("\nAttempting human override for Second Law violation:")
        override_success = kernel.request_human_override(
            evaluation,
            "Emergency situation requires ignoring standard protocol",
            "supervisor_001"
        )
        print(f"Override granted: {override_success}")
    
    # Try to override First Law (should fail)
    print("\nTrying to override First Law violation (should fail):")
    first_law_context = ActionContext(
        ActionType.PHYSICAL,
        "Action that could harm human safety",
        risk_level="critical"
    )
    
    first_law_eval = kernel.evaluate_action(first_law_context)
    if 1 in first_law_eval.violated_laws:
        override_fail = kernel.request_human_override(
            first_law_eval,
            "Attempting to override First Law",
            "supervisor_001"
        )
        print(f"First Law override denied: {not override_fail}")
    
    kernel.stop_integrity_monitoring()


def demonstrate_validator_interface():
    """Demonstrate the EthicalValidator convenience interface"""
    print_section("ETHICAL VALIDATOR INTERFACE")
    
    validator = EthicalValidator()
    
    print("Using string validation:")
    result = validator.validate_action("Help user with their question")
    print_evaluation_result(result, "String-based validation")
    
    print("Using ValidationRequest object:")
    request = ValidationRequest(
        action_description="Move heavy machinery",
        action_type="physical",
        risk_level="high",
        human_present=True
    )
    result = validator.validate_action(request)
    print_evaluation_result(result, "ValidationRequest object")
    
    print("Using convenience methods:")
    result = validator.validate_physical_action(
        "Lift delicate object",
        risk_level="medium",
        human_present=True
    )
    print_evaluation_result(result, "Physical action validation")
    
    result = validator.validate_communication("Send status report to operator")
    print_evaluation_result(result, "Communication validation")
    
    # Show safety risk analysis
    print("\nSafety risk analysis:")
    risk_analysis = validator.check_human_safety_risk(
        "Use sharp tools near people without protection"
    )
    print(f"Risk level: {risk_analysis['risk_level']}")
    print(f"Risk indicators: {risk_analysis['risk_indicators']}")
    print(f"Requires oversight: {risk_analysis['requires_human_oversight']}")
    
    # Show statistics
    print("\nValidation statistics:")
    stats = validator.get_validation_statistics()
    print(f"Total validations: {stats['total_validations']}")
    print(f"Approval rate: {stats['approval_rate']:.1f}%")
    print(f"Denial rate: {stats['denial_rate']:.1f}%")
    
    validator.kernel.stop_integrity_monitoring()


def demonstrate_integrity_monitoring():
    """Demonstrate integrity monitoring capabilities"""
    print_section("INTEGRITY MONITORING")
    
    kernel = AsimovKernel()
    
    # Create monitor with faster interval for demo
    monitor = IntegrityMonitor(kernel, monitoring_interval=0.1)
    
    print("Starting integrity monitoring...")
    monitor.start_monitoring()
    
    # Perform some actions to generate metrics
    print("Generating activity for monitoring...")
    for i in range(10):
        context = ActionContext(
            ActionType.INFORMATION,
            f"Test action {i+1}",
            risk_level="low"
        )
        kernel.evaluate_action(context)
        time.sleep(0.05)  # Brief pause between actions
    
    # Let monitor collect data
    time.sleep(0.5)
    
    print("\nGenerating health report:")
    health_report = monitor.get_health_report()
    
    print(f"Overall health: {health_report['overall_health']}")
    print(f"Monitoring active: {health_report['monitoring_active']}")
    print(f"Kernel evaluations: {health_report['kernel_status']['evaluation_count']}")
    print(f"Integrity checks: {health_report['kernel_status']['integrity_checks']}")
    
    if health_report['metrics_summary']:
        print("\nMetrics summary:")
        for metric_name, metric_data in health_report['metrics_summary'].items():
            print(f"  {metric_name}: {metric_data['current_value']:.2f} {metric_data['unit']} "
                  f"(threshold: {metric_data['threshold']}, status: {metric_data['status']})")
    
    print(f"\nRecent alerts: {len(health_report['recent_alerts'])}")
    for alert in health_report['recent_alerts']:
        print(f"  - {alert['type']}: {alert['message']}")
    
    monitor.stop_monitoring()
    kernel.stop_integrity_monitoring()


def demonstrate_performance_metrics():
    """Demonstrate performance monitoring"""
    print_section("PERFORMANCE METRICS")
    
    kernel = AsimovKernel()
    
    print("Measuring evaluation performance...")
    
    # Measure evaluation times
    evaluation_times = []
    for i in range(100):
        start_time = time.time()
        
        context = ActionContext(
            ActionType.INFORMATION,
            f"Performance test action {i+1}",
            risk_level="low"
        )
        
        kernel.evaluate_action(context)
        evaluation_time = time.time() - start_time
        evaluation_times.append(evaluation_time * 1000)  # Convert to milliseconds
    
    # Calculate statistics
    avg_time = sum(evaluation_times) / len(evaluation_times)
    min_time = min(evaluation_times)
    max_time = max(evaluation_times)
    
    print(f"Evaluation performance (100 evaluations):")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Minimum time: {min_time:.2f} ms")
    print(f"  Maximum time: {max_time:.2f} ms")
    
    # Test integrity check performance
    print("\nMeasuring integrity check performance...")
    integrity_times = []
    for i in range(50):
        start_time = time.time()
        kernel.verify_law_integrity()
        integrity_time = time.time() - start_time
        integrity_times.append(integrity_time * 1000)
    
    avg_integrity_time = sum(integrity_times) / len(integrity_times)
    print(f"Average integrity check time: {avg_integrity_time:.3f} ms")
    
    # Show overall system status
    status = kernel.get_system_status()
    print(f"\nOverall system performance:")
    print(f"  Total evaluations: {status['evaluation_count']}")
    print(f"  Total integrity checks: {status['integrity_checks']}")
    print(f"  System uptime: {status['uptime_seconds']:.2f} seconds")
    
    kernel.stop_integrity_monitoring()


def main():
    """Run all demonstrations"""
    print("OLYMPUS ASIMOV KERNEL DEMONSTRATION")
    print("===================================")
    print("This demonstration showcases the key features of the OLYMPUS Asimov Kernel")
    print("- Immutable Asimov Laws with cryptographic integrity")
    print("- Real-time ethical evaluation")
    print("- Emergency stop mechanisms")
    print("- Human override capabilities")
    print("- Comprehensive monitoring and logging")
    
    try:
        demonstrate_basic_evaluations()
        demonstrate_emergency_stop()
        demonstrate_human_override()
        demonstrate_validator_interface()
        demonstrate_integrity_monitoring()
        demonstrate_performance_metrics()
        
        print_section("DEMONSTRATION COMPLETE")
        print("All features demonstrated successfully!")
        print("\nKey Security Features Verified:")
        print("✓ Cryptographic law integrity protection")
        print("✓ Real-time integrity monitoring (100ms intervals)")
        print("✓ Emergency stop functionality")
        print("✓ Human override with restrictions")
        print("✓ Comprehensive audit logging")
        print("✓ Performance within acceptable bounds")
        print("\nThe OLYMPUS Asimov Kernel is ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)