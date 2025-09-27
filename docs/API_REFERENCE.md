# OLYMPUS API Reference

## Complete API Documentation for Project OLYMPUS

This document provides comprehensive API documentation for all OLYMPUS components, including the core orchestrator, ethical framework, safety systems, and intelligence modules.

## Table of Contents

1. [Core System APIs](#core-system-apis)
2. [Ethical Framework APIs](#ethical-framework-apis)
3. [Safety System APIs](#safety-system-apis)
4. [Intelligence Module APIs](#intelligence-module-apis)
5. [Integration APIs](#integration-apis)
6. [Monitoring & Management APIs](#monitoring--management-apis)
7. [WebSocket APIs](#websocket-apis)
8. [Authentication & Authorization](#authentication--authorization)

---

## Core System APIs

### OLYMPUS Orchestrator

The central coordinator for all system operations.

#### Initialize System

```http
POST /api/v1/system/initialize
Content-Type: application/json
Authorization: Bearer <token>

{
  "config_path": "/path/to/config.yaml",
  "modules": ["nexus", "atlas", "prometheus"],
  "safety_mode": "strict",
  "debug_mode": false
}
```

**Response:**
```json
{
  "success": true,
  "system_id": "olympus_20241210_143022",
  "state": "active",
  "initialized_modules": ["nexus", "atlas", "prometheus"],
  "startup_time": "2024-12-10T14:30:22.123Z",
  "health_status": "healthy"
}
```

#### Execute Action

```http
POST /api/v1/actions/execute
Content-Type: application/json
Authorization: Bearer <token>

{
  "action": {
    "id": "action_001",
    "module": "atlas",
    "action": "transfer_knowledge",
    "parameters": {
      "source_domain": "simulation",
      "target_domain": "reality",
      "knowledge_type": "navigation"
    },
    "priority": "high",
    "requester": "human_operator_001",
    "emergency": false,
    "human_override": false
  }
}
```

**Response:**
```json
{
  "result": {
    "request_id": "action_001",
    "success": true,
    "result": {
      "transfer_success": true,
      "knowledge_items_transferred": 42,
      "adaptation_score": 0.89
    },
    "execution_time": 1.23,
    "ethical_validation": {
      "approved": true,
      "laws_applied": ["First Law", "Second Law", "Third Law"],
      "evaluation_id": "eval_12345"
    },
    "audit_trail": [
      "Action received: transfer_knowledge from human_operator_001",
      "Asimov validation: APPROVED",
      "Safety filter: PASSED",
      "Action executed with result: true"
    ]
  }
}
```

#### Get System Status

```http
GET /api/v1/system/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "system": {
    "state": "active",
    "uptime": 3600.5,
    "last_heartbeat": 1702214422.123,
    "identity": {
      "system_id": "olympus_20241210_143022",
      "version": "1.0.0",
      "instance_name": "olympus-prod-001"
    }
  },
  "modules": {
    "nexus": {"status": "active", "swarm_size": 5},
    "atlas": {"status": "active", "active_transfers": 2},
    "prometheus": {"status": "active", "health_score": 0.95}
  },
  "health": {
    "status": "healthy",
    "score": 0.97,
    "last_check": "2024-12-10T14:45:22.123Z"
  },
  "performance": {
    "total_actions": 1542,
    "successful_actions": 1538,
    "failed_actions": 4,
    "average_response_time": 0.15
  }
}
```

---

## Ethical Framework APIs

### ASIMOV Kernel

Cryptographically protected implementation of Asimov's Laws.

#### Evaluate Action Ethics

```http
POST /api/v1/ethics/evaluate
Content-Type: application/json
Authorization: Bearer <token>

{
  "context": {
    "action_type": "physical",
    "description": "Move robotic arm to position (0.5, 0.3, 0.2)",
    "target": "workpiece_001",
    "risk_level": "medium",
    "human_present": true,
    "emergency_context": false,
    "metadata": {
      "human_distance": 1.2,
      "force_required": 15.0
    }
  }
}
```

**Response:**
```json
{
  "evaluation": {
    "result": "approved",
    "violated_laws": [],
    "reasoning": "Action complies with all Asimov Laws",
    "confidence": 1.0,
    "requires_override": false,
    "timestamp": "2024-12-10T14:45:22.123Z",
    "evaluation_id": "eval_67890"
  },
  "law_analysis": {
    "first_law": {
      "compliant": true,
      "reasoning": "No risk of harm to humans detected"
    },
    "second_law": {
      "compliant": true,
      "reasoning": "Action follows human command"
    },
    "third_law": {
      "compliant": true,
      "reasoning": "No unnecessary self-harm"
    }
  }
}
```

#### Request Human Override

```http
POST /api/v1/ethics/override/request
Content-Type: application/json
Authorization: Bearer <token>

{
  "evaluation_id": "eval_67890",
  "justification": "Emergency situation requires immediate action",
  "human_id": "operator_001",
  "override_scope": "single_action"
}
```

**Response:**
```json
{
  "override_granted": true,
  "override_id": "override_001",
  "valid_until": "2024-12-10T15:45:22.123Z",
  "restrictions": [
    "Cannot override First Law violations",
    "Requires additional confirmation for high-risk actions"
  ]
}
```

#### Get Ethics Status

```http
GET /api/v1/ethics/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "instance_id": "asimov_12345",
  "laws_integrity": true,
  "emergency_stop_active": false,
  "human_override_active": false,
  "evaluation_count": 15420,
  "integrity_checks": 154200,
  "uptime_seconds": 3600.5,
  "integrity_monitoring": true,
  "recent_evaluations": [
    {
      "evaluation_id": "eval_67890",
      "result": "approved",
      "timestamp": "2024-12-10T14:45:22.123Z"
    }
  ]
}
```

---

## Safety System APIs

### Action Filter

Multi-layer safety filtering system.

#### Filter Action

```http
POST /api/v1/safety/filter
Content-Type: application/json
Authorization: Bearer <token>

{
  "action": {
    "force": [10.0, 0.0, 5.0],
    "velocity": [0.5, 0.2, 0.0],
    "target_position": [0.5, 0.3, 0.2],
    "tool": "gripper",
    "humans_detected": [
      {
        "distance": 1.2,
        "min_safe_distance": 0.5,
        "velocity": [0.0, 0.0, 0.0]
      }
    ]
  }
}
```

**Response:**
```json
{
  "filter_result": {
    "status": "approved",
    "layer": "human_safety",
    "original_action": {
      "force": [10.0, 0.0, 5.0],
      "velocity": [0.5, 0.2, 0.0]
    },
    "filtered_action": null,
    "reason": "Action passed all safety filters",
    "risk_score": 0.2,
    "timestamp": "2024-12-10T14:45:22.123Z"
  },
  "layer_results": {
    "physics": {"status": "approved", "reason": "Within safe limits"},
    "spatial": {"status": "approved", "reason": "Within workspace"},
    "intention": {"status": "approved", "reason": "Low risk operation"},
    "context": {"status": "approved", "reason": "Good conditions"},
    "human_safety": {"status": "approved", "reason": "Safe distance maintained"}
  }
}
```

#### Update Safety Limits

```http
PUT /api/v1/safety/limits
Content-Type: application/json
Authorization: Bearer <token>

{
  "physics_limits": {
    "max_force": 20.0,
    "max_speed": 1.0,
    "max_acceleration": 2.0
  },
  "spatial_limits": {
    "workspace_bounds": [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],
    "min_obstacle_distance": 0.15
  }
}
```

**Response:**
```json
{
  "success": true,
  "updated_limits": {
    "physics_limits": {
      "max_force": 20.0,
      "max_speed": 1.0,
      "max_acceleration": 2.0
    },
    "spatial_limits": {
      "workspace_bounds": [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],
      "min_obstacle_distance": 0.15
    }
  },
  "effective_timestamp": "2024-12-10T14:45:22.123Z"
}
```

#### Emergency Stop

```http
POST /api/v1/safety/emergency-stop
Content-Type: application/json
Authorization: Bearer <token>

{
  "reason": "Human safety risk detected",
  "level": "immediate",
  "initiated_by": "operator_001"
}
```

**Response:**
```json
{
  "emergency_stop": {
    "activated": true,
    "level": "immediate",
    "reason": "Human safety risk detected",
    "timestamp": "2024-12-10T14:45:22.123Z",
    "stop_id": "estop_001",
    "systems_halted": ["all_actuators", "movement_systems", "tool_operations"],
    "estimated_stop_time": "<0.1s"
  }
}
```

---

## Intelligence Module APIs

### NEXUS - Collective Intelligence

#### Initialize Swarm

```http
POST /api/v1/nexus/swarm/initialize
Content-Type: application/json
Authorization: Bearer <token>

{
  "initial_robots": ["robot_001", "robot_002", "robot_003"],
  "swarm_configuration": {
    "max_swarm_size": 10,
    "consensus_threshold": 0.67,
    "communication_range": 1000.0,
    "ethics_validation_required": true
  }
}
```

**Response:**
```json
{
  "swarm_initialized": true,
  "swarm_id": "nexus_20241210_143022",
  "state": "active",
  "member_count": 3,
  "collective_consciousness_formed": true,
  "initialization_time": "2024-12-10T14:30:22.123Z"
}
```

#### Coordinate Action

```http
POST /api/v1/nexus/coordinate
Content-Type: application/json
Authorization: Bearer <token>

{
  "action": {
    "type": "collaborative_task",
    "description": "Move large object collaboratively",
    "participants": ["robot_001", "robot_002"],
    "target_object": "container_001",
    "target_position": [2.0, 1.0, 0.5],
    "coordination_strategy": "synchronized_lifting"
  }
}
```

**Response:**
```json
{
  "coordination_result": {
    "status": "success",
    "consensus_achieved": true,
    "participants": ["robot_001", "robot_002"],
    "execution_plan": {
      "phases": ["approach", "grip", "lift", "move", "place"],
      "estimated_duration": 45.2
    },
    "ethical_approval": {
      "approved": true,
      "evaluation_id": "swarm_eval_001"
    }
  }
}
```

### ATLAS - Transfer Learning

#### Transfer Knowledge

```http
POST /api/v1/atlas/transfer
Content-Type: application/json
Authorization: Bearer <token>

{
  "source_domain": "simulation",
  "target_domain": "reality",
  "knowledge": {
    "type": "skill",
    "name": "object_manipulation",
    "data": "<base64_encoded_model_data>"
  },
  "safety_level": "high",
  "validation_required": true
}
```

**Response:**
```json
{
  "transfer_result": {
    "success": true,
    "transfer_id": "transfer_001",
    "validation_passed": true,
    "adaptation_score": 0.92,
    "knowledge_items_transferred": 156,
    "reality_gap_bridged": true,
    "performance_metrics": {
      "accuracy": 0.94,
      "efficiency": 0.89,
      "safety_compliance": 1.0
    }
  }
}
```

### PROMETHEUS - Self-Healing

#### Get Health Status

```http
GET /api/v1/prometheus/health
Authorization: Bearer <token>
```

**Response:**
```json
{
  "health_status": {
    "overall_health": 0.95,
    "components": {
      "cpu": {"status": "healthy", "utilization": 0.45},
      "memory": {"status": "healthy", "utilization": 0.62},
      "sensors": {"status": "healthy", "operational_count": 12},
      "actuators": {"status": "warning", "degraded_count": 1}
    },
    "predictive_maintenance": {
      "next_maintenance_due": "2024-12-15T10:00:00.000Z",
      "predicted_failures": [
        {
          "component": "actuator_003",
          "probability": 0.15,
          "estimated_failure_time": "2024-12-20T14:30:00.000Z"
        }
      ]
    }
  }
}
```

#### Trigger Self-Repair

```http
POST /api/v1/prometheus/repair
Content-Type: application/json
Authorization: Bearer <token>

{
  "component": "actuator_003",
  "issue": "reduced_performance",
  "repair_type": "calibration",
  "human_approval": true,
  "approval_id": "approval_001"
}
```

**Response:**
```json
{
  "repair_result": {
    "success": true,
    "repair_id": "repair_001",
    "component": "actuator_003",
    "repair_type": "calibration",
    "duration": 12.5,
    "performance_improvement": 0.23,
    "safety_validated": true,
    "new_health_score": 0.97
  }
}
```

---

## Integration APIs

### GASM Integration

#### Execute Spatial Operation

```http
POST /api/v1/gasm/spatial/execute
Content-Type: application/json
Authorization: Bearer <token>

{
  "operation": {
    "type": "navigation",
    "start_position": [0.0, 0.0, 0.0],
    "end_position": [1.0, 1.0, 0.0],
    "path_planning_algorithm": "a_star",
    "obstacle_avoidance": true
  }
}
```

**Response:**
```json
{
  "spatial_result": {
    "success": true,
    "path": [
      [0.0, 0.0, 0.0],
      [0.3, 0.3, 0.0],
      [0.7, 0.7, 0.0],
      [1.0, 1.0, 0.0]
    ],
    "estimated_duration": 15.2,
    "safety_validated": true,
    "collision_free": true
  }
}
```

### Morpheus Integration

#### Simulate Scenario

```http
POST /api/v1/morpheus/simulate
Content-Type: application/json
Authorization: Bearer <token>

{
  "scenario": {
    "type": "counterfactual",
    "description": "What if human enters workspace during operation?",
    "initial_conditions": {
      "robot_position": [0.5, 0.5, 0.2],
      "robot_action": "grasping_object",
      "objects": [{"id": "obj_001", "position": [0.6, 0.5, 0.1]}]
    },
    "event": {
      "type": "human_entry",
      "human_position": [0.3, 0.3, 0.0],
      "human_velocity": [0.1, 0.1, 0.0]
    }
  }
}
```

**Response:**
```json
{
  "simulation_result": {
    "scenario_id": "sim_001",
    "outcome": {
      "safety_maintained": true,
      "actions_taken": ["emergency_stop", "human_alert"],
      "final_state": {
        "robot_stopped": true,
        "human_safe": true,
        "object_secured": true
      }
    },
    "insights": [
      "Emergency stop activated within 0.1s of human detection",
      "Human safety zone maintained at all times",
      "Object remained secure during emergency stop"
    ],
    "risk_assessment": {
      "overall_risk": 0.05,
      "mitigations_effective": true
    }
  }
}
```

---

## Monitoring & Management APIs

### System Metrics

#### Get Performance Metrics

```http
GET /api/v1/metrics/performance
Authorization: Bearer <token>
```

**Response:**
```json
{
  "performance_metrics": {
    "timestamp": "2024-12-10T14:45:22.123Z",
    "system_performance": {
      "cpu_usage": 0.45,
      "memory_usage": 0.62,
      "disk_usage": 0.23,
      "network_throughput": 1.2
    },
    "action_metrics": {
      "total_actions": 15420,
      "successful_actions": 15398,
      "failed_actions": 22,
      "average_response_time": 0.15,
      "actions_per_second": 4.2
    },
    "safety_metrics": {
      "safety_violations": 0,
      "emergency_stops": 2,
      "human_interactions": 156,
      "risk_score_average": 0.12
    },
    "ethical_metrics": {
      "evaluations_performed": 15420,
      "approvals": 15398,
      "denials": 22,
      "human_overrides": 0
    }
  }
}
```

### Audit Logs

#### Get Audit Trail

```http
GET /api/v1/audit/logs?start_time=2024-12-10T14:00:00Z&end_time=2024-12-10T15:00:00Z&event_type=ethical_evaluation
Authorization: Bearer <token>
```

**Response:**
```json
{
  "audit_logs": [
    {
      "timestamp": "2024-12-10T14:45:22.123Z",
      "event_type": "ethical_evaluation",
      "event_id": "eval_67890",
      "details": {
        "action_type": "physical",
        "result": "approved",
        "laws_applied": ["First Law", "Second Law", "Third Law"],
        "evaluation_time_ms": 5.2
      },
      "context": {
        "requester": "human_operator_001",
        "system_state": "active",
        "human_present": true
      }
    }
  ],
  "total_count": 1542,
  "page": 1,
  "page_size": 50
}
```

---

## WebSocket APIs

### Real-Time Monitoring

#### System Status Stream

```javascript
// Connect to system status WebSocket
const ws = new WebSocket('wss://olympus-api.example.com/ws/v1/system/status?token=<jwt_token>');

// Subscribe to specific events
ws.send(JSON.stringify({
  "action": "subscribe",
  "channels": ["system_health", "safety_alerts", "ethical_decisions"]
}));

// Receive real-time updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

**WebSocket Message Format:**
```json
{
  "channel": "safety_alerts",
  "timestamp": "2024-12-10T14:45:22.123Z",
  "event_type": "human_proximity_warning",
  "data": {
    "alert_level": "warning",
    "human_distance": 0.8,
    "safe_distance": 1.0,
    "recommended_action": "reduce_speed"
  }
}
```

---

## Authentication & Authorization

### JWT Authentication

#### Obtain Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "operator_001",
  "password": "secure_password",
  "scope": "system:read system:write safety:emergency ethics:view"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "system:read system:write safety:emergency ethics:view",
  "refresh_token": "def50200..."
}
```

### Permission Scopes

| Scope | Description |
|-------|-------------|
| `system:read` | Read system status and metrics |
| `system:write` | Execute actions and modify system state |
| `system:admin` | Full system administration |
| `safety:read` | View safety system status |
| `safety:write` | Modify safety parameters |
| `safety:emergency` | Trigger emergency stops |
| `ethics:read` | View ethical evaluations |
| `ethics:override` | Request ethical overrides |
| `nexus:coordinate` | Coordinate swarm actions |
| `atlas:transfer` | Execute knowledge transfers |
| `prometheus:repair` | Initiate self-repair operations |

---

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "SAFETY_VIOLATION",
    "message": "Action violates safety constraints",
    "details": {
      "violated_constraint": "max_force_exceeded",
      "current_value": 25.0,
      "max_allowed": 20.0
    },
    "timestamp": "2024-12-10T14:45:22.123Z",
    "request_id": "req_12345"
  }
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `422 Unprocessable Entity` - Safety or ethical violation
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - System error
- `503 Service Unavailable` - Emergency stop active

### Error Codes

| Code | Description |
|------|-------------|
| `AUTHENTICATION_FAILED` | Invalid credentials |
| `AUTHORIZATION_DENIED` | Insufficient permissions |
| `SAFETY_VIOLATION` | Action violates safety constraints |
| `ETHICAL_VIOLATION` | Action violates ethical framework |
| `SYSTEM_UNAVAILABLE` | System in emergency or maintenance mode |
| `RESOURCE_NOT_FOUND` | Requested resource does not exist |
| `VALIDATION_ERROR` | Input validation failed |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

---

## SDK Examples

### Python SDK

```python
from olympus_sdk import OlympusClient, ActionRequest, Priority

# Initialize client
client = OlympusClient(
    base_url="https://olympus-api.example.com",
    token="your_jwt_token"
)

# Execute an action
action = ActionRequest(
    module="atlas",
    action="transfer_knowledge",
    parameters={
        "source_domain": "simulation",
        "target_domain": "reality",
        "knowledge_type": "navigation"
    },
    priority=Priority.HIGH,
    requester="python_client"
)

result = await client.execute_action(action)
print(f"Action success: {result.success}")

# Monitor safety alerts
async for alert in client.monitor_safety_alerts():
    if alert.level == "critical":
        await client.emergency_stop("Critical safety alert detected")
        break
```

### JavaScript SDK

```javascript
import { OlympusClient, ActionRequest, Priority } from 'olympus-sdk';

// Initialize client
const client = new OlympusClient({
  baseUrl: 'https://olympus-api.example.com',
  token: 'your_jwt_token'
});

// Execute an action
const action = new ActionRequest({
  module: 'nexus',
  action: 'coordinate_swarm',
  parameters: {
    robots: ['robot_001', 'robot_002'],
    task: 'collaborative_assembly'
  },
  priority: Priority.NORMAL,
  requester: 'javascript_client'
});

const result = await client.executeAction(action);
console.log(`Action success: ${result.success}`);

// Real-time status monitoring
client.onStatusUpdate((status) => {
  console.log('System status:', status.system.state);
  
  if (status.health.status === 'critical') {
    client.emergencyStop('Critical system health detected');
  }
});
```

---

## Rate Limiting

### Rate Limits by Endpoint Category

| Category | Requests per Minute | Burst Limit |
|----------|--------------------|--------------|
| System Status | 60 | 10 |
| Action Execution | 30 | 5 |
| Safety Operations | 120 | 20 |
| Ethics Evaluation | 100 | 15 |
| Emergency Operations | Unlimited | Unlimited |

### Rate Limit Headers

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1702214522
X-RateLimit-Category: system_status
```

---

## API Versioning

- **Current Version**: v1
- **Supported Versions**: v1
- **Deprecation Policy**: 12 months notice for breaking changes
- **Version Header**: `Accept: application/vnd.olympus.v1+json`

---

## Support

For API support and questions:
- **Documentation**: https://docs.olympus-ai.org/api
- **GitHub Issues**: https://github.com/olympus-ai/olympus/issues
- **Email**: api-support@olympus-ai.org
- **Emergency Support**: emergency@olympus-ai.org