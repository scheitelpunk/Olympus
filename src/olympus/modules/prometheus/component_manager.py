"""
Component Manager - Lifecycle Management and State Tracking

The Component Manager tracks the lifecycle, state, and dependencies of all
system components. It provides centralized component management for the
PROMETHEUS self-healing system, enabling intelligent repair decisions and
dependency-aware maintenance operations.

Key Features:
- Component lifecycle tracking (birth to death)
- Real-time state monitoring and management
- Dependency mapping and cascade analysis
- Component health scoring and trending
- Resource usage tracking per component
- Component versioning and configuration management
- Integration with repair and redundancy systems
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json


class ComponentState(Enum):
    """Possible states for system components."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    REPAIRING = "repairing"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    CORE_SERVICE = "core_service"
    SUPPORT_SERVICE = "support_service"
    DATABASE = "database"
    NETWORK = "network"
    STORAGE = "storage"
    SECURITY = "security"
    MONITORING = "monitoring"
    API = "api"
    UI = "ui"
    EXTERNAL_DEPENDENCY = "external_dependency"


class DependencyType(Enum):
    """Types of component dependencies."""
    HARD_DEPENDENCY = "hard_dependency"  # Component cannot function without this
    SOFT_DEPENDENCY = "soft_dependency"  # Component can function with reduced capability
    OPTIONAL_DEPENDENCY = "optional_dependency"  # Component can function normally without this
    CIRCULAR_DEPENDENCY = "circular_dependency"  # Mutual dependency


@dataclass
class ComponentInfo:
    """Comprehensive information about a system component."""
    component_id: str
    name: str
    component_type: ComponentType
    version: str
    state: ComponentState
    health_score: float  # 0.0 to 1.0
    
    # Lifecycle information
    created_time: float
    last_updated: float
    uptime: float
    restart_count: int = 0
    
    # Configuration and resources
    configuration: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Health and performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    
    # Maintenance information
    last_maintenance: Optional[float] = None
    next_maintenance: Optional[float] = None
    maintenance_interval: Optional[float] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component info to dictionary."""
        return {
            "component_id": self.component_id,
            "name": self.name,
            "component_type": self.component_type.value,
            "version": self.version,
            "state": self.state.value,
            "health_score": self.health_score,
            "created_time": self.created_time,
            "last_updated": self.last_updated,
            "uptime": self.uptime,
            "restart_count": self.restart_count,
            "configuration": self.configuration,
            "resource_limits": self.resource_limits,
            "resource_usage": self.resource_usage,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "performance_metrics": self.performance_metrics,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "last_maintenance": self.last_maintenance,
            "next_maintenance": self.next_maintenance,
            "maintenance_interval": self.maintenance_interval,
            "tags": list(self.tags),
            "metadata": self.metadata
        }


@dataclass
class ComponentDependency:
    """Represents a dependency relationship between components."""
    source_component: str
    target_component: str
    dependency_type: DependencyType
    strength: float  # 0.0 to 1.0, how critical this dependency is
    created_time: float
    last_validated: Optional[float] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dependency to dictionary."""
        return {
            "source_component": self.source_component,
            "target_component": self.target_component,
            "dependency_type": self.dependency_type.value,
            "strength": self.strength,
            "created_time": self.created_time,
            "last_validated": self.last_validated,
            "is_active": self.is_active
        }


class ComponentHealthCalculator:
    """Calculates health scores for components based on various metrics."""
    
    def __init__(self):
        self.weight_factors = {
            "performance": 0.3,
            "errors": 0.25,
            "resource_usage": 0.2,
            "uptime": 0.15,
            "dependencies": 0.1
        }
    
    def calculate_health_score(self, component: ComponentInfo, 
                             dependency_health: Dict[str, float] = None) -> float:
        """Calculate overall health score for a component."""
        try:
            scores = {}
            
            # Performance score (based on response time, throughput, etc.)
            scores["performance"] = self._calculate_performance_score(component)
            
            # Error score (based on error rates and counts)
            scores["errors"] = self._calculate_error_score(component)
            
            # Resource usage score (CPU, memory, disk usage)
            scores["resource_usage"] = self._calculate_resource_score(component)
            
            # Uptime score (based on availability and restart frequency)
            scores["uptime"] = self._calculate_uptime_score(component)
            
            # Dependency score (health of dependencies)
            scores["dependencies"] = self._calculate_dependency_score(
                component, dependency_health or {}
            )
            
            # Weighted average
            total_score = sum(
                scores[factor] * weight 
                for factor, weight in self.weight_factors.items()
            )
            
            return max(0.0, min(1.0, total_score))
            
        except Exception:
            return 0.5  # Default to medium health if calculation fails
    
    def _calculate_performance_score(self, component: ComponentInfo) -> float:
        """Calculate performance-based health score."""
        try:
            metrics = component.performance_metrics
            
            # Default high score if no performance data
            if not metrics:
                return 0.8
            
            score = 1.0
            
            # Response time penalty
            response_time = metrics.get("response_time", 0)
            if response_time > 0:
                # Penalty increases exponentially with response time
                penalty = min(response_time / 1000, 1.0)  # 1 second = full penalty
                score -= penalty * 0.5
            
            # Throughput bonus/penalty
            throughput = metrics.get("throughput", 0)
            expected_throughput = metrics.get("expected_throughput", 100)
            if expected_throughput > 0:
                throughput_ratio = throughput / expected_throughput
                if throughput_ratio < 0.5:  # Less than 50% expected
                    score -= 0.3
                elif throughput_ratio > 1.2:  # More than 120% expected (good)
                    score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_error_score(self, component: ComponentInfo) -> float:
        """Calculate error-based health score."""
        try:
            # Start with perfect score
            score = 1.0
            
            # Error count penalty
            if component.error_count > 0:
                # Logarithmic penalty for errors
                error_penalty = min(0.1 * math.log(component.error_count + 1), 0.8)
                score -= error_penalty
            
            # Warning count penalty (less severe)
            if component.warning_count > 0:
                warning_penalty = min(0.05 * math.log(component.warning_count + 1), 0.3)
                score -= warning_penalty
            
            # Error rate penalty (if available)
            error_rate = component.performance_metrics.get("error_rate", 0)
            if error_rate > 0:
                # Penalty for error rates > 1%
                if error_rate > 0.01:
                    score -= min((error_rate - 0.01) * 10, 0.5)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_resource_score(self, component: ComponentInfo) -> float:
        """Calculate resource usage-based health score."""
        try:
            usage = component.resource_usage
            limits = component.resource_limits
            
            if not usage:
                return 0.8  # Default good score if no resource data
            
            score = 1.0
            resource_penalties = []
            
            # Check CPU usage
            cpu_usage = usage.get("cpu_percent", 0)
            cpu_limit = limits.get("cpu_percent", 100)
            cpu_ratio = cpu_usage / cpu_limit if cpu_limit > 0 else 0
            
            if cpu_ratio > 0.9:  # Over 90% of limit
                resource_penalties.append(0.4)
            elif cpu_ratio > 0.8:  # Over 80% of limit
                resource_penalties.append(0.2)
            
            # Check Memory usage
            memory_usage = usage.get("memory_percent", 0)
            memory_limit = limits.get("memory_percent", 100)
            memory_ratio = memory_usage / memory_limit if memory_limit > 0 else 0
            
            if memory_ratio > 0.9:
                resource_penalties.append(0.4)
            elif memory_ratio > 0.8:
                resource_penalties.append(0.2)
            
            # Check Disk usage
            disk_usage = usage.get("disk_percent", 0)
            disk_limit = limits.get("disk_percent", 100)
            disk_ratio = disk_usage / disk_limit if disk_limit > 0 else 0
            
            if disk_ratio > 0.95:  # Disk is more critical
                resource_penalties.append(0.5)
            elif disk_ratio > 0.85:
                resource_penalties.append(0.3)
            
            # Apply worst penalty (resources are often correlated)
            if resource_penalties:
                score -= max(resource_penalties)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_uptime_score(self, component: ComponentInfo) -> float:
        """Calculate uptime-based health score."""
        try:
            current_time = time.time()
            
            # Base uptime score
            uptime_hours = component.uptime / 3600 if component.uptime > 0 else 0
            
            # Bonus for long uptime (up to 1 week)
            max_uptime_bonus_hours = 24 * 7  # 1 week
            uptime_bonus = min(uptime_hours / max_uptime_bonus_hours, 1.0) * 0.2
            
            # Penalty for frequent restarts
            restart_penalty = 0
            if component.restart_count > 0:
                # Penalty increases with restart frequency
                hours_since_creation = (current_time - component.created_time) / 3600
                restart_frequency = component.restart_count / max(hours_since_creation, 1)
                
                if restart_frequency > 1:  # More than 1 restart per hour
                    restart_penalty = 0.5
                elif restart_frequency > 0.1:  # More than 1 restart per 10 hours
                    restart_penalty = 0.3
                elif restart_frequency > 0.01:  # More than 1 restart per 100 hours
                    restart_penalty = 0.1
            
            score = 0.8 + uptime_bonus - restart_penalty
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_dependency_score(self, component: ComponentInfo, 
                                   dependency_health: Dict[str, float]) -> float:
        """Calculate score based on dependency health."""
        try:
            if not component.dependencies:
                return 1.0  # Perfect score if no dependencies
            
            dependency_scores = []
            for dep_id in component.dependencies:
                dep_health = dependency_health.get(dep_id, 0.5)  # Default to medium
                dependency_scores.append(dep_health)
            
            if dependency_scores:
                # Use average dependency health, but weight critical deps more
                return sum(dependency_scores) / len(dependency_scores)
            
            return 1.0
            
        except Exception:
            return 0.5


class DependencyAnalyzer:
    """Analyzes component dependencies and detects potential issues."""
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        self.dependency_cache = {}
        self.circular_dependency_cache = set()
    
    async def analyze_dependencies(self, components: Dict[str, ComponentInfo],
                                 dependencies: List[ComponentDependency]) -> Dict[str, Any]:
        """Perform comprehensive dependency analysis."""
        try:
            analysis = {
                "total_components": len(components),
                "total_dependencies": len(dependencies),
                "circular_dependencies": [],
                "critical_paths": [],
                "single_points_of_failure": [],
                "dependency_depth": {},
                "component_criticality": {}
            }
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(dependencies)
            
            # Detect circular dependencies
            circular_deps = await self._detect_circular_dependencies(dependency_graph)
            analysis["circular_dependencies"] = circular_deps
            
            # Find critical paths
            critical_paths = await self._find_critical_paths(dependency_graph, components)
            analysis["critical_paths"] = critical_paths
            
            # Identify single points of failure
            spofs = await self._identify_single_points_of_failure(dependency_graph, components)
            analysis["single_points_of_failure"] = spofs
            
            # Calculate dependency depth for each component
            for comp_id in components:
                depth = await self._calculate_dependency_depth(comp_id, dependency_graph)
                analysis["dependency_depth"][comp_id] = depth
            
            # Calculate component criticality
            criticality = await self._calculate_component_criticality(dependency_graph, components)
            analysis["component_criticality"] = criticality
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_analysis_completed",
                    {
                        "components_analyzed": len(components),
                        "circular_dependencies_found": len(circular_deps),
                        "single_points_of_failure": len(spofs)
                    }
                )
            
            return analysis
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_analysis_failed",
                    {"error": str(e)}
                )
            raise
    
    def _build_dependency_graph(self, dependencies: List[ComponentDependency]) -> Dict[str, List[str]]:
        """Build a dependency graph from dependency list."""
        graph = defaultdict(list)
        
        for dep in dependencies:
            if dep.is_active:
                graph[dep.source_component].append(dep.target_component)
        
        return dict(graph)
    
    async def _detect_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph."""
        visited = set()
        rec_stack = set()
        circular_deps = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle, extract it
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if cycle not in circular_deps:
                    circular_deps.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return circular_deps
    
    async def _find_critical_paths(self, graph: Dict[str, List[str]], 
                                 components: Dict[str, ComponentInfo]) -> List[List[str]]:
        """Find critical paths in the dependency graph."""
        critical_paths = []
        
        # Find paths that involve critical components
        critical_components = [
            comp_id for comp_id, comp in components.items()
            if comp.component_type == ComponentType.CORE_SERVICE or
               comp.health_score < 0.5
        ]
        
        for critical_comp in critical_components:
            # Find all paths leading to this critical component
            paths = self._find_all_paths_to_component(graph, critical_comp)
            
            # Filter for paths longer than 2 components
            long_paths = [path for path in paths if len(path) > 2]
            critical_paths.extend(long_paths)
        
        return critical_paths[:10]  # Limit to top 10
    
    def _find_all_paths_to_component(self, graph: Dict[str, List[str]], 
                                   target: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths leading to a specific component."""
        paths = []
        
        def dfs(current, path, depth):
            if depth > max_depth:
                return
            
            if current == target and len(path) > 1:
                paths.append(path.copy())
                return
            
            for neighbor in graph.get(current, []):
                if neighbor not in path:  # Avoid cycles
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
        
        # Start DFS from all nodes
        for start_node in graph:
            if start_node != target:
                dfs(start_node, [start_node], 0)
        
        return paths
    
    async def _identify_single_points_of_failure(self, graph: Dict[str, List[str]],
                                               components: Dict[str, ComponentInfo]) -> List[str]:
        """Identify components that are single points of failure."""
        spofs = []
        
        for comp_id, comp in components.items():
            # Check if this component has many dependents but no alternatives
            dependents = [c_id for c_id, c in components.items() if comp_id in c.dependencies]
            
            if len(dependents) > 2:  # Has multiple dependents
                # Check if dependents have alternative dependencies
                has_alternatives = True
                for dependent_id in dependents:
                    dependent_comp = components[dependent_id]
                    alternative_deps = [dep for dep in dependent_comp.dependencies if dep != comp_id]
                    
                    if len(alternative_deps) == 0:  # No alternative dependencies
                        has_alternatives = False
                        break
                
                if not has_alternatives:
                    spofs.append(comp_id)
        
        return spofs
    
    async def _calculate_dependency_depth(self, comp_id: str, 
                                        graph: Dict[str, List[str]]) -> int:
        """Calculate the maximum dependency depth for a component."""
        visited = set()
        
        def dfs(node, depth):
            if node in visited:
                return depth
            
            visited.add(node)
            max_depth = depth
            
            for neighbor in graph.get(node, []):
                neighbor_depth = dfs(neighbor, depth + 1)
                max_depth = max(max_depth, neighbor_depth)
            
            return max_depth
        
        return dfs(comp_id, 0)
    
    async def _calculate_component_criticality(self, graph: Dict[str, List[str]],
                                             components: Dict[str, ComponentInfo]) -> Dict[str, float]:
        """Calculate criticality score for each component."""
        criticality_scores = {}
        
        for comp_id, comp in components.items():
            score = 0.0
            
            # Base criticality from component type
            type_scores = {
                ComponentType.CORE_SERVICE: 1.0,
                ComponentType.SECURITY: 0.9,
                ComponentType.DATABASE: 0.8,
                ComponentType.API: 0.7,
                ComponentType.NETWORK: 0.6,
                ComponentType.SUPPORT_SERVICE: 0.5,
                ComponentType.STORAGE: 0.4,
                ComponentType.MONITORING: 0.3,
                ComponentType.UI: 0.2,
                ComponentType.EXTERNAL_DEPENDENCY: 0.1
            }
            score += type_scores.get(comp.component_type, 0.5)
            
            # Add score based on number of dependents
            dependents = [c_id for c_id, c in components.items() if comp_id in c.dependencies]
            score += min(len(dependents) * 0.1, 0.5)  # Max 0.5 bonus for dependents
            
            # Reduce score based on health
            score *= comp.health_score
            
            criticality_scores[comp_id] = min(score, 2.0)  # Cap at 2.0
        
        return criticality_scores


class ComponentManager:
    """
    Main component manager that coordinates component lifecycle and state management.
    """
    
    def __init__(self, audit_system=None):
        self.audit_system = audit_system
        
        # Core data structures
        self.components = {}  # component_id -> ComponentInfo
        self.dependencies = {}  # dependency_id -> ComponentDependency
        self.component_history = defaultdict(lambda: deque(maxlen=100))  # component_id -> history
        
        # Analysis components
        self.health_calculator = ComponentHealthCalculator()
        self.dependency_analyzer = DependencyAnalyzer(audit_system)
        
        # System state
        self.is_active = False
        self.update_interval = 30.0  # seconds
        self.update_task = None
        
        # Caches and optimizations
        self.dependency_graph_cache = None
        self.last_dependency_analysis = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the component manager."""
        try:
            self.is_active = True
            
            # Start periodic update task
            self.update_task = asyncio.create_task(self._update_loop())
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_manager_initialized",
                    {"update_interval": self.update_interval}
                )
            
            self.logger.info("Component manager initialized")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_manager_init_failed",
                    {"error": str(e)}
                )
            raise
    
    async def shutdown(self):
        """Shutdown the component manager."""
        try:
            self.is_active = False
            
            if self.update_task:
                self.update_task.cancel()
                try:
                    await self.update_task
                except asyncio.CancelledError:
                    pass
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_manager_shutdown",
                    {
                        "total_components": len(self.components),
                        "total_dependencies": len(self.dependencies)
                    }
                )
            
            self.logger.info("Component manager shut down")
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_manager_shutdown_failed",
                    {"error": str(e)}
                )
            raise
    
    async def register_component(self, component_id: str, name: str, 
                               component_type: ComponentType, version: str = "1.0.0",
                               configuration: Dict[str, Any] = None) -> bool:
        """Register a new component with the manager."""
        try:
            if component_id in self.components:
                self.logger.warning(f"Component {component_id} already exists")
                return False
            
            current_time = time.time()
            
            component = ComponentInfo(
                component_id=component_id,
                name=name,
                component_type=component_type,
                version=version,
                state=ComponentState.INITIALIZING,
                health_score=0.5,
                created_time=current_time,
                last_updated=current_time,
                uptime=0,
                configuration=configuration or {}
            )
            
            self.components[component_id] = component
            
            # Add to history
            self.component_history[component_id].append({
                "timestamp": current_time,
                "event": "registered",
                "state": ComponentState.INITIALIZING.value,
                "health_score": 0.5
            })
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_registered",
                    {
                        "component_id": component_id,
                        "name": name,
                        "type": component_type.value,
                        "version": version
                    }
                )
            
            self.logger.info(f"Component {component_id} registered successfully")
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_registration_failed",
                    {"component_id": component_id, "error": str(e)}
                )
            self.logger.error(f"Failed to register component {component_id}: {e}")
            return False
    
    async def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from the manager."""
        try:
            if component_id not in self.components:
                return False
            
            component = self.components[component_id]
            
            # Remove all dependencies involving this component
            deps_to_remove = []
            for dep_id, dependency in self.dependencies.items():
                if (dependency.source_component == component_id or 
                    dependency.target_component == component_id):
                    deps_to_remove.append(dep_id)
            
            for dep_id in deps_to_remove:
                await self.remove_dependency(dep_id)
            
            # Add final history entry
            self.component_history[component_id].append({
                "timestamp": time.time(),
                "event": "unregistered",
                "state": component.state.value,
                "health_score": component.health_score
            })
            
            # Remove component
            del self.components[component_id]
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_unregistered",
                    {
                        "component_id": component_id,
                        "final_state": component.state.value,
                        "uptime": component.uptime
                    }
                )
            
            self.logger.info(f"Component {component_id} unregistered successfully")
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_unregistration_failed",
                    {"component_id": component_id, "error": str(e)}
                )
            self.logger.error(f"Failed to unregister component {component_id}: {e}")
            return False
    
    async def update_component_state(self, component_id: str, new_state: ComponentState,
                                   metrics: Dict[str, Any] = None) -> bool:
        """Update the state of a component."""
        try:
            if component_id not in self.components:
                return False
            
            component = self.components[component_id]
            old_state = component.state
            current_time = time.time()
            
            # Update state
            component.state = new_state
            component.last_updated = current_time
            
            # Update uptime
            if new_state == ComponentState.HEALTHY:
                component.uptime = current_time - component.created_time
            
            # Track restarts
            if new_state == ComponentState.INITIALIZING and old_state in [ComponentState.FAILED, ComponentState.CRITICAL]:
                component.restart_count += 1
            
            # Update metrics if provided
            if metrics:
                component.performance_metrics.update(metrics)
                
                # Update resource usage
                if "resource_usage" in metrics:
                    component.resource_usage.update(metrics["resource_usage"])
                
                # Update error counts
                if "error_count" in metrics:
                    component.error_count = metrics["error_count"]
                if "warning_count" in metrics:
                    component.warning_count = metrics["warning_count"]
            
            # Recalculate health score
            await self._update_component_health(component_id)
            
            # Add to history
            self.component_history[component_id].append({
                "timestamp": current_time,
                "event": "state_change",
                "old_state": old_state.value,
                "new_state": new_state.value,
                "health_score": component.health_score
            })
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_state_updated",
                    {
                        "component_id": component_id,
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "health_score": component.health_score
                    }
                )
            
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_state_update_failed",
                    {"component_id": component_id, "error": str(e)}
                )
            return False
    
    async def add_dependency(self, source_id: str, target_id: str, 
                           dependency_type: DependencyType, strength: float = 1.0) -> bool:
        """Add a dependency relationship between components."""
        try:
            if source_id not in self.components or target_id not in self.components:
                return False
            
            dependency_id = f"{source_id}->{target_id}"
            
            if dependency_id in self.dependencies:
                self.logger.warning(f"Dependency {dependency_id} already exists")
                return False
            
            dependency = ComponentDependency(
                source_component=source_id,
                target_component=target_id,
                dependency_type=dependency_type,
                strength=strength,
                created_time=time.time()
            )
            
            self.dependencies[dependency_id] = dependency
            
            # Update component dependency lists
            self.components[source_id].dependencies.add(target_id)
            self.components[target_id].dependents.add(source_id)
            
            # Invalidate dependency graph cache
            self.dependency_graph_cache = None
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_added",
                    {
                        "source": source_id,
                        "target": target_id,
                        "type": dependency_type.value,
                        "strength": strength
                    }
                )
            
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_addition_failed",
                    {"source": source_id, "target": target_id, "error": str(e)}
                )
            return False
    
    async def remove_dependency(self, dependency_id: str) -> bool:
        """Remove a dependency relationship."""
        try:
            if dependency_id not in self.dependencies:
                return False
            
            dependency = self.dependencies[dependency_id]
            
            # Update component dependency lists
            if dependency.target_component in self.components[dependency.source_component].dependencies:
                self.components[dependency.source_component].dependencies.remove(dependency.target_component)
            
            if dependency.source_component in self.components[dependency.target_component].dependents:
                self.components[dependency.target_component].dependents.remove(dependency.source_component)
            
            # Remove dependency
            del self.dependencies[dependency_id]
            
            # Invalidate dependency graph cache
            self.dependency_graph_cache = None
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_removed",
                    {
                        "dependency_id": dependency_id,
                        "source": dependency.source_component,
                        "target": dependency.target_component
                    }
                )
            
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_removal_failed",
                    {"dependency_id": dependency_id, "error": str(e)}
                )
            return False
    
    async def _update_loop(self):
        """Periodic update loop for component health and analysis."""
        while self.is_active:
            try:
                # Update health scores for all components
                await self._update_all_component_health()
                
                # Perform dependency analysis periodically
                if (self.last_dependency_analysis is None or 
                    time.time() - self.last_dependency_analysis > 300):  # 5 minutes
                    await self._perform_dependency_analysis()
                
            except Exception as e:
                self.logger.error(f"Error in component manager update loop: {e}")
                if self.audit_system:
                    await self.audit_system.log_event(
                        "component_manager_update_error",
                        {"error": str(e)}
                    )
            
            await asyncio.sleep(self.update_interval)
    
    async def _update_all_component_health(self):
        """Update health scores for all components."""
        try:
            # Get dependency health for context
            dependency_health = {
                comp_id: comp.health_score 
                for comp_id, comp in self.components.items()
            }
            
            # Update health scores
            for comp_id, component in self.components.items():
                old_score = component.health_score
                new_score = self.health_calculator.calculate_health_score(
                    component, dependency_health
                )
                
                if abs(new_score - old_score) > 0.1:  # Significant change
                    component.health_score = new_score
                    component.last_updated = time.time()
                    
                    # Add to history
                    self.component_history[comp_id].append({
                        "timestamp": time.time(),
                        "event": "health_update",
                        "old_health_score": old_score,
                        "new_health_score": new_score,
                        "state": component.state.value
                    })
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "health_update_failed",
                    {"error": str(e)}
                )
    
    async def _update_component_health(self, component_id: str):
        """Update health score for a specific component."""
        try:
            if component_id not in self.components:
                return
            
            component = self.components[component_id]
            
            # Get dependency health for context
            dependency_health = {
                comp_id: comp.health_score 
                for comp_id, comp in self.components.items()
            }
            
            old_score = component.health_score
            new_score = self.health_calculator.calculate_health_score(
                component, dependency_health
            )
            
            component.health_score = new_score
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "single_component_health_update_failed",
                    {"component_id": component_id, "error": str(e)}
                )
    
    async def _perform_dependency_analysis(self):
        """Perform comprehensive dependency analysis."""
        try:
            if not self.components or not self.dependencies:
                return
            
            analysis = await self.dependency_analyzer.analyze_dependencies(
                self.components, list(self.dependencies.values())
            )
            
            self.last_dependency_analysis = time.time()
            
            # Log significant findings
            if analysis["circular_dependencies"]:
                self.logger.warning(
                    f"Found {len(analysis['circular_dependencies'])} circular dependencies"
                )
            
            if analysis["single_points_of_failure"]:
                self.logger.warning(
                    f"Found {len(analysis['single_points_of_failure'])} single points of failure"
                )
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "dependency_analysis_error",
                    {"error": str(e)}
                )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component manager status."""
        return {
            "is_active": self.is_active,
            "total_components": len(self.components),
            "total_dependencies": len(self.dependencies),
            "component_states": {
                state.value: len([c for c in self.components.values() if c.state == state])
                for state in ComponentState
            },
            "average_health_score": (
                sum(c.health_score for c in self.components.values()) / len(self.components)
                if self.components else 0.0
            ),
            "update_interval": self.update_interval,
            "last_dependency_analysis": self.last_dependency_analysis
        }
    
    async def get_component(self, component_id: str) -> Optional[ComponentInfo]:
        """Get component information by ID."""
        return self.components.get(component_id)
    
    async def get_components(self, component_type: Optional[ComponentType] = None,
                           state: Optional[ComponentState] = None,
                           min_health: Optional[float] = None) -> List[ComponentInfo]:
        """Get components with optional filtering."""
        components = list(self.components.values())
        
        if component_type:
            components = [c for c in components if c.component_type == component_type]
        
        if state:
            components = [c for c in components if c.state == state]
        
        if min_health is not None:
            components = [c for c in components if c.health_score >= min_health]
        
        return components
    
    async def get_component_dependencies(self, component_id: str) -> List[ComponentDependency]:
        """Get all dependencies for a component."""
        return [
            dep for dep in self.dependencies.values()
            if dep.source_component == component_id
        ]
    
    async def get_component_dependents(self, component_id: str) -> List[ComponentDependency]:
        """Get all components that depend on this component."""
        return [
            dep for dep in self.dependencies.values()
            if dep.target_component == component_id
        ]
    
    async def get_component_history(self, component_id: str) -> List[Dict[str, Any]]:
        """Get historical data for a component."""
        return list(self.component_history.get(component_id, []))
    
    async def update_component_configuration(self, component_id: str, 
                                           configuration: Dict[str, Any]) -> bool:
        """Update component configuration."""
        try:
            if component_id not in self.components:
                return False
            
            component = self.components[component_id]
            old_config = component.configuration.copy()
            
            component.configuration.update(configuration)
            component.last_updated = time.time()
            
            # Add to history
            self.component_history[component_id].append({
                "timestamp": time.time(),
                "event": "configuration_update",
                "old_configuration": old_config,
                "new_configuration": component.configuration,
                "state": component.state.value
            })
            
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_configuration_updated",
                    {
                        "component_id": component_id,
                        "updated_keys": list(configuration.keys())
                    }
                )
            
            return True
            
        except Exception as e:
            if self.audit_system:
                await self.audit_system.log_event(
                    "component_configuration_update_failed",
                    {"component_id": component_id, "error": str(e)}
                )
            return False