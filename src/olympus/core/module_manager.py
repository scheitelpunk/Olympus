"""
Module Manager - Lifecycle Management for OLYMPUS Modules
=========================================================

Manages the lifecycle of all OLYMPUS modules including:
- GASM (General AI Safety Module)
- MORPHEUS (Monitoring and Response)
- PROMETHEUS (Predictive Risk Operations)
- ATLAS (Advanced Threat Analysis)
- NEXUS (Network and Exchange)

Key Responsibilities:
- Module registration and discovery
- Lifecycle management (initialize, start, stop, shutdown)
- Health monitoring and status reporting
- Inter-module communication coordination
- Resource allocation and management
"""

import asyncio
import logging
import importlib
import inspect
from typing import Dict, Any, Optional, List, Type, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
from pathlib import Path


class ModuleState(Enum):
    """Module operational states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ModulePriority(Enum):
    """Module priority levels for startup/shutdown ordering"""
    CRITICAL = 1    # Core safety systems
    HIGH = 2        # Essential operations
    NORMAL = 3      # Standard modules
    LOW = 4         # Optional enhancements


@dataclass
class ModuleInfo:
    """Information about a registered module"""
    name: str
    version: str
    description: str
    priority: ModulePriority
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    health_checks: List[str] = field(default_factory=list)


@dataclass
class ModuleStatus:
    """Current status of a module"""
    name: str
    state: ModuleState
    health: str  # healthy, degraded, critical
    last_heartbeat: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class ModuleProtocol(Protocol):
    """Protocol that all OLYMPUS modules must implement"""
    
    def get_module_info(self) -> ModuleInfo:
        """Return module information"""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the module"""
        ...
    
    async def start(self) -> bool:
        """Start the module"""
        ...
    
    async def stop(self) -> bool:
        """Stop the module"""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the module"""
        ...
    
    async def get_status(self) -> ModuleStatus:
        """Get current module status"""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        ...
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute a module-specific action"""
        ...


class ModuleManager:
    """
    Manages the lifecycle of all OLYMPUS modules
    
    Provides centralized module management with dependency resolution,
    health monitoring, and coordinated lifecycle management.
    """
    
    def __init__(self, config_manager):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Module registry and instances
        self.modules: Dict[str, ModuleProtocol] = {}
        self.module_info: Dict[str, ModuleInfo] = {}
        self.module_status: Dict[str, ModuleStatus] = {}
        
        # Lifecycle management
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Monitoring and coordination
        self.health_check_interval = 10.0  # seconds
        self.heartbeat_timeout = 30.0  # seconds
        self._monitoring_active = False
        self._monitoring_tasks: List[asyncio.Task] = []
        self._shutdown_event = threading.Event()
        
        # Module discovery paths
        self.module_paths = [
            Path("src/olympus/safety"),      # GASM
            Path("src/olympus/monitoring"),  # MORPHEUS
            Path("src/olympus/prediction"),  # PROMETHEUS
            Path("src/olympus/analysis"),    # ATLAS
            Path("src/olympus/network")      # NEXUS
        ]
        
        self.logger.info("Module Manager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the Module Manager
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Module Manager...")
            
            # Discover available modules
            discovered_modules = await self._discover_modules()
            self.logger.info(f"Discovered {len(discovered_modules)} modules")
            
            # Register discovered modules
            for module_class in discovered_modules:
                await self._register_module_class(module_class)
            
            # Resolve dependencies and create startup order
            self._resolve_dependencies()
            
            # Initialize all modules
            for module_name in self.startup_order:
                success = await self._initialize_module(module_name)
                if not success:
                    self.logger.error(f"Failed to initialize module: {module_name}")
                    return False
            
            # Start health monitoring
            await self._start_monitoring()
            
            self.logger.info("Module Manager initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Module Manager initialization failed: {e}")
            return False
    
    async def start_all_modules(self) -> bool:
        """
        Start all registered modules in dependency order
        
        Returns:
            bool: True if all modules started successfully
        """
        try:
            self.logger.info("Starting all modules...")
            
            for module_name in self.startup_order:
                self.logger.info(f"Starting module: {module_name}")
                success = await self._start_module(module_name)
                if not success:
                    self.logger.error(f"Failed to start module: {module_name}")
                    return False
                self.logger.info(f"Module {module_name} started successfully")
            
            self.logger.info("All modules started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start all modules: {e}")
            return False
    
    async def stop_all_modules(self) -> bool:
        """
        Stop all modules in reverse dependency order
        
        Returns:
            bool: True if all modules stopped successfully
        """
        try:
            self.logger.info("Stopping all modules...")
            
            for module_name in self.shutdown_order:
                self.logger.info(f"Stopping module: {module_name}")
                success = await self._stop_module(module_name)
                if not success:
                    self.logger.warning(f"Module {module_name} did not stop cleanly")
                else:
                    self.logger.info(f"Module {module_name} stopped successfully")
            
            self.logger.info("All modules stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop all modules: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the Module Manager"""
        self.logger.info("Shutting down Module Manager...")
        
        self._shutdown_event.set()
        
        # Stop monitoring
        await self._stop_monitoring()
        
        # Shutdown all modules
        await self.stop_all_modules()
        
        # Final cleanup
        for module_name, module in self.modules.items():
            try:
                await module.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down module {module_name}: {e}")
        
        self.modules.clear()
        self.module_status.clear()
        
        self.logger.info("Module Manager shutdown complete")
    
    async def get_module(self, module_name: str) -> Optional[ModuleProtocol]:
        """
        Get a module instance by name
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module instance or None if not found
        """
        return self.modules.get(module_name)
    
    async def get_module_status(self, module_name: str) -> Optional[ModuleStatus]:
        """
        Get status of a specific module
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module status or None if not found
        """
        return self.module_status.get(module_name)
    
    async def get_all_module_status(self) -> Dict[str, ModuleStatus]:
        """
        Get status of all modules
        
        Returns:
            Dictionary of module statuses
        """
        return self.module_status.copy()
    
    async def restart_module(self, module_name: str) -> bool:
        """
        Restart a specific module
        
        Args:
            module_name: Name of the module to restart
            
        Returns:
            bool: True if restart successful
        """
        try:
            self.logger.info(f"Restarting module: {module_name}")
            
            # Stop the module
            success = await self._stop_module(module_name)
            if not success:
                self.logger.warning(f"Module {module_name} did not stop cleanly")
            
            # Start the module
            success = await self._start_module(module_name)
            if success:
                self.logger.info(f"Module {module_name} restarted successfully")
                return True
            else:
                self.logger.error(f"Failed to restart module: {module_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error restarting module {module_name}: {e}")
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health from module perspective
        
        Returns:
            Dictionary containing system health information
        """
        total_modules = len(self.modules)
        healthy_modules = sum(1 for status in self.module_status.values() 
                            if status.health == "healthy")
        critical_modules = sum(1 for status in self.module_status.values() 
                             if status.health == "critical")
        
        overall_health = "healthy"
        if critical_modules > 0:
            overall_health = "critical"
        elif healthy_modules < total_modules * 0.8:
            overall_health = "degraded"
        
        return {
            'overall_health': overall_health,
            'total_modules': total_modules,
            'healthy_modules': healthy_modules,
            'degraded_modules': total_modules - healthy_modules - critical_modules,
            'critical_modules': critical_modules,
            'module_details': {name: status.__dict__ for name, status in self.module_status.items()}
        }
    
    # Private methods
    
    async def _discover_modules(self) -> List[Type[ModuleProtocol]]:
        """Discover available modules in the module paths"""
        discovered = []
        
        for module_path in self.module_paths:
            if not module_path.exists():
                continue
            
            try:
                # Look for module files
                for python_file in module_path.glob("*.py"):
                    if python_file.name.startswith("__"):
                        continue
                    
                    module_name = f"olympus.{module_path.name}.{python_file.stem}"
                    
                    try:
                        module = importlib.import_module(module_name)
                        
                        # Look for classes implementing ModuleProtocol
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (hasattr(obj, 'get_module_info') and 
                                hasattr(obj, 'initialize') and
                                hasattr(obj, 'start') and
                                hasattr(obj, 'stop')):
                                discovered.append(obj)
                    
                    except ImportError as e:
                        self.logger.debug(f"Could not import {module_name}: {e}")
                        continue
            
            except Exception as e:
                self.logger.error(f"Error discovering modules in {module_path}: {e}")
        
        return discovered
    
    async def _register_module_class(self, module_class: Type[ModuleProtocol]) -> bool:
        """Register a module class"""
        try:
            # Create instance
            instance = module_class()
            info = instance.get_module_info()
            
            # Register the module
            self.modules[info.name] = instance
            self.module_info[info.name] = info
            
            # Initialize status
            self.module_status[info.name] = ModuleStatus(
                name=info.name,
                state=ModuleState.UNINITIALIZED,
                health="unknown",
                last_heartbeat=datetime.now()
            )
            
            # Add to dependency graph
            self.dependency_graph[info.name] = info.dependencies
            
            self.logger.info(f"Registered module: {info.name} v{info.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register module class: {e}")
            return False
    
    def _resolve_dependencies(self) -> None:
        """Resolve module dependencies and create startup/shutdown orders"""
        # Topological sort for startup order
        visited = set()
        temp_visited = set()
        self.startup_order = []
        
        def visit(module_name):
            if module_name in temp_visited:
                raise Exception(f"Circular dependency detected involving {module_name}")
            if module_name in visited:
                return
            
            temp_visited.add(module_name)
            
            for dependency in self.dependency_graph.get(module_name, []):
                if dependency in self.modules:
                    visit(dependency)
            
            temp_visited.remove(module_name)
            visited.add(module_name)
            self.startup_order.append(module_name)
        
        # Sort modules by priority first
        modules_by_priority = sorted(
            self.modules.keys(), 
            key=lambda m: self.module_info[m].priority.value
        )
        
        for module_name in modules_by_priority:
            if module_name not in visited:
                visit(module_name)
        
        # Shutdown order is reverse of startup
        self.shutdown_order = list(reversed(self.startup_order))
        
        self.logger.info(f"Startup order: {self.startup_order}")
        self.logger.info(f"Shutdown order: {self.shutdown_order}")
    
    async def _initialize_module(self, module_name: str) -> bool:
        """Initialize a specific module"""
        try:
            module = self.modules[module_name]
            status = self.module_status[module_name]
            
            status.state = ModuleState.INITIALIZING
            
            # Get module configuration
            module_config = await self.config_manager.get_module_config(module_name)
            
            # Initialize the module
            success = await module.initialize(module_config)
            
            if success:
                status.state = ModuleState.INITIALIZED
                status.health = "healthy"
                self.logger.info(f"Module {module_name} initialized successfully")
            else:
                status.state = ModuleState.ERROR
                status.health = "critical"
                self.logger.error(f"Module {module_name} initialization failed")
            
            status.last_heartbeat = datetime.now()
            return success
            
        except Exception as e:
            self.module_status[module_name].state = ModuleState.ERROR
            self.module_status[module_name].health = "critical"
            self.module_status[module_name].last_error = str(e)
            self.module_status[module_name].error_count += 1
            self.logger.error(f"Error initializing module {module_name}: {e}")
            return False
    
    async def _start_module(self, module_name: str) -> bool:
        """Start a specific module"""
        try:
            module = self.modules[module_name]
            status = self.module_status[module_name]
            
            status.state = ModuleState.STARTING
            
            success = await module.start()
            
            if success:
                status.state = ModuleState.ACTIVE
                status.health = "healthy"
            else:
                status.state = ModuleState.ERROR
                status.health = "critical"
            
            status.last_heartbeat = datetime.now()
            return success
            
        except Exception as e:
            self.module_status[module_name].state = ModuleState.ERROR
            self.module_status[module_name].health = "critical"
            self.module_status[module_name].last_error = str(e)
            self.module_status[module_name].error_count += 1
            self.logger.error(f"Error starting module {module_name}: {e}")
            return False
    
    async def _stop_module(self, module_name: str) -> bool:
        """Stop a specific module"""
        try:
            module = self.modules[module_name]
            status = self.module_status[module_name]
            
            status.state = ModuleState.STOPPING
            
            success = await module.stop()
            
            if success:
                status.state = ModuleState.STOPPED
                status.health = "healthy"
            else:
                status.state = ModuleState.ERROR
                status.health = "degraded"
            
            status.last_heartbeat = datetime.now()
            return success
            
        except Exception as e:
            self.module_status[module_name].state = ModuleState.ERROR
            self.module_status[module_name].health = "critical"
            self.module_status[module_name].last_error = str(e)
            self.module_status[module_name].error_count += 1
            self.logger.error(f"Error stopping module {module_name}: {e}")
            return False
    
    async def _start_monitoring(self) -> None:
        """Start health monitoring tasks"""
        self._monitoring_active = True
        
        # Health check monitoring
        task = asyncio.create_task(self._health_check_monitor())
        self._monitoring_tasks.append(task)
        
        # Heartbeat monitoring
        task = asyncio.create_task(self._heartbeat_monitor())
        self._monitoring_tasks.append(task)
        
        self.logger.info("Module monitoring started")
    
    async def _stop_monitoring(self) -> None:
        """Stop health monitoring tasks"""
        self._monitoring_active = False
        
        for task in self._monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._monitoring_tasks.clear()
        self.logger.info("Module monitoring stopped")
    
    async def _health_check_monitor(self) -> None:
        """Monitor module health"""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                for module_name, module in self.modules.items():
                    try:
                        health_result = await module.health_check()
                        status = self.module_status[module_name]
                        
                        # Update health based on check results
                        if health_result.get('healthy', False):
                            status.health = "healthy"
                        elif health_result.get('degraded', False):
                            status.health = "degraded"
                        else:
                            status.health = "critical"
                        
                        status.last_heartbeat = datetime.now()
                        status.performance_metrics = health_result.get('performance', {})
                        status.resource_usage = health_result.get('resources', {})
                        
                    except Exception as e:
                        status = self.module_status[module_name]
                        status.health = "critical"
                        status.last_error = str(e)
                        status.error_count += 1
                        self.logger.error(f"Health check failed for {module_name}: {e}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor module heartbeats"""
        while self._monitoring_active and not self._shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                for module_name, status in self.module_status.items():
                    time_since_heartbeat = (current_time - status.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        self.logger.warning(f"Module {module_name} heartbeat timeout")
                        if status.health != "critical":
                            status.health = "degraded"
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5.0)