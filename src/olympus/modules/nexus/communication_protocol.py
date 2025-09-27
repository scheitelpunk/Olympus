"""
Communication Protocol - Robot-to-Robot Communication System
===========================================================

The Communication Protocol module implements secure, reliable, and efficient
communication between robots in the swarm, enabling coordination, knowledge
sharing, and collective decision-making while maintaining network resilience.

Features:
- Secure encrypted communication channels
- Multi-hop routing and mesh networking
- Message prioritization and QoS
- Fault-tolerant communication
- Bandwidth optimization
- Emergency communication protocols
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import hmac
import base64
from collections import defaultdict, deque
import time
import random

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the communication protocol"""
    HEARTBEAT = "heartbeat"
    DATA = "data"
    COMMAND = "command"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    EMERGENCY = "emergency"
    CONSENSUS = "consensus"
    KNOWLEDGE_SHARE = "knowledge_share"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"


class MessagePriority(Enum):
    """Message priority levels"""
    EMERGENCY = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


class RoutingStrategy(Enum):
    """Routing strategies for message delivery"""
    DIRECT = "direct"
    FLOODING = "flooding"
    SHORTEST_PATH = "shortest_path"
    LOAD_BALANCED = "load_balanced"
    REDUNDANT = "redundant"


class NetworkTopology(Enum):
    """Network topology configurations"""
    MESH = "mesh"
    STAR = "star"
    TREE = "tree"
    RING = "ring"
    HYBRID = "hybrid"


@dataclass
class Message:
    """A message in the communication system"""
    id: str
    sender_id: str
    recipient_id: str  # Can be broadcast for all robots
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: datetime
    ttl: int = 10  # Time to live (hops)
    encrypted: bool = True
    signature: Optional[str] = None
    routing_history: List[str] = field(default_factory=list)
    delivery_attempts: int = 0
    max_delivery_attempts: int = 3


@dataclass
class CommunicationChannel:
    """A communication channel between robots"""
    channel_id: str
    robot_a: str
    robot_b: str
    channel_type: str  # 'wifi', 'bluetooth', 'radio', etc.
    signal_strength: float
    bandwidth: float  # Mbps
    latency: float  # ms
    packet_loss_rate: float
    last_heartbeat: datetime
    encryption_key: Optional[str] = None
    active: bool = True


@dataclass
class NetworkNode:
    """A node in the communication network"""
    robot_id: str
    position: Tuple[float, float, float]
    communication_range: float
    available_channels: List[str]
    neighbors: Set[str]
    routing_table: Dict[str, List[str]]  # destination -> [path]
    message_queue: deque = field(default_factory=deque)
    bandwidth_usage: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)


class CommunicationProtocol:
    """
    Robot-to-robot communication protocol system
    
    Manages secure, reliable communication between robots with mesh networking,
    routing optimization, and fault tolerance capabilities.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Network state
        self.network_nodes: Dict[str, NetworkNode] = {}
        self.communication_channels: Dict[str, CommunicationChannel] = {}
        self.active_connections: Dict[Tuple[str, str], str] = {}  # (robot_a, robot_b) -> channel_id
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.pending_messages: Dict[str, Message] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.delivered_messages: Set[str] = set()  # For duplicate prevention
        
        # Network topology and routing
        self.topology = NetworkTopology.MESH
        self.routing_tables: Dict[str, Dict[str, List[str]]] = {}
        self.network_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self.message_processor_task = None
        self.network_monitor_task = None
        self.heartbeat_task = None
        
        # Security
        self.encryption_keys: Dict[str, str] = {}
        self.robot_certificates: Dict[str, str] = {}
        
        # Quality of Service
        self.qos_policies: Dict[MessageType, Dict[str, Any]] = {}
        self.bandwidth_limits: Dict[str, float] = {}
        
        # Network metrics
        self.message_delivery_rate = 0.0
        self.average_latency = 0.0
        self.network_throughput = 0.0
        self.packet_loss_rate = 0.0
        
        # Emergency protocols
        self.emergency_mode = False
        self.emergency_channels: List[str] = []
        
        logger.info("Communication Protocol system initialized")
    
    async def initialize(self) -> bool:
        """Initialize the communication protocol system"""
        try:
            # Initialize QoS policies
            await self._initialize_qos_policies()
            
            # Initialize security
            await self._initialize_security()
            
            # Start background tasks
            self.message_processor_task = asyncio.create_task(self._process_messages())
            self.network_monitor_task = asyncio.create_task(self._monitor_network())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_protocol())
            
            logger.info("Communication Protocol system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Communication Protocol initialization failed: {e}")
            return False
    
    async def register_robot(self, robot_id: str, position: Tuple[float, float, float],
                           communication_range: float = 1000.0,
                           available_channels: List[str] = None) -> bool:
        """Register a robot in the communication network"""
        try:
            node = NetworkNode(
                robot_id=robot_id,
                position=position,
                communication_range=communication_range,
                available_channels=available_channels or ["wifi", "bluetooth"],
                neighbors=set(),
                routing_table={}
            )
            
            self.network_nodes[robot_id] = node
            self.network_graph[robot_id] = set()
            
            # Generate encryption keys
            await self._generate_robot_keys(robot_id)
            
            # Update network topology
            await self._update_network_topology()
            
            # Build routing tables
            await self._build_routing_tables()
            
            logger.info(f"Robot {robot_id} registered in communication network")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register robot {robot_id}: {e}")
            return False
    
    async def send_message(self, sender_id: str, recipient_id: str,
                         message_type: MessageType, payload: Dict[str, Any],
                         priority: MessagePriority = MessagePriority.NORMAL,
                         routing_strategy: RoutingStrategy = RoutingStrategy.SHORTEST_PATH) -> str:
        """Send a message from one robot to another"""
        try:
            if sender_id not in self.network_nodes:
                logger.error(f"Sender robot {sender_id} not registered")
                return ""
            
            # Create message
            message_id = f"msg_{sender_id}_{datetime.now().timestamp()}_{random.randint(1000, 9999)}"
            message = Message(
                id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                priority=priority,
                payload=payload,
                timestamp=datetime.now(),
                ttl=10,
                encrypted=True
            )
            
            # Apply security
            if message.encrypted:
                await self._encrypt_message(message)
                await self._sign_message(message)
            
            # Add to pending messages
            self.pending_messages[message_id] = message
            
            # Queue for processing
            await self.message_queue.put((message_id, routing_strategy))
            
            logger.debug(f"Message queued: {message_id} from {sender_id} to {recipient_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return ""
    
    async def broadcast_message(self, sender_id: str, message_type: MessageType,
                              payload: Dict[str, Any],
                              priority: MessagePriority = MessagePriority.NORMAL,
                              ttl: int = 5) -> str:
        """Broadcast a message to all robots in the network"""
        try:
            return await self.send_message(
                sender_id=sender_id,
                recipient_id="broadcast",
                message_type=message_type,
                payload=payload,
                priority=priority,
                routing_strategy=RoutingStrategy.FLOODING
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return ""
    
    async def register_message_handler(self, message_type: MessageType, 
                                     handler: Callable[[Message], Any]) -> bool:
        """Register a handler for specific message types"""
        try:
            self.message_handlers[message_type].append(handler)
            logger.info(f"Message handler registered for {message_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register message handler: {e}")
            return False
    
    async def establish_channel(self, robot_a: str, robot_b: str,
                              channel_type: str = "wifi") -> str:
        """Establish a communication channel between two robots"""
        try:
            if robot_a not in self.network_nodes or robot_b not in self.network_nodes:
                logger.error("One or both robots not registered")
                return ""
            
            # Check if robots are within communication range
            if not await self._robots_in_range(robot_a, robot_b):
                logger.warning(f"Robots {robot_a} and {robot_b} not in communication range")
                return ""
            
            # Create channel
            channel_id = f"channel_{robot_a}_{robot_b}_{channel_type}_{datetime.now().timestamp()}"
            
            # Calculate channel properties
            distance = await self._calculate_distance(robot_a, robot_b)
            signal_strength = max(0.1, 1.0 - (distance / self.network_nodes[robot_a].communication_range))
            
            channel = CommunicationChannel(
                channel_id=channel_id,
                robot_a=robot_a,
                robot_b=robot_b,
                channel_type=channel_type,
                signal_strength=signal_strength,
                bandwidth=self._get_channel_bandwidth(channel_type, signal_strength),
                latency=self._calculate_channel_latency(distance, channel_type),
                packet_loss_rate=max(0.001, (1.0 - signal_strength) * 0.1),
                last_heartbeat=datetime.now(),
                encryption_key=await self._generate_channel_key(robot_a, robot_b)
            )
            
            self.communication_channels[channel_id] = channel
            self.active_connections[(robot_a, robot_b)] = channel_id
            self.active_connections[(robot_b, robot_a)] = channel_id
            
            # Update network graph
            self.network_graph[robot_a].add(robot_b)
            self.network_graph[robot_b].add(robot_a)
            
            # Update neighbor lists
            self.network_nodes[robot_a].neighbors.add(robot_b)
            self.network_nodes[robot_b].neighbors.add(robot_a)
            
            logger.info(f"Communication channel established: {channel_id}")
            return channel_id
            
        except Exception as e:
            logger.error(f"Failed to establish channel: {e}")
            return ""
    
    async def send_emergency_message(self, sender_id: str, emergency_type: str,
                                   details: Dict[str, Any]) -> List[str]:
        """Send emergency message with highest priority and redundancy"""
        try:
            emergency_payload = {
                "emergency_type": emergency_type,
                "details": details,
                "timestamp": datetime.now().isoformat(),
                "sender_position": self.network_nodes[sender_id].position if sender_id in self.network_nodes else None
            }
            
            # Send multiple redundant messages
            message_ids = []
            
            # Broadcast with flooding
            flood_msg_id = await self.send_message(
                sender_id=sender_id,
                recipient_id="broadcast",
                message_type=MessageType.EMERGENCY,
                payload=emergency_payload,
                priority=MessagePriority.EMERGENCY,
                routing_strategy=RoutingStrategy.FLOODING
            )
            if flood_msg_id:
                message_ids.append(flood_msg_id)
            
            # Direct messages to all reachable robots
            for robot_id in self.network_nodes:
                if robot_id != sender_id:
                    direct_msg_id = await self.send_message(
                        sender_id=sender_id,
                        recipient_id=robot_id,
                        message_type=MessageType.EMERGENCY,
                        payload=emergency_payload,
                        priority=MessagePriority.EMERGENCY,
                        routing_strategy=RoutingStrategy.REDUNDANT
                    )
                    if direct_msg_id:
                        message_ids.append(direct_msg_id)
            
            logger.critical(f"Emergency messages sent: {len(message_ids)} messages")
            return message_ids
            
        except Exception as e:
            logger.error(f"Failed to send emergency message: {e}")
            return []
    
    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        try:
            active_nodes = len([node for node in self.network_nodes.values()
                              if (datetime.now() - node.last_seen).seconds < 60])
            
            active_channels = len([ch for ch in self.communication_channels.values() if ch.active])
            
            # Calculate network connectivity
            total_possible_connections = len(self.network_nodes) * (len(self.network_nodes) - 1) // 2
            actual_connections = len(self.active_connections) // 2  # Each connection counted twice
            connectivity_ratio = actual_connections / max(1, total_possible_connections)
            
            return {
                "total_robots": len(self.network_nodes),
                "active_robots": active_nodes,
                "total_channels": len(self.communication_channels),
                "active_channels": active_channels,
                "connectivity_ratio": connectivity_ratio,
                "message_delivery_rate": self.message_delivery_rate,
                "average_latency": self.average_latency,
                "network_throughput": self.network_throughput,
                "packet_loss_rate": self.packet_loss_rate,
                "emergency_mode": self.emergency_mode,
                "pending_messages": len(self.pending_messages),
                "topology": self.topology.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return {}
    
    async def get_health(self) -> Dict[str, Any]:
        """Get communication system health metrics"""
        try:
            # Calculate health metrics
            unhealthy_channels = len([ch for ch in self.communication_channels.values()
                                    if ch.packet_loss_rate > 0.1 or ch.signal_strength < 0.3])
            
            total_channels = len(self.communication_channels)
            channel_health = 1.0 - (unhealthy_channels / max(1, total_channels))
            
            # Message queue health
            queue_size = self.message_queue.qsize()
            queue_health = max(0.0, 1.0 - (queue_size / 1000))  # Healthy if < 1000 pending
            
            # Overall health
            overall_health = (channel_health + queue_health + self.message_delivery_rate) / 3
            
            return {
                "overall_health": overall_health,
                "channel_health": channel_health,
                "queue_health": queue_health,
                "unhealthy_channels": unhealthy_channels,
                "message_queue_size": queue_size,
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get communication health: {e}")
            return {"overall_health": 0.0}
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown communication protocol system"""
        try:
            logger.info("Shutting down Communication Protocol system")
            
            # Stop background tasks
            if self.message_processor_task:
                self.message_processor_task.cancel()
            if self.network_monitor_task:
                self.network_monitor_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            
            # Close all channels
            for channel in self.communication_channels.values():
                channel.active = False
            
            # Clear queues and state
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.pending_messages.clear()
            self.network_nodes.clear()
            self.communication_channels.clear()
            
            logger.info("Communication Protocol system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Communication Protocol shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _process_messages(self):
        """Background task to process message queue"""
        while True:
            try:
                # Get message from queue
                try:
                    message_id, routing_strategy = await asyncio.wait_for(
                        self.message_queue.get(), timeout=1.0
                    )
                    await self._route_message(message_id, routing_strategy)
                except asyncio.TimeoutError:
                    continue
                
                # Process heartbeats and maintenance
                await self._process_heartbeats()
                
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
    
    async def _route_message(self, message_id: str, routing_strategy: RoutingStrategy):
        """Route a message based on the specified strategy"""
        try:
            if message_id not in self.pending_messages:
                return
            
            message = self.pending_messages[message_id]
            
            # Check TTL
            if message.ttl <= 0:
                logger.warning(f"Message {message_id} expired (TTL reached)")
                del self.pending_messages[message_id]
                return
            
            # Handle broadcast messages
            if message.recipient_id == "broadcast":
                await self._handle_broadcast(message, routing_strategy)
            else:
                await self._handle_unicast(message, routing_strategy)
            
        except Exception as e:
            logger.error(f"Error routing message {message_id}: {e}")
    
    async def _handle_broadcast(self, message: Message, routing_strategy: RoutingStrategy):
        """Handle broadcast message routing"""
        try:
            if routing_strategy == RoutingStrategy.FLOODING:
                # Send to all directly connected neighbors
                sender_neighbors = self.network_nodes[message.sender_id].neighbors
                
                for neighbor_id in sender_neighbors:
                    if neighbor_id not in message.routing_history:
                        await self._deliver_message_to_robot(message, neighbor_id)
                        message.routing_history.append(neighbor_id)
            
            # Remove from pending
            if message.id in self.pending_messages:
                del self.pending_messages[message.id]
                
        except Exception as e:
            logger.error(f"Error handling broadcast message: {e}")
    
    async def _handle_unicast(self, message: Message, routing_strategy: RoutingStrategy):
        """Handle unicast message routing"""
        try:
            sender_id = message.sender_id
            recipient_id = message.recipient_id
            
            # Check if recipient is directly connected
            if recipient_id in self.network_nodes[sender_id].neighbors:
                await self._deliver_message_to_robot(message, recipient_id)
                del self.pending_messages[message.id]
                return
            
            # Find route based on strategy
            route = await self._find_route(sender_id, recipient_id, routing_strategy)
            
            if route and len(route) > 1:
                next_hop = route[1]  # First hop after sender
                await self._forward_message(message, next_hop)
            else:
                logger.warning(f"No route found from {sender_id} to {recipient_id}")
                # Try different routing strategy as fallback
                if routing_strategy != RoutingStrategy.FLOODING:
                    await self._handle_broadcast(message, RoutingStrategy.FLOODING)
                else:
                    # Message delivery failed
                    message.delivery_attempts += 1
                    if message.delivery_attempts >= message.max_delivery_attempts:
                        logger.error(f"Message {message.id} delivery failed after {message.delivery_attempts} attempts")
                        del self.pending_messages[message.id]
            
        except Exception as e:
            logger.error(f"Error handling unicast message: {e}")
    
    async def _find_route(self, source: str, destination: str, strategy: RoutingStrategy) -> List[str]:
        """Find route between source and destination"""
        try:
            if strategy == RoutingStrategy.SHORTEST_PATH:
                return await self._shortest_path(source, destination)
            elif strategy == RoutingStrategy.LOAD_BALANCED:
                return await self._load_balanced_path(source, destination)
            elif strategy == RoutingStrategy.REDUNDANT:
                routes = await self._find_multiple_paths(source, destination, 2)
                return routes[0] if routes else []
            else:
                return await self._shortest_path(source, destination)
            
        except Exception as e:
            logger.error(f"Error finding route from {source} to {destination}: {e}")
            return []
    
    async def _shortest_path(self, source: str, destination: str) -> List[str]:
        """Find shortest path using Dijkstra's algorithm"""
        try:
            if source == destination:
                return [source]
            
            # Simple BFS for unweighted graph
            visited = set()
            queue = deque([(source, [source])])
            
            while queue:
                current, path = queue.popleft()
                
                if current == destination:
                    return path
                
                if current in visited:
                    continue
                
                visited.add(current)
                
                for neighbor in self.network_graph.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
            
            return []  # No path found
            
        except Exception as e:
            logger.error(f"Error calculating shortest path: {e}")
            return []
    
    async def _load_balanced_path(self, source: str, destination: str) -> List[str]:
        """Find path considering current load on nodes"""
        try:
            # For simplicity, use shortest path with load consideration
            # In a real implementation, this would consider bandwidth usage
            return await self._shortest_path(source, destination)
            
        except Exception as e:
            logger.error(f"Error calculating load balanced path: {e}")
            return []
    
    async def _find_multiple_paths(self, source: str, destination: str, num_paths: int) -> List[List[str]]:
        """Find multiple disjoint paths for redundancy"""
        try:
            paths = []
            
            # Find first path
            path1 = await self._shortest_path(source, destination)
            if path1:
                paths.append(path1)
            
            # For additional paths, temporarily remove nodes from first path
            # (simplified implementation)
            if len(path1) > 2 and num_paths > 1:
                # Remove intermediate nodes and try again
                original_graph = self.network_graph.copy()
                
                for node in path1[1:-1]:  # Remove intermediate nodes
                    if node in self.network_graph:
                        del self.network_graph[node]
                
                path2 = await self._shortest_path(source, destination)
                if path2 and path2 != path1:
                    paths.append(path2)
                
                # Restore graph
                self.network_graph = original_graph
            
            return paths
            
        except Exception as e:
            logger.error(f"Error finding multiple paths: {e}")
            return []
    
    async def _deliver_message_to_robot(self, message: Message, robot_id: str):
        """Deliver message to a specific robot"""
        try:
            # Verify message signature if encrypted
            if message.encrypted and not await self._verify_message_signature(message):
                logger.warning(f"Message signature verification failed: {message.id}")
                return
            
            # Decrypt message if needed
            decrypted_payload = message.payload
            if message.encrypted:
                decrypted_payload = await self._decrypt_message_payload(message)
            
            # Check for duplicate delivery
            if message.id in self.delivered_messages:
                return  # Already delivered
            
            self.delivered_messages.add(message.id)
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            
            # Update delivery metrics
            delivery_time = (datetime.now() - message.timestamp).total_seconds() * 1000  # ms
            self._update_delivery_metrics(delivery_time)
            
            logger.debug(f"Message delivered to {robot_id}: {message.id}")
            
        except Exception as e:
            logger.error(f"Error delivering message to {robot_id}: {e}")
    
    async def _forward_message(self, message: Message, next_hop: str):
        """Forward message to next hop"""
        try:
            message.ttl -= 1
            message.routing_history.append(message.sender_id)
            
            # In a real implementation, this would send to the next hop robot
            # For now, we simulate by adding back to queue with updated sender
            forwarded_message = Message(
                id=message.id,
                sender_id=next_hop,  # Now forwarding from next hop
                recipient_id=message.recipient_id,
                message_type=message.message_type,
                priority=message.priority,
                payload=message.payload,
                timestamp=message.timestamp,
                ttl=message.ttl,
                encrypted=message.encrypted,
                signature=message.signature,
                routing_history=message.routing_history.copy(),
                delivery_attempts=message.delivery_attempts,
                max_delivery_attempts=message.max_delivery_attempts
            )
            
            self.pending_messages[message.id] = forwarded_message
            await self.message_queue.put((message.id, RoutingStrategy.SHORTEST_PATH))
            
        except Exception as e:
            logger.error(f"Error forwarding message: {e}")
    
    async def _monitor_network(self):
        """Background task to monitor network health"""
        while True:
            try:
                await self._update_network_topology()
                await self._monitor_channel_health()
                await self._update_routing_tables()
                await self._calculate_network_metrics()
                
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring network: {e}")
    
    async def _heartbeat_protocol(self):
        """Background task for heartbeat protocol"""
        while True:
            try:
                current_time = datetime.now()
                
                # Send heartbeats to all neighbors
                for robot_id, node in self.network_nodes.items():
                    for neighbor_id in node.neighbors:
                        heartbeat_message = {
                            "type": "heartbeat",
                            "sender": robot_id,
                            "timestamp": current_time.isoformat(),
                            "status": "active"
                        }
                        
                        await self.send_message(
                            sender_id=robot_id,
                            recipient_id=neighbor_id,
                            message_type=MessageType.HEARTBEAT,
                            payload=heartbeat_message,
                            priority=MessagePriority.BACKGROUND
                        )
                
                # Check for dead connections
                await self._check_dead_connections()
                
                await asyncio.sleep(10.0)  # Heartbeat every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat protocol: {e}")
    
    async def _process_heartbeats(self):
        """Process heartbeat messages"""
        try:
            current_time = datetime.now()
            
            for robot_id, node in self.network_nodes.items():
                # Update last seen time based on message activity
                if hasattr(node, 'last_message_time'):
                    time_since_last = (current_time - node.last_message_time).seconds
                    if time_since_last < 60:  # Active within last minute
                        node.last_seen = current_time
            
        except Exception as e:
            logger.error(f"Error processing heartbeats: {e}")
    
    async def _check_dead_connections(self):
        """Check for and remove dead connections"""
        try:
            current_time = datetime.now()
            dead_channels = []
            
            for channel_id, channel in self.communication_channels.items():
                time_since_heartbeat = (current_time - channel.last_heartbeat).seconds
                
                if time_since_heartbeat > 60:  # No heartbeat for 1 minute
                    dead_channels.append(channel_id)
            
            # Remove dead channels
            for channel_id in dead_channels:
                await self._remove_channel(channel_id)
                logger.info(f"Removed dead channel: {channel_id}")
            
        except Exception as e:
            logger.error(f"Error checking dead connections: {e}")
    
    async def _remove_channel(self, channel_id: str):
        """Remove a communication channel"""
        try:
            if channel_id in self.communication_channels:
                channel = self.communication_channels[channel_id]
                
                # Update network graph
                self.network_graph[channel.robot_a].discard(channel.robot_b)
                self.network_graph[channel.robot_b].discard(channel.robot_a)
                
                # Update neighbor lists
                self.network_nodes[channel.robot_a].neighbors.discard(channel.robot_b)
                self.network_nodes[channel.robot_b].neighbors.discard(channel.robot_a)
                
                # Remove from active connections
                self.active_connections.pop((channel.robot_a, channel.robot_b), None)
                self.active_connections.pop((channel.robot_b, channel.robot_a), None)
                
                # Remove channel
                del self.communication_channels[channel_id]
            
        except Exception as e:
            logger.error(f"Error removing channel {channel_id}: {e}")
    
    async def _update_network_topology(self):
        """Update network topology based on robot positions"""
        try:
            # Check all robot pairs for connectivity
            robot_ids = list(self.network_nodes.keys())
            
            for i, robot_a in enumerate(robot_ids):
                for robot_b in robot_ids[i+1:]:
                    in_range = await self._robots_in_range(robot_a, robot_b)
                    connection_exists = (robot_a, robot_b) in self.active_connections
                    
                    if in_range and not connection_exists:
                        # Establish new connection
                        await self.establish_channel(robot_a, robot_b)
                    elif not in_range and connection_exists:
                        # Remove out-of-range connection
                        channel_id = self.active_connections[(robot_a, robot_b)]
                        await self._remove_channel(channel_id)
            
        except Exception as e:
            logger.error(f"Error updating network topology: {e}")
    
    async def _monitor_channel_health(self):
        """Monitor health of communication channels"""
        try:
            for channel in self.communication_channels.values():
                # Update signal strength based on distance
                distance = await self._calculate_distance(channel.robot_a, channel.robot_b)
                max_range = min(
                    self.network_nodes[channel.robot_a].communication_range,
                    self.network_nodes[channel.robot_b].communication_range
                )
                
                channel.signal_strength = max(0.1, 1.0 - (distance / max_range))
                
                # Update other channel properties
                channel.packet_loss_rate = max(0.001, (1.0 - channel.signal_strength) * 0.1)
                channel.bandwidth = self._get_channel_bandwidth(channel.channel_type, channel.signal_strength)
                
                # Mark as inactive if signal too weak
                if channel.signal_strength < 0.1:
                    channel.active = False
            
        except Exception as e:
            logger.error(f"Error monitoring channel health: {e}")
    
    async def _build_routing_tables(self):
        """Build routing tables for all robots"""
        try:
            for robot_id in self.network_nodes:
                routing_table = {}
                
                # Calculate shortest paths to all other robots
                for destination_id in self.network_nodes:
                    if destination_id != robot_id:
                        path = await self._shortest_path(robot_id, destination_id)
                        if path:
                            routing_table[destination_id] = path
                
                self.network_nodes[robot_id].routing_table = routing_table
            
        except Exception as e:
            logger.error(f"Error building routing tables: {e}")
    
    async def _update_routing_tables(self):
        """Update routing tables based on current topology"""
        await self._build_routing_tables()
    
    def _calculate_network_metrics(self):
        """Calculate network performance metrics"""
        try:
            # These would be calculated from actual message delivery data
            # For now, using simulated metrics
            
            active_channels = [ch for ch in self.communication_channels.values() if ch.active]
            if active_channels:
                self.packet_loss_rate = np.mean([ch.packet_loss_rate for ch in active_channels])
                self.average_latency = np.mean([ch.latency for ch in active_channels])
                self.network_throughput = sum([ch.bandwidth for ch in active_channels])
            
            # Message delivery rate based on pending vs delivered
            total_messages = len(self.delivered_messages) + len(self.pending_messages)
            if total_messages > 0:
                self.message_delivery_rate = len(self.delivered_messages) / total_messages
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
    
    def _update_delivery_metrics(self, delivery_time_ms: float):
        """Update message delivery metrics"""
        try:
            # Update running average of latency
            self.average_latency = (self.average_latency * 0.9) + (delivery_time_ms * 0.1)
            
        except Exception as e:
            logger.error(f"Error updating delivery metrics: {e}")
    
    # Helper methods for distance, encryption, etc.
    
    async def _robots_in_range(self, robot_a: str, robot_b: str) -> bool:
        """Check if two robots are within communication range"""
        try:
            distance = await self._calculate_distance(robot_a, robot_b)
            max_range = min(
                self.network_nodes[robot_a].communication_range,
                self.network_nodes[robot_b].communication_range
            )
            return distance <= max_range
            
        except Exception as e:
            logger.error(f"Error checking robot range: {e}")
            return False
    
    async def _calculate_distance(self, robot_a: str, robot_b: str) -> float:
        """Calculate Euclidean distance between two robots"""
        try:
            pos_a = self.network_nodes[robot_a].position
            pos_b = self.network_nodes[robot_b].position
            
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def _get_channel_bandwidth(self, channel_type: str, signal_strength: float) -> float:
        """Get bandwidth based on channel type and signal strength"""
        base_bandwidths = {
            "wifi": 100.0,      # Mbps
            "bluetooth": 2.0,   # Mbps
            "radio": 0.1,       # Mbps
            "cellular": 50.0    # Mbps
        }
        
        base_bw = base_bandwidths.get(channel_type, 10.0)
        return base_bw * signal_strength
    
    def _calculate_channel_latency(self, distance: float, channel_type: str) -> float:
        """Calculate channel latency based on distance and type"""
        base_latencies = {
            "wifi": 1.0,        # ms
            "bluetooth": 5.0,   # ms
            "radio": 10.0,      # ms
            "cellular": 20.0    # ms
        }
        
        base_latency = base_latencies.get(channel_type, 5.0)
        propagation_delay = distance / 300000  # Speed of light in km/ms (approx)
        
        return base_latency + propagation_delay
    
    # Security methods (simplified implementations)
    
    async def _initialize_security(self):
        """Initialize security components"""
        try:
            # In a real implementation, this would set up proper cryptographic systems
            pass
            
        except Exception as e:
            logger.error(f"Error initializing security: {e}")
    
    async def _generate_robot_keys(self, robot_id: str):
        """Generate encryption keys for a robot"""
        try:
            # Simplified key generation
            key = hashlib.sha256(f"robot_{robot_id}_key".encode()).hexdigest()
            self.encryption_keys[robot_id] = key
            
        except Exception as e:
            logger.error(f"Error generating keys for {robot_id}: {e}")
    
    async def _generate_channel_key(self, robot_a: str, robot_b: str) -> str:
        """Generate shared encryption key for a channel"""
        try:
            # Simplified shared key generation
            combined = f"{robot_a}_{robot_b}_channel"
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
            
        except Exception as e:
            logger.error(f"Error generating channel key: {e}")
            return ""
    
    async def _encrypt_message(self, message: Message):
        """Encrypt message payload"""
        try:
            # Simplified encryption (in production, use proper cryptography)
            payload_json = json.dumps(message.payload)
            # In real implementation: encrypted_payload = encrypt(payload_json, key)
            # For now, just mark as encrypted
            message.payload["encrypted"] = True
            
        except Exception as e:
            logger.error(f"Error encrypting message: {e}")
    
    async def _decrypt_message_payload(self, message: Message) -> Dict[str, Any]:
        """Decrypt message payload"""
        try:
            # Simplified decryption
            payload = message.payload.copy()
            payload.pop("encrypted", None)
            return payload
            
        except Exception as e:
            logger.error(f"Error decrypting message: {e}")
            return message.payload
    
    async def _sign_message(self, message: Message):
        """Sign message for integrity verification"""
        try:
            # Simplified signing
            payload_json = json.dumps(message.payload, sort_keys=True)
            signature_data = f"{message.sender_id}_{payload_json}_{message.timestamp}"
            message.signature = hashlib.sha256(signature_data.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error signing message: {e}")
    
    async def _verify_message_signature(self, message: Message) -> bool:
        """Verify message signature"""
        try:
            # Simplified verification
            if not message.signature:
                return False
            
            payload_json = json.dumps(message.payload, sort_keys=True)
            signature_data = f"{message.sender_id}_{payload_json}_{message.timestamp}"
            expected_signature = hashlib.sha256(signature_data.encode()).hexdigest()
            
            return message.signature == expected_signature
            
        except Exception as e:
            logger.error(f"Error verifying message signature: {e}")
            return False
    
    async def _initialize_qos_policies(self):
        """Initialize Quality of Service policies"""
        try:
            self.qos_policies = {
                MessageType.EMERGENCY: {"priority": 5, "bandwidth_guarantee": 0.5, "max_latency": 100},
                MessageType.COMMAND: {"priority": 4, "bandwidth_guarantee": 0.3, "max_latency": 500},
                MessageType.CONSENSUS: {"priority": 3, "bandwidth_guarantee": 0.2, "max_latency": 1000},
                MessageType.DATA: {"priority": 2, "bandwidth_guarantee": 0.1, "max_latency": 2000},
                MessageType.HEARTBEAT: {"priority": 1, "bandwidth_guarantee": 0.05, "max_latency": 5000}
            }
            
        except Exception as e:
            logger.error(f"Error initializing QoS policies: {e}")