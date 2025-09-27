"""
Collective Memory - Shared Memory Pool System
============================================

The Collective Memory module implements a distributed memory system that allows
the robot swarm to store, retrieve, and share memories collectively while
maintaining data integrity, access control, and efficient storage management.

Features:
- Distributed memory storage across the swarm
- Memory synchronization and replication
- Hierarchical memory organization
- Memory compression and optimization
- Temporal memory management
- Memory access control and security
- Memory consolidation and garbage collection
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import pickle
import gzip
import numpy as np
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories in the collective system"""
    EPISODIC = "episodic"           # Specific experiences/events
    SEMANTIC = "semantic"           # General knowledge/facts
    PROCEDURAL = "procedural"       # Skills and procedures
    WORKING = "working"             # Temporary/active memory
    DECLARATIVE = "declarative"     # Explicit knowledge
    SENSORY = "sensory"            # Sensory experiences
    EMOTIONAL = "emotional"        # Emotional associations
    SPATIAL = "spatial"            # Spatial/location memories


class MemoryPersistence(Enum):
    """Memory persistence levels"""
    VOLATILE = "volatile"           # Lost on restart
    SESSION = "session"             # Persists during session
    PERMANENT = "permanent"         # Persists indefinitely
    ARCHIVED = "archived"           # Long-term archival storage


class MemoryAccess(Enum):
    """Memory access levels"""
    PUBLIC = "public"               # Accessible to all robots
    PRIVATE = "private"             # Robot-specific memory
    GROUP = "group"                 # Accessible to specific groups
    RESTRICTED = "restricted"       # Special permission required


@dataclass
class MemoryEntry:
    """A single memory entry in the collective system"""
    id: str
    owner_robot: str
    memory_type: MemoryType
    title: str
    content: Any
    metadata: Dict[str, Any]
    tags: Set[str]
    timestamp: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 1.0
    persistence: MemoryPersistence = MemoryPersistence.SESSION
    access_level: MemoryAccess = MemoryAccess.PUBLIC
    related_memories: Set[str] = field(default_factory=set)
    compression_ratio: float = 1.0
    replicated_to: Set[str] = field(default_factory=set)
    version: int = 1
    checksum: Optional[str] = None


@dataclass
class MemoryCluster:
    """A cluster of related memories"""
    id: str
    name: str
    memory_ids: Set[str]
    cluster_type: str
    importance_score: float
    last_updated: datetime
    access_pattern: Dict[str, int]  # robot_id -> access_count


@dataclass
class MemoryStats:
    """Statistics about memory usage"""
    total_memories: int
    memory_by_type: Dict[str, int]
    storage_used: int  # bytes
    storage_available: int  # bytes
    compression_ratio: float
    replication_factor: float
    access_frequency: Dict[str, int]
    memory_health_score: float


class CollectiveMemory:
    """
    Distributed collective memory system for robot swarms
    
    Manages shared memories across robots with replication, compression,
    and intelligent access patterns while maintaining data integrity.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Memory storage
        self.local_memories: Dict[str, MemoryEntry] = {}
        self.memory_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> memory_ids
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        
        # Robot memory mappings
        self.robot_memories: Dict[str, Set[str]] = defaultdict(set)
        self.robot_memory_quotas: Dict[str, int] = {}
        self.robot_access_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Memory replication and distribution
        self.replication_factor = 3  # Number of replicas per memory
        self.memory_replicas: Dict[str, Set[str]] = defaultdict(set)  # memory_id -> robot_ids
        self.pending_replications: deque = deque()
        
        # Memory management
        self.memory_cache: Dict[str, Any] = {}  # Fast access cache
        self.cache_size_limit = 1000
        self.compression_enabled = True
        self.auto_cleanup_enabled = True
        
        # Background tasks
        self.replication_task = None
        self.cleanup_task = None
        self.consolidation_task = None
        self.sync_task = None
        
        # Memory metrics
        self.total_storage_used = 0
        self.total_storage_available = 1000000000  # 1GB default
        self.average_compression_ratio = 1.0
        self.memory_health_score = 1.0
        
        # Access control
        self.access_permissions: Dict[str, Dict[str, bool]] = {}  # memory_id -> {robot_id: can_access}
        self.memory_locks: Dict[str, str] = {}  # memory_id -> locking_robot_id
        
        logger.info("Collective Memory system initialized")
    
    async def initialize(self) -> bool:
        """Initialize the collective memory system"""
        try:
            # Set default quotas
            self.robot_memory_quotas["default"] = 100000000  # 100MB per robot
            
            # Initialize memory structures
            await self._initialize_memory_structures()
            
            # Start background tasks
            self.replication_task = asyncio.create_task(self._manage_replication())
            self.cleanup_task = asyncio.create_task(self._cleanup_memories())
            self.consolidation_task = asyncio.create_task(self._consolidate_memories())
            self.sync_task = asyncio.create_task(self._synchronize_memories())
            
            logger.info("Collective Memory system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Collective Memory initialization failed: {e}")
            return False
    
    async def store_memory(self, robot_id: str, memory_type: MemoryType,
                         title: str, content: Any,
                         metadata: Dict[str, Any] = None,
                         tags: Set[str] = None,
                         importance: float = 0.5,
                         persistence: MemoryPersistence = MemoryPersistence.SESSION,
                         access_level: MemoryAccess = MemoryAccess.PUBLIC) -> str:
        """Store a new memory in the collective system"""
        try:
            # Generate unique memory ID
            memory_id = f"mem_{robot_id}_{datetime.now().timestamp()}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
            
            # Check storage quota
            if not await self._check_storage_quota(robot_id, content):
                logger.warning(f"Storage quota exceeded for robot {robot_id}")
                return ""
            
            # Compress content if enabled
            compressed_content = content
            compression_ratio = 1.0
            if self.compression_enabled and isinstance(content, (dict, list, str)):
                compressed_content, compression_ratio = await self._compress_content(content)
            
            # Calculate checksum for integrity
            checksum = await self._calculate_checksum(compressed_content)
            
            # Create memory entry
            memory_entry = MemoryEntry(
                id=memory_id,
                owner_robot=robot_id,
                memory_type=memory_type,
                title=title,
                content=compressed_content,
                metadata=metadata or {},
                tags=tags or set(),
                timestamp=datetime.now(),
                last_accessed=datetime.now(),
                importance=importance,
                persistence=persistence,
                access_level=access_level,
                compression_ratio=compression_ratio,
                checksum=checksum
            )
            
            # Store locally
            self.local_memories[memory_id] = memory_entry
            self.robot_memories[robot_id].add(memory_id)
            
            # Update indexes
            await self._update_memory_indexes(memory_entry)
            
            # Schedule replication if needed
            if persistence != MemoryPersistence.VOLATILE:
                await self._schedule_replication(memory_id)
            
            # Update cache
            if len(self.memory_cache) < self.cache_size_limit:
                self.memory_cache[memory_id] = compressed_content
            
            # Update storage metrics
            content_size = await self._calculate_memory_size(compressed_content)
            self.total_storage_used += content_size
            
            logger.info(f"Memory stored: {memory_id} by {robot_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return ""
    
    async def retrieve_memory(self, memory_id: str, requesting_robot: str) -> Optional[Any]:
        """Retrieve a memory from the collective system"""
        try:
            # Check if memory exists locally
            if memory_id not in self.local_memories:
                # Try to fetch from replicas
                success = await self._fetch_from_replicas(memory_id)
                if not success:
                    logger.warning(f"Memory {memory_id} not found")
                    return None
            
            memory_entry = self.local_memories[memory_id]
            
            # Check access permissions
            if not await self._check_access_permission(memory_id, requesting_robot):
                logger.warning(f"Access denied for robot {requesting_robot} to memory {memory_id}")
                return None
            
            # Update access statistics
            memory_entry.access_count += 1
            memory_entry.last_accessed = datetime.now()
            self.robot_access_patterns[requesting_robot][memory_id] += 1
            
            # Decompress content if needed
            content = memory_entry.content
            if memory_entry.compression_ratio < 1.0:
                content = await self._decompress_content(memory_entry.content)
            
            # Add to cache for faster future access
            if memory_id not in self.memory_cache and len(self.memory_cache) < self.cache_size_limit:
                self.memory_cache[memory_id] = content
            
            logger.debug(f"Memory retrieved: {memory_id} by {requesting_robot}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def search_memories(self, requesting_robot: str,
                            query: str = None,
                            memory_types: List[MemoryType] = None,
                            tags: Set[str] = None,
                            importance_threshold: float = 0.0,
                            max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for memories matching specified criteria"""
        try:
            matching_memories = []
            
            for memory_id, memory_entry in self.local_memories.items():
                # Check access permissions
                if not await self._check_access_permission(memory_id, requesting_robot):
                    continue
                
                # Apply filters
                if memory_types and memory_entry.memory_type not in memory_types:
                    continue
                
                if memory_entry.importance < importance_threshold:
                    continue
                
                if tags and not tags.intersection(memory_entry.tags):
                    continue
                
                # Text search in title and metadata
                if query:
                    searchable_text = f"{memory_entry.title} {json.dumps(memory_entry.metadata)}"
                    if query.lower() not in searchable_text.lower():
                        continue
                
                # Calculate relevance score
                relevance_score = await self._calculate_relevance_score(memory_entry, query, tags)
                
                memory_info = {
                    "id": memory_id,
                    "title": memory_entry.title,
                    "type": memory_entry.memory_type.value,
                    "owner": memory_entry.owner_robot,
                    "timestamp": memory_entry.timestamp.isoformat(),
                    "importance": memory_entry.importance,
                    "tags": list(memory_entry.tags),
                    "relevance_score": relevance_score,
                    "access_count": memory_entry.access_count
                }
                
                matching_memories.append(memory_info)
            
            # Sort by relevance and importance
            matching_memories.sort(key=lambda x: (x["relevance_score"], x["importance"]), reverse=True)
            
            return matching_memories[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def update_memory(self, memory_id: str, updating_robot: str,
                          new_content: Any = None,
                          new_metadata: Dict[str, Any] = None,
                          add_tags: Set[str] = None,
                          remove_tags: Set[str] = None) -> bool:
        """Update an existing memory"""
        try:
            if memory_id not in self.local_memories:
                logger.error(f"Memory {memory_id} not found for update")
                return False
            
            memory_entry = self.local_memories[memory_id]
            
            # Check write permissions
            if not await self._check_write_permission(memory_id, updating_robot):
                logger.warning(f"Write access denied for robot {updating_robot} to memory {memory_id}")
                return False
            
            # Acquire lock
            if not await self._acquire_memory_lock(memory_id, updating_robot):
                logger.warning(f"Cannot acquire lock for memory {memory_id}")
                return False
            
            try:
                # Update content
                if new_content is not None:
                    if self.compression_enabled:
                        compressed_content, compression_ratio = await self._compress_content(new_content)
                        memory_entry.content = compressed_content
                        memory_entry.compression_ratio = compression_ratio
                    else:
                        memory_entry.content = new_content
                    
                    # Update checksum
                    memory_entry.checksum = await self._calculate_checksum(memory_entry.content)
                    
                    # Update cache
                    if memory_id in self.memory_cache:
                        self.memory_cache[memory_id] = new_content
                
                # Update metadata
                if new_metadata:
                    memory_entry.metadata.update(new_metadata)
                
                # Update tags
                if add_tags:
                    memory_entry.tags.update(add_tags)
                if remove_tags:
                    memory_entry.tags.difference_update(remove_tags)
                
                # Update version and timestamps
                memory_entry.version += 1
                memory_entry.last_accessed = datetime.now()
                
                # Update indexes
                await self._update_memory_indexes(memory_entry)
                
                # Schedule replication of updated memory
                await self._schedule_replication(memory_id)
                
                logger.info(f"Memory updated: {memory_id} by {updating_robot}")
                return True
                
            finally:
                # Release lock
                await self._release_memory_lock(memory_id, updating_robot)
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str, deleting_robot: str) -> bool:
        """Delete a memory from the collective system"""
        try:
            if memory_id not in self.local_memories:
                logger.warning(f"Memory {memory_id} not found for deletion")
                return True  # Already deleted
            
            memory_entry = self.local_memories[memory_id]
            
            # Check delete permissions (only owner or admin can delete)
            if memory_entry.owner_robot != deleting_robot:
                logger.warning(f"Delete access denied for robot {deleting_robot} to memory {memory_id}")
                return False
            
            # Remove from local storage
            del self.local_memories[memory_id]
            
            # Remove from robot's memory set
            self.robot_memories[memory_entry.owner_robot].discard(memory_id)
            
            # Remove from indexes
            await self._remove_from_indexes(memory_entry)
            
            # Remove from cache
            self.memory_cache.pop(memory_id, None)
            
            # Schedule deletion from replicas
            await self._schedule_replica_deletion(memory_id)
            
            # Update storage metrics
            content_size = await self._calculate_memory_size(memory_entry.content)
            self.total_storage_used -= content_size
            
            logger.info(f"Memory deleted: {memory_id} by {deleting_robot}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def create_memory_cluster(self, cluster_name: str, memory_ids: List[str],
                                  cluster_type: str = "thematic") -> str:
        """Create a cluster of related memories"""
        try:
            cluster_id = f"cluster_{hashlib.md5(cluster_name.encode()).hexdigest()[:8]}_{datetime.now().timestamp()}"
            
            # Validate memory IDs
            valid_memory_ids = set()
            for memory_id in memory_ids:
                if memory_id in self.local_memories:
                    valid_memory_ids.add(memory_id)
                else:
                    logger.warning(f"Memory {memory_id} not found for clustering")
            
            if not valid_memory_ids:
                logger.error("No valid memories for clustering")
                return ""
            
            # Calculate cluster importance
            total_importance = sum(self.local_memories[mid].importance for mid in valid_memory_ids)
            avg_importance = total_importance / len(valid_memory_ids)
            
            # Create cluster
            cluster = MemoryCluster(
                id=cluster_id,
                name=cluster_name,
                memory_ids=valid_memory_ids,
                cluster_type=cluster_type,
                importance_score=avg_importance,
                last_updated=datetime.now(),
                access_pattern={}
            )
            
            self.memory_clusters[cluster_id] = cluster
            
            # Update memory relationships
            for memory_id in valid_memory_ids:
                memory_entry = self.local_memories[memory_id]
                memory_entry.related_memories.update(valid_memory_ids - {memory_id})
            
            logger.info(f"Memory cluster created: {cluster_id} with {len(valid_memory_ids)} memories")
            return cluster_id
            
        except Exception as e:
            logger.error(f"Failed to create memory cluster: {e}")
            return ""
    
    async def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory system statistics"""
        try:
            # Count memories by type
            memory_by_type = {}
            for memory_entry in self.local_memories.values():
                mem_type = memory_entry.memory_type.value
                memory_by_type[mem_type] = memory_by_type.get(mem_type, 0) + 1
            
            # Calculate average compression ratio
            compression_ratios = [m.compression_ratio for m in self.local_memories.values()]
            avg_compression = np.mean(compression_ratios) if compression_ratios else 1.0
            
            # Calculate replication factor
            total_replicas = sum(len(replicas) for replicas in self.memory_replicas.values())
            replication_factor = total_replicas / max(1, len(self.local_memories))
            
            # Access frequency analysis
            access_frequency = {}
            for robot_id, patterns in self.robot_access_patterns.items():
                for memory_id, count in patterns.items():
                    access_frequency[memory_id] = access_frequency.get(memory_id, 0) + count
            
            # Calculate memory health score
            health_score = await self._calculate_memory_health()
            
            return MemoryStats(
                total_memories=len(self.local_memories),
                memory_by_type=memory_by_type,
                storage_used=self.total_storage_used,
                storage_available=self.total_storage_available - self.total_storage_used,
                compression_ratio=avg_compression,
                replication_factor=replication_factor,
                access_frequency=access_frequency,
                memory_health_score=health_score
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, {}, 0, 0, 1.0, 0.0, {}, 0.0)
    
    async def store_event(self, event: Dict[str, Any]) -> bool:
        """Store an event in collective memory"""
        try:
            # Extract event details
            event_type = event.get("type", "unknown")
            timestamp = event.get("timestamp", datetime.now().isoformat())
            details = event.get("details", {})
            
            # Create memory entry for the event
            memory_id = await self.store_memory(
                robot_id="system",
                memory_type=MemoryType.EPISODIC,
                title=f"Event: {event_type}",
                content=event,
                metadata={
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "source": "system"
                },
                tags={event_type, "event", "system"},
                importance=0.7,
                persistence=MemoryPersistence.PERMANENT,
                access_level=MemoryAccess.PUBLIC
            )
            
            return bool(memory_id)
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False
    
    async def get_size(self) -> int:
        """Get total size of collective memory"""
        return len(self.local_memories)
    
    async def persist_state(self) -> bool:
        """Persist memory state for recovery"""
        try:
            # In production, this would save to persistent storage
            logger.info(f"Persisting memory state with {len(self.local_memories)} memories")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist memory state: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown collective memory system"""
        try:
            logger.info("Shutting down Collective Memory system")
            
            # Stop background tasks
            if self.replication_task:
                self.replication_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.consolidation_task:
                self.consolidation_task.cancel()
            if self.sync_task:
                self.sync_task.cancel()
            
            # Persist critical memories
            await self.persist_state()
            
            # Clear memory structures
            self.local_memories.clear()
            self.memory_cache.clear()
            self.memory_clusters.clear()
            
            logger.info("Collective Memory system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Collective Memory shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _initialize_memory_structures(self):
        """Initialize memory data structures"""
        try:
            # Initialize index structures
            self.memory_index.clear()
            
            # Initialize access patterns
            self.robot_access_patterns.clear()
            
            # Initialize replication structures
            self.memory_replicas.clear()
            self.pending_replications.clear()
            
        except Exception as e:
            logger.error(f"Error initializing memory structures: {e}")
    
    async def _check_storage_quota(self, robot_id: str, content: Any) -> bool:
        """Check if robot has sufficient storage quota"""
        try:
            quota = self.robot_memory_quotas.get(robot_id, self.robot_memory_quotas["default"])
            
            # Calculate current usage for robot
            current_usage = 0
            for memory_id in self.robot_memories[robot_id]:
                if memory_id in self.local_memories:
                    memory_size = await self._calculate_memory_size(self.local_memories[memory_id].content)
                    current_usage += memory_size
            
            # Calculate new content size
            new_content_size = await self._calculate_memory_size(content)
            
            return current_usage + new_content_size <= quota
            
        except Exception as e:
            logger.error(f"Error checking storage quota: {e}")
            return False
    
    async def _calculate_memory_size(self, content: Any) -> int:
        """Calculate approximate memory size of content"""
        try:
            if isinstance(content, (str, bytes)):
                return len(content)
            elif isinstance(content, (dict, list)):
                return len(json.dumps(content).encode('utf-8'))
            elif hasattr(content, '__sizeof__'):
                return content.__sizeof__()
            else:
                return len(str(content).encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error calculating memory size: {e}")
            return 1000  # Default estimate
    
    async def _compress_content(self, content: Any) -> Tuple[bytes, float]:
        """Compress content and return compressed data with compression ratio"""
        try:
            # Serialize content
            if isinstance(content, (dict, list)):
                serialized = json.dumps(content).encode('utf-8')
            elif isinstance(content, str):
                serialized = content.encode('utf-8')
            else:
                serialized = pickle.dumps(content)
            
            # Compress using gzip
            compressed = gzip.compress(serialized)
            
            # Calculate compression ratio
            compression_ratio = len(compressed) / len(serialized) if len(serialized) > 0 else 1.0
            
            return compressed, compression_ratio
            
        except Exception as e:
            logger.error(f"Error compressing content: {e}")
            return pickle.dumps(content), 1.0
    
    async def _decompress_content(self, compressed_content: bytes) -> Any:
        """Decompress content"""
        try:
            # Decompress using gzip
            decompressed = gzip.decompress(compressed_content)
            
            # Try to deserialize as JSON first, then pickle
            try:
                return json.loads(decompressed.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    return decompressed.decode('utf-8')
                except UnicodeDecodeError:
                    return pickle.loads(decompressed)
                    
        except Exception as e:
            logger.error(f"Error decompressing content: {e}")
            return None
    
    async def _calculate_checksum(self, content: Any) -> str:
        """Calculate checksum for content integrity"""
        try:
            if isinstance(content, bytes):
                data = content
            else:
                data = pickle.dumps(content)
            
            return hashlib.sha256(data).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    async def _update_memory_indexes(self, memory_entry: MemoryEntry):
        """Update memory indexes for search and retrieval"""
        try:
            memory_id = memory_entry.id
            
            # Index by tags
            for tag in memory_entry.tags:
                self.memory_index[tag].add(memory_id)
            
            # Index by memory type
            self.memory_index[memory_entry.memory_type.value].add(memory_id)
            
            # Index by owner
            self.memory_index[f"owner:{memory_entry.owner_robot}"].add(memory_id)
            
            # Index by access level
            self.memory_index[f"access:{memory_entry.access_level.value}"].add(memory_id)
            
        except Exception as e:
            logger.error(f"Error updating memory indexes: {e}")
    
    async def _remove_from_indexes(self, memory_entry: MemoryEntry):
        """Remove memory from all indexes"""
        try:
            memory_id = memory_entry.id
            
            # Remove from all index sets
            for index_set in self.memory_index.values():
                index_set.discard(memory_id)
            
            # Clean up empty index entries
            empty_keys = [key for key, value in self.memory_index.items() if not value]
            for key in empty_keys:
                del self.memory_index[key]
                
        except Exception as e:
            logger.error(f"Error removing from indexes: {e}")
    
    async def _check_access_permission(self, memory_id: str, robot_id: str) -> bool:
        """Check if robot has permission to access memory"""
        try:
            if memory_id not in self.local_memories:
                return False
            
            memory_entry = self.local_memories[memory_id]
            
            # Check access level
            if memory_entry.access_level == MemoryAccess.PUBLIC:
                return True
            elif memory_entry.access_level == MemoryAccess.PRIVATE:
                return memory_entry.owner_robot == robot_id
            elif memory_entry.access_level == MemoryAccess.RESTRICTED:
                # Check explicit permissions
                return self.access_permissions.get(memory_id, {}).get(robot_id, False)
            else:  # GROUP access
                # In a real implementation, this would check group membership
                return True
            
        except Exception as e:
            logger.error(f"Error checking access permission: {e}")
            return False
    
    async def _check_write_permission(self, memory_id: str, robot_id: str) -> bool:
        """Check if robot has permission to modify memory"""
        try:
            if memory_id not in self.local_memories:
                return False
            
            memory_entry = self.local_memories[memory_id]
            
            # Only owner can write (simplified policy)
            return memory_entry.owner_robot == robot_id
            
        except Exception as e:
            logger.error(f"Error checking write permission: {e}")
            return False
    
    async def _acquire_memory_lock(self, memory_id: str, robot_id: str) -> bool:
        """Acquire exclusive lock on memory for updates"""
        try:
            if memory_id in self.memory_locks:
                # Already locked by another robot
                return False
            
            self.memory_locks[memory_id] = robot_id
            return True
            
        except Exception as e:
            logger.error(f"Error acquiring memory lock: {e}")
            return False
    
    async def _release_memory_lock(self, memory_id: str, robot_id: str):
        """Release exclusive lock on memory"""
        try:
            if self.memory_locks.get(memory_id) == robot_id:
                del self.memory_locks[memory_id]
                
        except Exception as e:
            logger.error(f"Error releasing memory lock: {e}")
    
    async def _calculate_relevance_score(self, memory_entry: MemoryEntry, 
                                       query: str = None, tags: Set[str] = None) -> float:
        """Calculate relevance score for search results"""
        try:
            score = 0.0
            
            # Base score from importance
            score += memory_entry.importance * 0.3
            
            # Access frequency score
            score += min(memory_entry.access_count / 100.0, 0.3)
            
            # Tag matching score
            if tags:
                tag_overlap = len(memory_entry.tags.intersection(tags))
                score += (tag_overlap / len(tags)) * 0.2 if tags else 0
            
            # Query matching score
            if query:
                searchable_text = f"{memory_entry.title} {json.dumps(memory_entry.metadata)}"
                query_words = query.lower().split()
                matches = sum(1 for word in query_words if word in searchable_text.lower())
                score += (matches / len(query_words)) * 0.2 if query_words else 0
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    async def _schedule_replication(self, memory_id: str):
        """Schedule memory for replication across the swarm"""
        try:
            self.pending_replications.append(memory_id)
            
        except Exception as e:
            logger.error(f"Error scheduling replication: {e}")
    
    async def _schedule_replica_deletion(self, memory_id: str):
        """Schedule deletion of memory replicas"""
        try:
            # In a real implementation, this would coordinate deletion across replicas
            if memory_id in self.memory_replicas:
                del self.memory_replicas[memory_id]
                
        except Exception as e:
            logger.error(f"Error scheduling replica deletion: {e}")
    
    async def _fetch_from_replicas(self, memory_id: str) -> bool:
        """Fetch memory from replicas if not available locally"""
        try:
            # In a real implementation, this would request from replica robots
            # For now, return False (memory not found)
            return False
            
        except Exception as e:
            logger.error(f"Error fetching from replicas: {e}")
            return False
    
    async def _calculate_memory_health(self) -> float:
        """Calculate overall memory system health score"""
        try:
            health_factors = []
            
            # Storage utilization (optimal around 70%)
            storage_utilization = self.total_storage_used / self.total_storage_available
            storage_health = 1.0 - abs(storage_utilization - 0.7) / 0.7
            health_factors.append(max(0.0, storage_health))
            
            # Compression efficiency
            if self.average_compression_ratio > 0:
                compression_health = min(1.0, (1.0 - self.average_compression_ratio) * 2)
                health_factors.append(compression_health)
            
            # Memory distribution (check for balanced distribution)
            if self.robot_memories:
                memory_counts = [len(memories) for memories in self.robot_memories.values()]
                mean_count = np.mean(memory_counts)
                std_count = np.std(memory_counts)
                distribution_health = 1.0 - min(1.0, std_count / max(mean_count, 1))
                health_factors.append(distribution_health)
            
            # Access pattern health (memories should be accessed regularly)
            recent_accesses = 0
            for memory_entry in self.local_memories.values():
                if (datetime.now() - memory_entry.last_accessed).days < 7:
                    recent_accesses += 1
            
            access_health = recent_accesses / max(len(self.local_memories), 1)
            health_factors.append(access_health)
            
            return np.mean(health_factors) if health_factors else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating memory health: {e}")
            return 0.5
    
    # Background task methods
    
    async def _manage_replication(self):
        """Background task to manage memory replication"""
        while True:
            try:
                # Process pending replications
                while self.pending_replications:
                    memory_id = self.pending_replications.popleft()
                    await self._replicate_memory(memory_id)
                
                # Check replication health
                await self._check_replication_health()
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in replication management: {e}")
    
    async def _replicate_memory(self, memory_id: str):
        """Replicate a memory to other robots"""
        try:
            if memory_id not in self.local_memories:
                return
            
            # In a real implementation, this would:
            # 1. Select target robots for replication
            # 2. Send memory data to target robots
            # 3. Verify successful replication
            # 4. Update replication tracking
            
            # For now, just track that replication was attempted
            self.memory_replicas[memory_id].add("replica_robot_1")
            self.memory_replicas[memory_id].add("replica_robot_2")
            
        except Exception as e:
            logger.error(f"Error replicating memory {memory_id}: {e}")
    
    async def _check_replication_health(self):
        """Check health of memory replication"""
        try:
            under_replicated = []
            for memory_id, memory_entry in self.local_memories.items():
                if memory_entry.persistence != MemoryPersistence.VOLATILE:
                    replica_count = len(self.memory_replicas.get(memory_id, set()))
                    if replica_count < self.replication_factor:
                        under_replicated.append(memory_id)
            
            # Re-replicate under-replicated memories
            for memory_id in under_replicated:
                await self._schedule_replication(memory_id)
            
        except Exception as e:
            logger.error(f"Error checking replication health: {e}")
    
    async def _cleanup_memories(self):
        """Background task to clean up old and unused memories"""
        while True:
            try:
                if not self.auto_cleanup_enabled:
                    await asyncio.sleep(3600)  # Sleep 1 hour if cleanup disabled
                    continue
                
                current_time = datetime.now()
                memories_to_delete = []
                
                for memory_id, memory_entry in self.local_memories.items():
                    # Delete volatile memories older than 1 hour
                    if (memory_entry.persistence == MemoryPersistence.VOLATILE and
                        (current_time - memory_entry.timestamp).seconds > 3600):
                        memories_to_delete.append(memory_id)
                    
                    # Delete session memories older than 24 hours
                    elif (memory_entry.persistence == MemoryPersistence.SESSION and
                          (current_time - memory_entry.timestamp).days > 1):
                        memories_to_delete.append(memory_id)
                    
                    # Archive old permanent memories with low importance
                    elif (memory_entry.persistence == MemoryPersistence.PERMANENT and
                          memory_entry.importance < 0.3 and
                          (current_time - memory_entry.last_accessed).days > 30):
                        memory_entry.persistence = MemoryPersistence.ARCHIVED
                
                # Delete identified memories
                for memory_id in memories_to_delete:
                    await self.delete_memory(memory_id, "system")
                
                if memories_to_delete:
                    logger.info(f"Cleaned up {len(memories_to_delete)} old memories")
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup: {e}")
    
    async def _consolidate_memories(self):
        """Background task to consolidate related memories"""
        while True:
            try:
                # Find memories that could be consolidated
                await self._find_consolidation_opportunities()
                
                # Update memory clusters
                await self._update_memory_clusters()
                
                await asyncio.sleep(1800)  # Consolidate every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory consolidation: {e}")
    
    async def _find_consolidation_opportunities(self):
        """Find opportunities to consolidate related memories"""
        try:
            # Group memories by similarity
            similar_groups = defaultdict(list)
            
            for memory_id, memory_entry in self.local_memories.items():
                # Simple grouping by tags and type
                group_key = f"{memory_entry.memory_type.value}_{hash(frozenset(memory_entry.tags))}"
                similar_groups[group_key].append(memory_id)
            
            # Create clusters for groups with multiple memories
            for group_key, memory_ids in similar_groups.items():
                if len(memory_ids) >= 3 and len(memory_ids) <= 10:  # Reasonable cluster size
                    cluster_name = f"Auto-cluster {group_key}"
                    await self.create_memory_cluster(cluster_name, memory_ids, "auto-generated")
            
        except Exception as e:
            logger.error(f"Error finding consolidation opportunities: {e}")
    
    async def _update_memory_clusters(self):
        """Update existing memory clusters"""
        try:
            current_time = datetime.now()
            
            for cluster_id, cluster in self.memory_clusters.items():
                # Remove non-existent memories from clusters
                valid_memory_ids = set()
                for memory_id in cluster.memory_ids:
                    if memory_id in self.local_memories:
                        valid_memory_ids.add(memory_id)
                
                cluster.memory_ids = valid_memory_ids
                cluster.last_updated = current_time
                
                # Remove empty clusters
                if not cluster.memory_ids:
                    del self.memory_clusters[cluster_id]
                    break
            
        except Exception as e:
            logger.error(f"Error updating memory clusters: {e}")
    
    async def _synchronize_memories(self):
        """Background task to synchronize memories across replicas"""
        while True:
            try:
                # In a real implementation, this would:
                # 1. Check for consistency across replicas
                # 2. Resolve conflicts
                # 3. Update outdated replicas
                # 4. Handle network partitions
                
                await asyncio.sleep(300)  # Sync every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory synchronization: {e}")