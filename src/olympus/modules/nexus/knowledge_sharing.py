"""
Knowledge Sharing - Distributed Experience Exchange System
=========================================================

The Knowledge Sharing module enables robots in the swarm to share experiences,
learned skills, observations, and insights with each other, creating a
collective knowledge base that benefits the entire swarm.

Features:
- Experience packaging and distribution
- Skill transfer and replication
- Observation sharing and validation
- Knowledge graph construction
- Selective sharing based on relevance
- Knowledge verification and validation
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge that can be shared"""
    EXPERIENCE = "experience"
    SKILL = "skill"
    OBSERVATION = "observation"
    STRATEGY = "strategy"
    ERROR_PATTERN = "error_pattern"
    SOLUTION = "solution"
    ENVIRONMENT_MAP = "environment_map"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class ShareScope(Enum):
    """Scope of knowledge sharing"""
    INDIVIDUAL = "individual"
    LOCAL_GROUP = "local_group" 
    SWARM_WIDE = "swarm_wide"
    GLOBAL = "global"


class KnowledgeRelevance(Enum):
    """Relevance levels for knowledge filtering"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ARCHIVE = "archive"


@dataclass
class KnowledgePacket:
    """A packet of knowledge to be shared"""
    id: str
    source_robot: str
    knowledge_type: KnowledgeType
    title: str
    content: Dict[str, Any]
    context: Dict[str, Any]
    relevance: KnowledgeRelevance
    confidence: float
    created_time: datetime
    expiry_time: Optional[datetime]
    verification_count: int = 0
    verification_score: float = 0.0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    recipients: Set[str] = field(default_factory=set)
    share_scope: ShareScope = ShareScope.SWARM_WIDE


@dataclass
class KnowledgeRequest:
    """Request for specific knowledge"""
    id: str
    requesting_robot: str
    knowledge_types: List[KnowledgeType]
    context_filters: Dict[str, Any]
    urgency: str
    max_results: int
    created_time: datetime


@dataclass
class KnowledgeGraph:
    """Graph structure for knowledge relationships"""
    nodes: Dict[str, KnowledgePacket]
    edges: Dict[str, List[str]]  # knowledge_id -> [related_knowledge_ids]
    clusters: Dict[str, Set[str]]  # topic -> {knowledge_ids}
    relevance_scores: Dict[str, float]


class KnowledgeSharing:
    """
    Distributed knowledge sharing system for robot swarms
    
    Enables robots to share experiences, skills, and insights while
    maintaining relevance filtering and verification mechanisms.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Knowledge storage
        self.knowledge_base: Dict[str, KnowledgePacket] = {}
        self.knowledge_graph = KnowledgeGraph(
            nodes={},
            edges={},
            clusters={},
            relevance_scores={}
        )
        
        # Robot knowledge profiles
        self.robot_profiles: Dict[str, Dict[str, Any]] = {}
        self.robot_interests: Dict[str, Set[str]] = {}
        self.robot_expertise: Dict[str, Dict[str, float]] = {}
        
        # Sharing queues and processing
        self.sharing_queue = asyncio.Queue()
        self.request_queue = asyncio.Queue()
        self.verification_queue = asyncio.Queue()
        
        # Background tasks
        self.processing_task = None
        self.cleanup_task = None
        self.analytics_task = None
        
        # Metrics
        self.knowledge_utilization = {}
        self.sharing_efficiency = 0.0
        self.verification_accuracy = 0.0
        
        # Filtering and relevance
        self.relevance_filters = {}
        self.context_matchers = {}
        
        logger.info("Knowledge Sharing system initialized")
    
    async def initialize(self) -> bool:
        """Initialize the knowledge sharing system"""
        try:
            # Initialize relevance filters
            await self._initialize_relevance_filters()
            
            # Start background processing
            self.processing_task = asyncio.create_task(self._process_knowledge_sharing())
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_knowledge())
            self.analytics_task = asyncio.create_task(self._analyze_knowledge_patterns())
            
            logger.info("Knowledge Sharing system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge Sharing initialization failed: {e}")
            return False
    
    async def register_robot(self, robot_id: str, profile: Dict[str, Any] = None) -> bool:
        """Register a robot for knowledge sharing"""
        try:
            self.robot_profiles[robot_id] = profile or {}
            self.robot_interests[robot_id] = set()
            self.robot_expertise[robot_id] = {}
            self.knowledge_utilization[robot_id] = {
                "shared": 0,
                "received": 0,
                "applied": 0,
                "verified": 0
            }
            
            logger.info(f"Robot {robot_id} registered for knowledge sharing")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register robot {robot_id}: {e}")
            return False
    
    async def share_knowledge(self, source_robot: str, knowledge_type: KnowledgeType,
                            title: str, content: Dict[str, Any],
                            context: Dict[str, Any] = None,
                            relevance: KnowledgeRelevance = KnowledgeRelevance.MEDIUM,
                            confidence: float = 1.0,
                            share_scope: ShareScope = ShareScope.SWARM_WIDE,
                            expiry_hours: int = 24) -> str:
        """Share knowledge with other robots in the swarm"""
        try:
            if source_robot not in self.robot_profiles:
                logger.error(f"Robot {source_robot} not registered")
                return ""
            
            # Create knowledge packet
            knowledge_id = f"knowledge_{source_robot}_{datetime.now().timestamp()}"
            expiry_time = datetime.now() + timedelta(hours=expiry_hours)
            
            packet = KnowledgePacket(
                id=knowledge_id,
                source_robot=source_robot,
                knowledge_type=knowledge_type,
                title=title,
                content=content,
                context=context or {},
                relevance=relevance,
                confidence=confidence,
                created_time=datetime.now(),
                expiry_time=expiry_time,
                share_scope=share_scope,
                tags=self._extract_tags(content, context)
            )
            
            # Store in knowledge base
            self.knowledge_base[knowledge_id] = packet
            
            # Add to knowledge graph
            await self._add_to_knowledge_graph(packet)
            
            # Queue for distribution
            await self.sharing_queue.put(knowledge_id)
            
            # Update metrics
            self.knowledge_utilization[source_robot]["shared"] += 1
            
            logger.info(f"Knowledge shared by {source_robot}: {title}")
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Failed to share knowledge: {e}")
            return ""
    
    async def request_knowledge(self, requesting_robot: str,
                              knowledge_types: List[KnowledgeType],
                              context_filters: Dict[str, Any] = None,
                              urgency: str = "normal",
                              max_results: int = 10) -> List[KnowledgePacket]:
        """Request relevant knowledge from the swarm"""
        try:
            if requesting_robot not in self.robot_profiles:
                logger.error(f"Robot {requesting_robot} not registered")
                return []
            
            # Create knowledge request
            request_id = f"req_{requesting_robot}_{datetime.now().timestamp()}"
            request = KnowledgeRequest(
                id=request_id,
                requesting_robot=requesting_robot,
                knowledge_types=knowledge_types,
                context_filters=context_filters or {},
                urgency=urgency,
                max_results=max_results,
                created_time=datetime.now()
            )
            
            # Process request immediately for urgent requests
            if urgency == "urgent":
                results = await self._process_knowledge_request(request)
            else:
                # Queue for processing
                await self.request_queue.put(request)
                # Wait briefly for processing
                await asyncio.sleep(1.0)
                results = await self._process_knowledge_request(request)
            
            # Update metrics
            self.knowledge_utilization[requesting_robot]["received"] += len(results)
            
            logger.info(f"Knowledge request from {requesting_robot} returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process knowledge request: {e}")
            return []
    
    async def share_experience(self, robot_id: str, action: str, outcome: Dict[str, Any],
                             context: Dict[str, Any] = None, success: bool = True) -> str:
        """Share an experience (action-outcome pair)"""
        try:
            experience_content = {
                "action": action,
                "outcome": outcome,
                "success": success,
                "learned_insights": self._extract_insights(action, outcome, success)
            }
            
            relevance = KnowledgeRelevance.HIGH if success else KnowledgeRelevance.MEDIUM
            confidence = 0.9 if success else 0.7
            
            return await self.share_knowledge(
                source_robot=robot_id,
                knowledge_type=KnowledgeType.EXPERIENCE,
                title=f"Experience: {action}",
                content=experience_content,
                context=context,
                relevance=relevance,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to share experience: {e}")
            return ""
    
    async def share_skill(self, robot_id: str, skill_name: str, skill_parameters: Dict[str, Any],
                         performance_metrics: Dict[str, float],
                         prerequisites: List[str] = None) -> str:
        """Share a learned skill with the swarm"""
        try:
            skill_content = {
                "skill_name": skill_name,
                "parameters": skill_parameters,
                "performance_metrics": performance_metrics,
                "prerequisites": prerequisites or [],
                "transferability": self._assess_skill_transferability(skill_parameters)
            }
            
            return await self.share_knowledge(
                source_robot=robot_id,
                knowledge_type=KnowledgeType.SKILL,
                title=f"Skill: {skill_name}",
                content=skill_content,
                relevance=KnowledgeRelevance.HIGH,
                confidence=self._calculate_skill_confidence(performance_metrics)
            )
            
        except Exception as e:
            logger.error(f"Failed to share skill: {e}")
            return ""
    
    async def verify_knowledge(self, verifier_robot: str, knowledge_id: str,
                             verification_result: bool, confidence: float = 1.0,
                             notes: str = None) -> bool:
        """Verify shared knowledge based on own experience"""
        try:
            if knowledge_id not in self.knowledge_base:
                logger.error(f"Knowledge {knowledge_id} not found")
                return False
            
            packet = self.knowledge_base[knowledge_id]
            
            # Update verification metrics
            packet.verification_count += 1
            
            # Update verification score (running average)
            current_score = packet.verification_score
            verification_value = 1.0 if verification_result else 0.0
            packet.verification_score = (
                (current_score * (packet.verification_count - 1) + 
                 verification_value * confidence) / packet.verification_count
            )
            
            # Store verification details
            verification_entry = {
                "verifier": verifier_robot,
                "result": verification_result,
                "confidence": confidence,
                "notes": notes,
                "timestamp": datetime.now().isoformat()
            }
            
            if "verifications" not in packet.content:
                packet.content["verifications"] = []
            packet.content["verifications"].append(verification_entry)
            
            # Update robot expertise
            knowledge_topic = packet.knowledge_type.value
            if knowledge_topic not in self.robot_expertise[verifier_robot]:
                self.robot_expertise[verifier_robot][knowledge_topic] = 0.0
            
            self.robot_expertise[verifier_robot][knowledge_topic] += 0.1 * confidence
            
            # Update metrics
            self.knowledge_utilization[verifier_robot]["verified"] += 1
            
            logger.info(f"Knowledge {knowledge_id} verified by {verifier_robot}: {verification_result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify knowledge: {e}")
            return False
    
    async def get_relevant_knowledge(self, robot_id: str, context: Dict[str, Any],
                                   knowledge_types: List[KnowledgeType] = None,
                                   max_results: int = 5) -> List[KnowledgePacket]:
        """Get knowledge relevant to a specific robot and context"""
        try:
            # Filter by knowledge types if specified
            candidates = []
            for packet in self.knowledge_base.values():
                if knowledge_types and packet.knowledge_type not in knowledge_types:
                    continue
                if packet.source_robot == robot_id:  # Don't return own knowledge
                    continue
                candidates.append(packet)
            
            # Calculate relevance scores
            scored_knowledge = []
            for packet in candidates:
                relevance_score = await self._calculate_relevance_score(packet, robot_id, context)
                if relevance_score > 0.3:  # Minimum relevance threshold
                    scored_knowledge.append((packet, relevance_score))
            
            # Sort by relevance and return top results
            scored_knowledge.sort(key=lambda x: x[1], reverse=True)
            return [packet for packet, score in scored_knowledge[:max_results]]
            
        except Exception as e:
            logger.error(f"Failed to get relevant knowledge: {e}")
            return []
    
    async def apply_knowledge(self, robot_id: str, knowledge_id: str,
                            application_context: Dict[str, Any]) -> Dict[str, Any]:
        """Mark knowledge as applied and provide feedback"""
        try:
            if knowledge_id not in self.knowledge_base:
                return {"success": False, "reason": "knowledge_not_found"}
            
            packet = self.knowledge_base[knowledge_id]
            
            # Record application
            application_entry = {
                "applier": robot_id,
                "context": application_context,
                "timestamp": datetime.now().isoformat()
            }
            
            if "applications" not in packet.content:
                packet.content["applications"] = []
            packet.content["applications"].append(application_entry)
            
            # Update metrics
            self.knowledge_utilization[robot_id]["applied"] += 1
            
            # Update robot interests based on application
            packet_tags = packet.tags
            self.robot_interests[robot_id].update(packet_tags)
            
            logger.info(f"Knowledge {knowledge_id} applied by {robot_id}")
            return {
                "success": True,
                "knowledge_type": packet.knowledge_type.value,
                "application_count": len(packet.content.get("applications", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to record knowledge application: {e}")
            return {"success": False, "reason": str(e)}
    
    async def get_knowledge_analytics(self) -> Dict[str, Any]:
        """Get analytics about knowledge sharing patterns"""
        try:
            total_knowledge = len(self.knowledge_base)
            verified_knowledge = sum(1 for p in self.knowledge_base.values() if p.verification_count > 0)
            
            knowledge_by_type = {}
            for packet in self.knowledge_base.values():
                k_type = packet.knowledge_type.value
                knowledge_by_type[k_type] = knowledge_by_type.get(k_type, 0) + 1
            
            top_sharers = sorted(
                [(robot_id, stats["shared"]) for robot_id, stats in self.knowledge_utilization.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            return {
                "total_knowledge_packets": total_knowledge,
                "verified_knowledge": verified_knowledge,
                "verification_rate": verified_knowledge / max(1, total_knowledge),
                "knowledge_by_type": knowledge_by_type,
                "top_sharers": top_sharers,
                "sharing_efficiency": self.sharing_efficiency,
                "verification_accuracy": self.verification_accuracy,
                "active_robots": len([r for r, s in self.knowledge_utilization.items() if s["shared"] > 0])
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge analytics: {e}")
            return {}
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown knowledge sharing system"""
        try:
            logger.info("Shutting down Knowledge Sharing system")
            
            # Stop background tasks
            if self.processing_task:
                self.processing_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.analytics_task:
                self.analytics_task.cancel()
            
            # Save knowledge base state
            await self._save_knowledge_state()
            
            logger.info("Knowledge Sharing system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge Sharing shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _process_knowledge_sharing(self):
        """Background task to process knowledge sharing"""
        while True:
            try:
                # Process sharing queue
                try:
                    knowledge_id = await asyncio.wait_for(self.sharing_queue.get(), timeout=1.0)
                    await self._distribute_knowledge(knowledge_id)
                except asyncio.TimeoutError:
                    pass
                
                # Process request queue
                try:
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                    await self._process_knowledge_request(request)
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing knowledge sharing: {e}")
    
    async def _distribute_knowledge(self, knowledge_id: str):
        """Distribute knowledge to relevant robots"""
        try:
            if knowledge_id not in self.knowledge_base:
                return
            
            packet = self.knowledge_base[knowledge_id]
            
            # Determine recipients based on scope and relevance
            recipients = await self._find_knowledge_recipients(packet)
            
            packet.recipients.update(recipients)
            
            logger.debug(f"Distributed knowledge {knowledge_id} to {len(recipients)} robots")
            
        except Exception as e:
            logger.error(f"Error distributing knowledge {knowledge_id}: {e}")
    
    async def _find_knowledge_recipients(self, packet: KnowledgePacket) -> Set[str]:
        """Find robots that should receive this knowledge"""
        recipients = set()
        
        # Filter by share scope
        if packet.share_scope == ShareScope.INDIVIDUAL:
            return recipients
        elif packet.share_scope == ShareScope.LOCAL_GROUP:
            # In a real implementation, this would consider proximity/groups
            pass
        
        # Find robots with relevant interests
        packet_tags = packet.tags
        for robot_id, interests in self.robot_interests.items():
            if robot_id == packet.source_robot:  # Don't send back to source
                continue
            
            # Check interest overlap
            interest_overlap = len(packet_tags.intersection(interests))
            if interest_overlap > 0:
                recipients.add(robot_id)
            
            # Check expertise relevance
            robot_expertise = self.robot_expertise.get(robot_id, {})
            knowledge_topic = packet.knowledge_type.value
            if knowledge_topic in robot_expertise and robot_expertise[knowledge_topic] > 0.5:
                recipients.add(robot_id)
        
        return recipients
    
    async def _process_knowledge_request(self, request: KnowledgeRequest) -> List[KnowledgePacket]:
        """Process a knowledge request and return relevant results"""
        try:
            results = []
            
            for packet in self.knowledge_base.values():
                # Filter by knowledge type
                if packet.knowledge_type not in request.knowledge_types:
                    continue
                
                # Filter by context
                if not self._matches_context_filters(packet, request.context_filters):
                    continue
                
                # Calculate relevance score
                relevance_score = await self._calculate_relevance_score(
                    packet, request.requesting_robot, request.context_filters
                )
                
                if relevance_score > 0.3:  # Minimum relevance
                    results.append((packet, relevance_score))
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return [packet for packet, score in results[:request.max_results]]
            
        except Exception as e:
            logger.error(f"Error processing knowledge request: {e}")
            return []
    
    async def _calculate_relevance_score(self, packet: KnowledgePacket, robot_id: str,
                                       context: Dict[str, Any]) -> float:
        """Calculate relevance score for knowledge packet"""
        try:
            score = 0.0
            
            # Base score from packet relevance level
            relevance_weights = {
                KnowledgeRelevance.CRITICAL: 1.0,
                KnowledgeRelevance.HIGH: 0.8,
                KnowledgeRelevance.MEDIUM: 0.6,
                KnowledgeRelevance.LOW: 0.4,
                KnowledgeRelevance.ARCHIVE: 0.2
            }
            score += relevance_weights[packet.relevance] * 0.3
            
            # Confidence boost
            score += packet.confidence * 0.2
            
            # Verification boost
            if packet.verification_count > 0:
                score += packet.verification_score * 0.2
            
            # Interest alignment
            robot_interests = self.robot_interests.get(robot_id, set())
            interest_overlap = len(packet.tags.intersection(robot_interests))
            if robot_interests:
                score += (interest_overlap / len(robot_interests)) * 0.2
            
            # Context similarity
            if context and packet.context:
                context_similarity = self._calculate_context_similarity(packet.context, context)
                score += context_similarity * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def _extract_tags(self, content: Dict[str, Any], context: Dict[str, Any] = None) -> Set[str]:
        """Extract relevant tags from knowledge content"""
        tags = set()
        
        # Extract from content keys and values
        for key, value in content.items():
            tags.add(key.lower())
            if isinstance(value, str):
                # Simple keyword extraction
                words = value.lower().split()
                tags.update(word for word in words if len(word) > 3)
        
        # Extract from context
        if context:
            for key, value in context.items():
                tags.add(key.lower())
                if isinstance(value, str):
                    words = value.lower().split()
                    tags.update(word for word in words if len(word) > 3)
        
        return tags
    
    def _extract_insights(self, action: str, outcome: Dict[str, Any], success: bool) -> List[str]:
        """Extract insights from an experience"""
        insights = []
        
        if success:
            insights.append(f"Action '{action}' led to successful outcome")
            if "performance" in outcome:
                insights.append(f"Performance metrics: {outcome['performance']}")
        else:
            insights.append(f"Action '{action}' resulted in failure")
            if "error" in outcome:
                insights.append(f"Error encountered: {outcome['error']}")
        
        return insights
    
    def _assess_skill_transferability(self, skill_parameters: Dict[str, Any]) -> float:
        """Assess how transferable a skill is to other robots"""
        # Simple heuristic based on parameter complexity
        param_count = len(skill_parameters)
        if param_count < 3:
            return 0.9  # Simple skills are highly transferable
        elif param_count < 10:
            return 0.6  # Moderate complexity
        else:
            return 0.3  # Complex skills are less transferable
    
    def _calculate_skill_confidence(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate confidence in a skill based on performance metrics"""
        if not performance_metrics:
            return 0.5
        
        # Average of all performance metrics
        return np.mean(list(performance_metrics.values()))
    
    def _matches_context_filters(self, packet: KnowledgePacket, filters: Dict[str, Any]) -> bool:
        """Check if packet matches context filters"""
        if not filters:
            return True
        
        packet_context = packet.context
        for key, expected_value in filters.items():
            if key not in packet_context:
                continue
            
            packet_value = packet_context[key]
            if packet_value != expected_value:
                return False
        
        return True
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()).intersection(set(context2.keys()))
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if context1[key] == context2[key])
        return matches / len(common_keys)
    
    async def _add_to_knowledge_graph(self, packet: KnowledgePacket):
        """Add knowledge packet to the knowledge graph"""
        try:
            self.knowledge_graph.nodes[packet.id] = packet
            self.knowledge_graph.edges[packet.id] = []
            
            # Find related knowledge for graph connections
            related_ids = await self._find_related_knowledge(packet)
            self.knowledge_graph.edges[packet.id] = list(related_ids)
            
            # Update clusters
            for tag in packet.tags:
                if tag not in self.knowledge_graph.clusters:
                    self.knowledge_graph.clusters[tag] = set()
                self.knowledge_graph.clusters[tag].add(packet.id)
            
        except Exception as e:
            logger.error(f"Error adding to knowledge graph: {e}")
    
    async def _find_related_knowledge(self, packet: KnowledgePacket) -> Set[str]:
        """Find knowledge related to the given packet"""
        related = set()
        
        for existing_id, existing_packet in self.knowledge_base.items():
            if existing_id == packet.id:
                continue
            
            # Check for tag overlap
            tag_overlap = len(packet.tags.intersection(existing_packet.tags))
            if tag_overlap >= 2:  # At least 2 common tags
                related.add(existing_id)
            
            # Check for same knowledge type and similar context
            if (packet.knowledge_type == existing_packet.knowledge_type and
                self._calculate_context_similarity(packet.context, existing_packet.context) > 0.7):
                related.add(existing_id)
        
        return related
    
    async def _initialize_relevance_filters(self):
        """Initialize relevance filtering mechanisms"""
        self.relevance_filters = {
            "time_decay": 0.1,  # Decay factor for old knowledge
            "verification_boost": 0.2,  # Boost for verified knowledge
            "confidence_weight": 0.3  # Weight of confidence in relevance
        }
    
    async def _cleanup_expired_knowledge(self):
        """Background task to clean up expired knowledge"""
        while True:
            try:
                current_time = datetime.now()
                expired_ids = []
                
                for knowledge_id, packet in self.knowledge_base.items():
                    if packet.expiry_time and current_time > packet.expiry_time:
                        expired_ids.append(knowledge_id)
                
                for knowledge_id in expired_ids:
                    del self.knowledge_base[knowledge_id]
                    if knowledge_id in self.knowledge_graph.nodes:
                        del self.knowledge_graph.nodes[knowledge_id]
                        del self.knowledge_graph.edges[knowledge_id]
                
                if expired_ids:
                    logger.info(f"Cleaned up {len(expired_ids)} expired knowledge packets")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error cleaning up expired knowledge: {e}")
    
    async def _analyze_knowledge_patterns(self):
        """Background task to analyze knowledge sharing patterns"""
        while True:
            try:
                # Update sharing efficiency
                total_shared = sum(stats["shared"] for stats in self.knowledge_utilization.values())
                total_applied = sum(stats["applied"] for stats in self.knowledge_utilization.values())
                
                if total_shared > 0:
                    self.sharing_efficiency = total_applied / total_shared
                
                # Update verification accuracy
                verified_packets = [p for p in self.knowledge_base.values() if p.verification_count > 0]
                if verified_packets:
                    self.verification_accuracy = np.mean([p.verification_score for p in verified_packets])
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error analyzing knowledge patterns: {e}")
    
    async def _save_knowledge_state(self):
        """Save knowledge state for persistence"""
        try:
            # In production, this would save to persistent storage
            logger.info(f"Saving knowledge state with {len(self.knowledge_base)} packets")
            
        except Exception as e:
            logger.error(f"Error saving knowledge state: {e}")