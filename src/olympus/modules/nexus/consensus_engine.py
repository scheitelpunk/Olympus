"""
Consensus Engine - Collective Decision Making System
==================================================

The Consensus Engine implements democratic decision-making processes for the
NEXUS collective intelligence system, ensuring fair and transparent collective
decisions while maintaining human authority override.

Features:
- Multiple consensus algorithms (Byzantine fault tolerance, RAFT, etc.)
- Voting mechanisms and quorum management
- Proposal and deliberation systems
- Consensus validation and verification
- Human authority override protection
- Transparency and auditability
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


class ConsensusAlgorithm(Enum):
    """Available consensus algorithms"""
    SIMPLE_MAJORITY = "simple_majority"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    RAFT = "raft"
    PRACTICAL_BFT = "practical_bft"
    WEIGHTED_VOTING = "weighted_voting"
    DELIBERATIVE = "deliberative"


class ProposalStatus(Enum):
    """Status of consensus proposals"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    DELIBERATION = "deliberation"
    VOTING = "voting"
    CONSENSUS_REACHED = "consensus_reached"
    REJECTED = "rejected"
    EXPIRED = "expired"
    HUMAN_OVERRIDE = "human_override"


class VoteType(Enum):
    """Types of votes"""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"
    VETO = "veto"


@dataclass
class Vote:
    """A single vote on a proposal"""
    voter_id: str
    proposal_id: str
    vote_type: VoteType
    weight: float
    reasoning: Optional[str]
    timestamp: datetime
    confidence: float = 1.0


@dataclass
class Proposal:
    """A proposal for collective decision making"""
    id: str
    title: str
    description: str
    proposer_id: str
    proposal_type: str
    content: Dict[str, Any]
    status: ProposalStatus
    created_time: datetime
    deliberation_deadline: Optional[datetime]
    voting_deadline: Optional[datetime]
    required_quorum: float
    consensus_threshold: float
    votes: List[Vote] = field(default_factory=list)
    deliberation_comments: List[Dict[str, Any]] = field(default_factory=list)
    consensus_result: Optional[Dict[str, Any]] = None


@dataclass
class ConsensusState:
    """Current state of the consensus system"""
    active_proposals: Dict[str, Proposal]
    participating_robots: Set[str]
    quorum_requirements: Dict[str, float]
    consensus_thresholds: Dict[str, float]
    algorithm_config: Dict[str, Any]
    voting_weights: Dict[str, float]
    last_consensus: Optional[datetime]


class ConsensusEngine:
    """
    Collective decision-making system for robot swarms
    
    Implements various consensus algorithms to enable democratic decision
    making while ensuring safety, transparency, and human authority.
    """
    
    def __init__(self, config):
        self.config = config
        self.algorithm = ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT
        
        # Consensus state
        self.consensus_state = ConsensusState(
            active_proposals={},
            participating_robots=set(),
            quorum_requirements={},
            consensus_thresholds={},
            algorithm_config={},
            voting_weights={},
            last_consensus=None
        )
        
        # Processing queues
        self.proposal_queue = asyncio.Queue()
        self.vote_queue = asyncio.Queue()
        
        # Background tasks
        self.processing_task = None
        self.monitor_task = None
        
        # Safety and override systems
        self.human_override_active = False
        self.emergency_stop = False
        
        # Metrics
        self.consensus_efficiency = 0.0
        self.decision_accuracy = 0.0
        self.participation_rate = 0.0
        
        # Audit trail
        self.decision_history = []
        self.audit_log = []
        
        logger.info("Consensus Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the consensus engine"""
        try:
            # Set default algorithm parameters
            await self._initialize_algorithm_config()
            
            # Start background processing
            self.processing_task = asyncio.create_task(self._process_proposals())
            self.monitor_task = asyncio.create_task(self._monitor_consensus())
            
            logger.info("Consensus Engine initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Consensus Engine initialization failed: {e}")
            return False
    
    async def register_participant(self, robot_id: str, voting_weight: float = 1.0) -> bool:
        """Register a robot as a consensus participant"""
        try:
            self.consensus_state.participating_robots.add(robot_id)
            self.consensus_state.voting_weights[robot_id] = voting_weight
            
            # Set default quorum and threshold for robot
            self.consensus_state.quorum_requirements[robot_id] = self.config.consensus_threshold
            self.consensus_state.consensus_thresholds[robot_id] = self.config.consensus_threshold
            
            logger.info(f"Robot {robot_id} registered as consensus participant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register participant {robot_id}: {e}")
            return False
    
    async def submit_proposal(self, proposer_id: str, title: str, description: str,
                            proposal_type: str, content: Dict[str, Any],
                            deliberation_minutes: int = 30,
                            voting_minutes: int = 60) -> str:
        """Submit a new proposal for consensus"""
        try:
            if self.human_override_active:
                logger.warning("Proposal submission blocked - human override active")
                return ""
            
            if proposer_id not in self.consensus_state.participating_robots:
                logger.error(f"Robot {proposer_id} not registered as participant")
                return ""
            
            # Create proposal
            proposal_id = f"prop_{proposer_id}_{datetime.now().timestamp()}"
            current_time = datetime.now()
            
            proposal = Proposal(
                id=proposal_id,
                title=title,
                description=description,
                proposer_id=proposer_id,
                proposal_type=proposal_type,
                content=content,
                status=ProposalStatus.SUBMITTED,
                created_time=current_time,
                deliberation_deadline=current_time + timedelta(minutes=deliberation_minutes),
                voting_deadline=current_time + timedelta(minutes=deliberation_minutes + voting_minutes),
                required_quorum=self.config.consensus_threshold,
                consensus_threshold=self.config.consensus_threshold
            )
            
            self.consensus_state.active_proposals[proposal_id] = proposal
            
            # Add to processing queue
            await self.proposal_queue.put(proposal_id)
            
            # Log for audit trail
            await self._log_audit_event("proposal_submitted", {
                "proposal_id": proposal_id,
                "proposer": proposer_id,
                "type": proposal_type,
                "title": title
            })
            
            logger.info(f"Proposal {proposal_id} submitted by {proposer_id}")
            return proposal_id
            
        except Exception as e:
            logger.error(f"Failed to submit proposal: {e}")
            return ""
    
    async def cast_vote(self, voter_id: str, proposal_id: str, vote_type: VoteType,
                       reasoning: str = None, confidence: float = 1.0) -> bool:
        """Cast a vote on a proposal"""
        try:
            if self.human_override_active:
                logger.warning("Voting blocked - human override active")
                return False
            
            if voter_id not in self.consensus_state.participating_robots:
                logger.error(f"Robot {voter_id} not registered as participant")
                return False
            
            if proposal_id not in self.consensus_state.active_proposals:
                logger.error(f"Proposal {proposal_id} not found")
                return False
            
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            # Check if proposal is in voting phase
            if proposal.status != ProposalStatus.VOTING:
                logger.error(f"Proposal {proposal_id} not in voting phase")
                return False
            
            # Check if already voted
            existing_vote = next((v for v in proposal.votes if v.voter_id == voter_id), None)
            if existing_vote:
                logger.warning(f"Robot {voter_id} already voted on {proposal_id}")
                return False
            
            # Create vote
            vote = Vote(
                voter_id=voter_id,
                proposal_id=proposal_id,
                vote_type=vote_type,
                weight=self.consensus_state.voting_weights.get(voter_id, 1.0),
                reasoning=reasoning,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
            proposal.votes.append(vote)
            
            # Add to vote processing queue
            await self.vote_queue.put((proposal_id, vote))
            
            # Log for audit trail
            await self._log_audit_event("vote_cast", {
                "proposal_id": proposal_id,
                "voter": voter_id,
                "vote": vote_type.value,
                "confidence": confidence
            })
            
            logger.info(f"Vote cast by {voter_id} on proposal {proposal_id}: {vote_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cast vote: {e}")
            return False
    
    async def add_deliberation_comment(self, robot_id: str, proposal_id: str,
                                     comment: str, comment_type: str = "general") -> bool:
        """Add a comment during proposal deliberation"""
        try:
            if proposal_id not in self.consensus_state.active_proposals:
                logger.error(f"Proposal {proposal_id} not found")
                return False
            
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            if proposal.status != ProposalStatus.DELIBERATION:
                logger.error(f"Proposal {proposal_id} not in deliberation phase")
                return False
            
            comment_entry = {
                "id": f"comment_{robot_id}_{datetime.now().timestamp()}",
                "author_id": robot_id,
                "content": comment,
                "type": comment_type,
                "timestamp": datetime.now().isoformat(),
                "responses": []
            }
            
            proposal.deliberation_comments.append(comment_entry)
            
            logger.info(f"Deliberation comment added by {robot_id} on proposal {proposal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add deliberation comment: {e}")
            return False
    
    async def build_consensus(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus for a specific action"""
        try:
            if self.human_override_active:
                return {"achieved": False, "reason": "human_override_active"}
            
            if self.emergency_stop:
                return {"achieved": False, "reason": "emergency_stop"}
            
            # Create rapid consensus proposal for action
            proposal_id = await self.submit_proposal(
                proposer_id="system",
                title=f"Action Consensus: {action.get('type', 'unknown')}",
                description=f"Consensus request for action: {action.get('description', '')}",
                proposal_type="action_consensus",
                content=action,
                deliberation_minutes=2,  # Rapid consensus
                voting_minutes=3
            )
            
            if not proposal_id:
                return {"achieved": False, "reason": "proposal_creation_failed"}
            
            # Wait for consensus with timeout
            consensus_result = await self._wait_for_consensus(proposal_id, timeout_seconds=300)
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Failed to build consensus: {e}")
            return {"achieved": False, "reason": str(e)}
    
    async def get_consensus_status(self, proposal_id: str) -> Dict[str, Any]:
        """Get status of a specific proposal consensus"""
        try:
            if proposal_id not in self.consensus_state.active_proposals:
                return {"status": "not_found"}
            
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            # Calculate current vote tallies
            vote_tally = await self._calculate_vote_tally(proposal)
            
            # Calculate participation
            total_participants = len(self.consensus_state.participating_robots)
            votes_cast = len(proposal.votes)
            participation_rate = votes_cast / max(1, total_participants)
            
            return {
                "proposal_id": proposal_id,
                "status": proposal.status.value,
                "title": proposal.title,
                "proposer": proposal.proposer_id,
                "created_time": proposal.created_time.isoformat(),
                "voting_deadline": proposal.voting_deadline.isoformat() if proposal.voting_deadline else None,
                "vote_tally": vote_tally,
                "participation_rate": participation_rate,
                "quorum_met": participation_rate >= proposal.required_quorum,
                "consensus_threshold": proposal.consensus_threshold,
                "deliberation_comments": len(proposal.deliberation_comments),
                "consensus_result": proposal.consensus_result
            }
            
        except Exception as e:
            logger.error(f"Failed to get consensus status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def abort_all_processes(self) -> bool:
        """Abort all active consensus processes (emergency)"""
        try:
            logger.critical("Aborting all consensus processes")
            
            self.emergency_stop = True
            
            # Mark all active proposals as expired
            for proposal in self.consensus_state.active_proposals.values():
                if proposal.status in [ProposalStatus.DELIBERATION, ProposalStatus.VOTING]:
                    proposal.status = ProposalStatus.EXPIRED
                    proposal.consensus_result = {
                        "decision": "aborted",
                        "reason": "emergency_stop",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Clear processing queues
            while not self.proposal_queue.empty():
                try:
                    self.proposal_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            while not self.vote_queue.empty():
                try:
                    self.vote_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            await self._log_audit_event("all_processes_aborted", {
                "reason": "emergency_stop",
                "active_proposals": len(self.consensus_state.active_proposals)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to abort consensus processes: {e}")
            return False
    
    async def human_override_decision(self, proposal_id: str, decision: Dict[str, Any]) -> bool:
        """Override consensus with human decision"""
        try:
            if proposal_id not in self.consensus_state.active_proposals:
                logger.error(f"Proposal {proposal_id} not found for override")
                return False
            
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            # Apply human override
            proposal.status = ProposalStatus.HUMAN_OVERRIDE
            proposal.consensus_result = {
                "decision": decision.get("decision", "override"),
                "reason": "human_authority_override",
                "details": decision,
                "timestamp": datetime.now().isoformat(),
                "override": True
            }
            
            # Log override for audit trail
            await self._log_audit_event("human_override", {
                "proposal_id": proposal_id,
                "original_status": proposal.status.value,
                "override_decision": decision,
                "authority": "human"
            })
            
            logger.warning(f"Human override applied to proposal {proposal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply human override: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive consensus engine status"""
        active_proposals = len([p for p in self.consensus_state.active_proposals.values()
                              if p.status in [ProposalStatus.DELIBERATION, ProposalStatus.VOTING]])
        
        return {
            "algorithm": self.algorithm.value,
            "active_proposals": active_proposals,
            "total_proposals": len(self.consensus_state.active_proposals),
            "participating_robots": len(self.consensus_state.participating_robots),
            "consensus_efficiency": self.consensus_efficiency,
            "decision_accuracy": self.decision_accuracy,
            "participation_rate": self.participation_rate,
            "human_override_active": self.human_override_active,
            "emergency_stop": self.emergency_stop,
            "last_consensus": self.consensus_state.last_consensus.isoformat() if self.consensus_state.last_consensus else None,
            "algorithm_config": self.consensus_state.algorithm_config
        }
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown consensus engine"""
        try:
            logger.info("Shutting down Consensus Engine")
            
            # Stop background tasks
            if self.processing_task:
                self.processing_task.cancel()
            if self.monitor_task:
                self.monitor_task.cancel()
            
            # Complete active proposals
            for proposal in self.consensus_state.active_proposals.values():
                if proposal.status in [ProposalStatus.DELIBERATION, ProposalStatus.VOTING]:
                    proposal.status = ProposalStatus.EXPIRED
            
            # Save audit log and decision history
            await self._save_audit_trail()
            
            logger.info("Consensus Engine shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Consensus Engine shutdown failed: {e}")
            return False
    
    # Private helper methods
    
    async def _initialize_algorithm_config(self):
        """Initialize configuration for consensus algorithms"""
        self.consensus_state.algorithm_config = {
            "byzantine_fault_tolerance": {
                "fault_tolerance": 1/3,  # Can tolerate up to 1/3 faulty nodes
                "message_rounds": 3,
                "timeout_seconds": 30
            },
            "raft": {
                "election_timeout": (5, 10),  # Random timeout range
                "heartbeat_interval": 1,
                "log_replication_batch_size": 100
            },
            "simple_majority": {
                "threshold": 0.51,
                "tie_breaker": "proposer_vote"
            },
            "weighted_voting": {
                "weight_normalization": True,
                "minimum_weight": 0.1
            }
        }
    
    async def _process_proposals(self):
        """Background task to process proposals through consensus phases"""
        while True:
            try:
                # Process proposals from queue
                try:
                    proposal_id = await asyncio.wait_for(self.proposal_queue.get(), timeout=1.0)
                    await self._advance_proposal_phase(proposal_id)
                except asyncio.TimeoutError:
                    pass
                
                # Check for phase transitions
                await self._check_phase_transitions()
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing proposals: {e}")
    
    async def _monitor_consensus(self):
        """Background task to monitor consensus health and metrics"""
        while True:
            try:
                await self._update_consensus_metrics()
                await self._cleanup_expired_proposals()
                await asyncio.sleep(5.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring consensus: {e}")
    
    async def _advance_proposal_phase(self, proposal_id: str):
        """Advance a proposal to the next phase"""
        try:
            if proposal_id not in self.consensus_state.active_proposals:
                return
            
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            if proposal.status == ProposalStatus.SUBMITTED:
                proposal.status = ProposalStatus.DELIBERATION
                await self._log_audit_event("proposal_phase_change", {
                    "proposal_id": proposal_id,
                    "new_phase": "deliberation"
                })
            
        except Exception as e:
            logger.error(f"Error advancing proposal phase: {e}")
    
    async def _check_phase_transitions(self):
        """Check if any proposals should transition phases"""
        current_time = datetime.now()
        
        for proposal in self.consensus_state.active_proposals.values():
            try:
                if proposal.status == ProposalStatus.DELIBERATION:
                    if proposal.deliberation_deadline and current_time >= proposal.deliberation_deadline:
                        proposal.status = ProposalStatus.VOTING
                        await self._log_audit_event("proposal_phase_change", {
                            "proposal_id": proposal.id,
                            "new_phase": "voting"
                        })
                
                elif proposal.status == ProposalStatus.VOTING:
                    if proposal.voting_deadline and current_time >= proposal.voting_deadline:
                        await self._finalize_consensus(proposal.id)
                
            except Exception as e:
                logger.error(f"Error checking phase transition for {proposal.id}: {e}")
    
    async def _finalize_consensus(self, proposal_id: str):
        """Finalize consensus for a proposal"""
        try:
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            # Calculate final results
            vote_tally = await self._calculate_vote_tally(proposal)
            consensus_result = await self._determine_consensus_result(proposal, vote_tally)
            
            proposal.consensus_result = consensus_result
            
            if consensus_result["decision"] == "accepted":
                proposal.status = ProposalStatus.CONSENSUS_REACHED
            else:
                proposal.status = ProposalStatus.REJECTED
            
            self.consensus_state.last_consensus = datetime.now()
            
            # Add to decision history
            self.decision_history.append({
                "proposal_id": proposal_id,
                "title": proposal.title,
                "result": consensus_result,
                "timestamp": datetime.now().isoformat(),
                "vote_tally": vote_tally
            })
            
            await self._log_audit_event("consensus_finalized", {
                "proposal_id": proposal_id,
                "result": consensus_result["decision"],
                "vote_tally": vote_tally
            })
            
        except Exception as e:
            logger.error(f"Error finalizing consensus for {proposal_id}: {e}")
    
    async def _calculate_vote_tally(self, proposal: Proposal) -> Dict[str, float]:
        """Calculate weighted vote tally"""
        tally = {"yes": 0.0, "no": 0.0, "abstain": 0.0, "veto": 0.0}
        
        for vote in proposal.votes:
            weighted_vote = vote.weight * vote.confidence
            tally[vote.vote_type.value] += weighted_vote
        
        return tally
    
    async def _determine_consensus_result(self, proposal: Proposal, vote_tally: Dict[str, float]) -> Dict[str, Any]:
        """Determine consensus result based on algorithm"""
        try:
            total_weight = sum(vote_tally.values())
            if total_weight == 0:
                return {
                    "decision": "rejected",
                    "reason": "no_votes",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check for veto
            if vote_tally["veto"] > 0:
                return {
                    "decision": "rejected", 
                    "reason": "vetoed",
                    "veto_count": vote_tally["veto"],
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate approval ratio
            approval_votes = vote_tally["yes"]
            rejection_votes = vote_tally["no"] 
            decisive_votes = approval_votes + rejection_votes
            
            if decisive_votes == 0:
                return {
                    "decision": "rejected",
                    "reason": "all_abstentions",
                    "timestamp": datetime.now().isoformat()
                }
            
            approval_ratio = approval_votes / decisive_votes
            
            if approval_ratio >= proposal.consensus_threshold:
                return {
                    "decision": "accepted",
                    "approval_ratio": approval_ratio,
                    "threshold": proposal.consensus_threshold,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "decision": "rejected",
                    "reason": "insufficient_consensus",
                    "approval_ratio": approval_ratio,
                    "threshold": proposal.consensus_threshold,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error determining consensus result: {e}")
            return {
                "decision": "rejected",
                "reason": "calculation_error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _wait_for_consensus(self, proposal_id: str, timeout_seconds: int = 300) -> Dict[str, Any]:
        """Wait for consensus to be reached on a proposal"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout_seconds:
            if proposal_id not in self.consensus_state.active_proposals:
                return {"achieved": False, "reason": "proposal_not_found"}
            
            proposal = self.consensus_state.active_proposals[proposal_id]
            
            if proposal.status == ProposalStatus.CONSENSUS_REACHED:
                return {
                    "achieved": True,
                    "result": proposal.consensus_result,
                    "proposal_id": proposal_id
                }
            elif proposal.status in [ProposalStatus.REJECTED, ProposalStatus.EXPIRED, ProposalStatus.HUMAN_OVERRIDE]:
                return {
                    "achieved": False,
                    "reason": proposal.status.value,
                    "result": proposal.consensus_result
                }
            
            await asyncio.sleep(1.0)
        
        return {"achieved": False, "reason": "timeout"}
    
    async def _update_consensus_metrics(self):
        """Update consensus performance metrics"""
        try:
            # Calculate consensus efficiency
            recent_proposals = [p for p in self.consensus_state.active_proposals.values()
                              if (datetime.now() - p.created_time).days <= 1]
            
            if recent_proposals:
                completed = sum(1 for p in recent_proposals 
                              if p.status == ProposalStatus.CONSENSUS_REACHED)
                self.consensus_efficiency = completed / len(recent_proposals)
            
            # Calculate participation rate
            if self.consensus_state.participating_robots:
                active_voters = set()
                for proposal in recent_proposals:
                    active_voters.update(vote.voter_id for vote in proposal.votes)
                
                self.participation_rate = len(active_voters) / len(self.consensus_state.participating_robots)
            
        except Exception as e:
            logger.error(f"Error updating consensus metrics: {e}")
    
    async def _cleanup_expired_proposals(self):
        """Clean up expired proposals"""
        try:
            current_time = datetime.now()
            expired_ids = []
            
            for proposal_id, proposal in self.consensus_state.active_proposals.items():
                if (proposal.voting_deadline and 
                    current_time > proposal.voting_deadline + timedelta(hours=24) and
                    proposal.status not in [ProposalStatus.CONSENSUS_REACHED, ProposalStatus.HUMAN_OVERRIDE]):
                    expired_ids.append(proposal_id)
            
            for proposal_id in expired_ids:
                del self.consensus_state.active_proposals[proposal_id]
            
            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired proposals")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired proposals: {e}")
    
    async def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log an event to the audit trail"""
        audit_entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "hash": hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
        }
        
        self.audit_log.append(audit_entry)
        
        # Maintain audit log size
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep most recent 5000 entries
    
    async def _save_audit_trail(self):
        """Save audit trail for persistence"""
        try:
            # In production, this would save to persistent storage
            logger.info(f"Saving audit trail with {len(self.audit_log)} entries")
            
        except Exception as e:
            logger.error(f"Error saving audit trail: {e}")