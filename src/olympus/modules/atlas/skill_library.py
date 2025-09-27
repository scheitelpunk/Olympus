"""
Skill Library Module

Manages a repository of reusable skills with hierarchical organization,
version control, and intelligent retrieval capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class SkillType(Enum):
    COGNITIVE = "cognitive"
    MOTOR = "motor" 
    PERCEPTUAL = "perceptual"
    SOCIAL = "social"
    REASONING = "reasoning"
    PLANNING = "planning"
    LEARNING = "learning"
    COMMUNICATION = "communication"

class SkillComplexity(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class SkillStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"

@dataclass
class SkillMetadata:
    """Metadata for a skill"""
    tags: List[str] = field(default_factory=list)
    domain: str = "general"
    complexity: SkillComplexity = SkillComplexity.BASIC
    prerequisites: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    creation_date: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    
@dataclass
class Skill:
    """Represents a reusable skill"""
    skill_id: str
    name: str
    description: str
    skill_type: SkillType
    implementation: Any  # The actual skill implementation
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    status: SkillStatus = SkillStatus.ACTIVE
    parent_skills: List[str] = field(default_factory=list)
    child_skills: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.skill_id:
            # Generate unique skill ID
            content = f"{self.name}_{self.skill_type.value}_{datetime.now().isoformat()}"
            self.skill_id = hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class SkillSearchQuery:
    """Query for searching skills"""
    keywords: List[str] = field(default_factory=list)
    skill_types: List[SkillType] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    complexity_range: Tuple[SkillComplexity, SkillComplexity] = None
    min_performance: float = 0.0
    exclude_deprecated: bool = True
    limit: int = 10

@dataclass
class SkillSearchResult:
    """Result of skill search"""
    skills: List[Skill]
    relevance_scores: Dict[str, float]
    total_matches: int
    query_time: float
    suggestions: List[str] = field(default_factory=list)

class SkillLibrary:
    """Centralized repository for reusable skills"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.skills: Dict[str, Skill] = {}
        self.skill_index = defaultdict(set)  # For efficient searching
        self.hierarchy_cache = {}
        self.usage_history = []
        self.version_history = defaultdict(list)
        
        # Search and retrieval components
        self.semantic_index = {}  # For semantic similarity search
        self.performance_cache = {}  # Cache skill performance data
        self.recommendation_engine = None
        
    async def add_skill(self, skill: Skill, 
                       replace_existing: bool = False) -> Dict[str, Any]:
        """Add a new skill to the library"""
        
        logger.info(f"Adding skill: {skill.name} ({skill.skill_id})")
        
        # Validate skill
        validation_result = await self._validate_skill(skill)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": f"Skill validation failed: {validation_result['reason']}"
            }
        
        # Check for existing skill
        if skill.skill_id in self.skills and not replace_existing:
            return {
                "success": False,
                "error": f"Skill {skill.skill_id} already exists. Use replace_existing=True to overwrite."
            }
        
        # Store previous version if replacing
        if skill.skill_id in self.skills:
            previous_version = self.skills[skill.skill_id]
            self.version_history[skill.skill_id].append(previous_version)
        
        # Add skill to library
        self.skills[skill.skill_id] = skill
        
        # Update indices
        await self._update_indices(skill)
        
        # Update skill relationships
        await self._update_skill_relationships(skill)
        
        # Clear caches that might be affected
        await self._invalidate_caches(skill)
        
        # Record addition
        await self._record_skill_addition(skill)
        
        return {
            "success": True,
            "skill_id": skill.skill_id,
            "message": f"Skill '{skill.name}' added successfully"
        }
    
    async def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID"""
        skill = self.skills.get(skill_id)
        
        if skill:
            # Update usage statistics
            skill.metadata.usage_count += 1
            skill.metadata.last_used = datetime.now()
            
            # Record usage
            await self._record_skill_usage(skill)
            
        return skill
    
    async def search_skills(self, query: SkillSearchQuery) -> SkillSearchResult:
        """Search for skills based on query criteria"""
        
        start_time = datetime.now()
        logger.info(f"Searching skills with query: {query.keywords}")
        
        # Multi-stage search process
        candidates = set(self.skills.keys())
        
        # Filter by skill types
        if query.skill_types:
            type_matches = set()
            for skill_type in query.skill_types:
                type_matches.update(self.skill_index.get(f"type:{skill_type.value}", set()))
            candidates &= type_matches
        
        # Filter by domains
        if query.domains:
            domain_matches = set()
            for domain in query.domains:
                domain_matches.update(self.skill_index.get(f"domain:{domain}", set()))
            candidates &= domain_matches
        
        # Filter by complexity
        if query.complexity_range:
            complexity_matches = await self._filter_by_complexity(candidates, query.complexity_range)
            candidates &= complexity_matches
        
        # Filter by performance
        if query.min_performance > 0:
            performance_matches = await self._filter_by_performance(candidates, query.min_performance)
            candidates &= performance_matches
        
        # Filter deprecated skills
        if query.exclude_deprecated:
            active_skills = {sid for sid in candidates if self.skills[sid].status != SkillStatus.DEPRECATED}
            candidates &= active_skills
        
        # Keyword-based relevance scoring
        relevance_scores = {}
        if query.keywords:
            for skill_id in candidates:
                score = await self._calculate_relevance_score(self.skills[skill_id], query.keywords)
                relevance_scores[skill_id] = score
        else:
            # Default scoring based on usage and performance
            for skill_id in candidates:
                score = await self._calculate_default_score(self.skills[skill_id])
                relevance_scores[skill_id] = score
        
        # Sort by relevance and apply limit
        sorted_skill_ids = sorted(
            relevance_scores.keys(), 
            key=lambda x: relevance_scores[x], 
            reverse=True
        )[:query.limit]
        
        # Get skill objects
        result_skills = [self.skills[skill_id] for skill_id in sorted_skill_ids]
        
        # Generate suggestions
        suggestions = await self._generate_search_suggestions(query, result_skills)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return SkillSearchResult(
            skills=result_skills,
            relevance_scores={sid: relevance_scores[sid] for sid in sorted_skill_ids},
            total_matches=len(candidates),
            query_time=query_time,
            suggestions=suggestions
        )
    
    async def recommend_skills(self, context: Dict[str, Any], 
                             target_skill_id: Optional[str] = None,
                             limit: int = 5) -> List[Tuple[Skill, float]]:
        """Recommend skills based on context or similarity to target skill"""
        
        if target_skill_id:
            # Recommend similar skills
            recommendations = await self._recommend_similar_skills(target_skill_id, limit)
        else:
            # Recommend based on context
            recommendations = await self._recommend_contextual_skills(context, limit)
        
        return recommendations
    
    async def get_skill_hierarchy(self, root_skill_id: str = None) -> Dict[str, Any]:
        """Get hierarchical view of skills"""
        
        # Check cache first
        cache_key = f"hierarchy_{root_skill_id or 'all'}"
        if cache_key in self.hierarchy_cache:
            return self.hierarchy_cache[cache_key]
        
        if root_skill_id:
            # Get hierarchy for specific skill
            hierarchy = await self._build_skill_subtree(root_skill_id)
        else:
            # Get full hierarchy
            hierarchy = await self._build_full_hierarchy()
        
        # Cache result
        self.hierarchy_cache[cache_key] = hierarchy
        
        return hierarchy
    
    async def update_skill_performance(self, skill_id: str, 
                                     performance_data: Dict[str, float]) -> bool:
        """Update performance metrics for a skill"""
        
        skill = self.skills.get(skill_id)
        if not skill:
            return False
        
        # Update performance metrics
        skill.metadata.performance_metrics.update(performance_data)
        
        # Update performance cache
        self.performance_cache[skill_id] = performance_data
        
        # Clear related caches
        await self._invalidate_performance_caches(skill_id)
        
        # Record performance update
        await self._record_performance_update(skill_id, performance_data)
        
        return True
    
    async def compose_skills(self, skill_ids: List[str], 
                           composition_type: str = "sequential") -> Dict[str, Any]:
        """Compose multiple skills into a complex skill"""
        
        logger.info(f"Composing skills: {skill_ids} using {composition_type}")
        
        # Validate all skills exist
        skills = []
        for skill_id in skill_ids:
            skill = self.skills.get(skill_id)
            if not skill:
                return {
                    "success": False,
                    "error": f"Skill {skill_id} not found"
                }
            skills.append(skill)
        
        # Check compatibility
        compatibility_result = await self._check_skill_compatibility(skills)
        if not compatibility_result["compatible"]:
            return {
                "success": False,
                "error": f"Skills are not compatible: {compatibility_result['reason']}"
            }
        
        # Create composed skill
        composed_skill = await self._create_composed_skill(skills, composition_type)
        
        return {
            "success": True,
            "composed_skill": composed_skill,
            "component_skills": skill_ids,
            "composition_type": composition_type
        }
    
    async def version_skill(self, skill_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new version of an existing skill"""
        
        original_skill = self.skills.get(skill_id)
        if not original_skill:
            return {
                "success": False,
                "error": f"Skill {skill_id} not found"
            }
        
        # Create new version
        new_skill = await self._create_skill_version(original_skill, changes)
        
        # Add to library
        add_result = await self.add_skill(new_skill, replace_existing=True)
        
        if add_result["success"]:
            return {
                "success": True,
                "new_version": new_skill.metadata.version,
                "previous_version": original_skill.metadata.version,
                "skill_id": skill_id
            }
        else:
            return add_result
    
    async def archive_skill(self, skill_id: str, reason: str = "") -> bool:
        """Archive a skill (soft delete)"""
        
        skill = self.skills.get(skill_id)
        if not skill:
            return False
        
        # Change status to archived
        skill.status = SkillStatus.ARCHIVED
        
        # Update metadata
        skill.metadata.tags.append("archived")
        if reason:
            skill.metadata.tags.append(f"archived_reason:{reason}")
        
        # Remove from active indices
        await self._remove_from_active_indices(skill)
        
        # Record archival
        await self._record_skill_archival(skill_id, reason)
        
        return True
    
    async def export_skills(self, skill_ids: List[str] = None, 
                          format_type: str = "json") -> Dict[str, Any]:
        """Export skills in specified format"""
        
        if skill_ids is None:
            skills_to_export = list(self.skills.values())
        else:
            skills_to_export = [self.skills[sid] for sid in skill_ids if sid in self.skills]
        
        if format_type == "json":
            exported_data = await self._export_as_json(skills_to_export)
        elif format_type == "yaml":
            exported_data = await self._export_as_yaml(skills_to_export)
        else:
            return {
                "success": False,
                "error": f"Unsupported export format: {format_type}"
            }
        
        return {
            "success": True,
            "format": format_type,
            "skill_count": len(skills_to_export),
            "data": exported_data
        }
    
    async def import_skills(self, data: Any, format_type: str = "json") -> Dict[str, Any]:
        """Import skills from external data"""
        
        try:
            if format_type == "json":
                skills = await self._import_from_json(data)
            elif format_type == "yaml":
                skills = await self._import_from_yaml(data)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported import format: {format_type}"
                }
            
            # Add imported skills
            imported_count = 0
            errors = []
            
            for skill in skills:
                result = await self.add_skill(skill, replace_existing=False)
                if result["success"]:
                    imported_count += 1
                else:
                    errors.append(f"Failed to import {skill.name}: {result['error']}")
            
            return {
                "success": True,
                "imported_count": imported_count,
                "total_skills": len(skills),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Skill import failed: {e}")
            return {
                "success": False,
                "error": f"Import failed: {str(e)}"
            }
    
    # Helper methods (mock implementations for brevity)
    async def _validate_skill(self, skill):
        return {"valid": True, "reason": ""}
    
    async def _update_indices(self, skill):
        # Update various indices
        self.skill_index[f"type:{skill.skill_type.value}"].add(skill.skill_id)
        self.skill_index[f"domain:{skill.metadata.domain}"].add(skill.skill_id)
        for tag in skill.metadata.tags:
            self.skill_index[f"tag:{tag}"].add(skill.skill_id)
    
    async def _update_skill_relationships(self, skill):
        # Update parent-child relationships
        for parent_id in skill.parent_skills:
            if parent_id in self.skills:
                if skill.skill_id not in self.skills[parent_id].child_skills:
                    self.skills[parent_id].child_skills.append(skill.skill_id)
    
    async def _invalidate_caches(self, skill):
        # Clear relevant caches
        self.hierarchy_cache.clear()
    
    async def _record_skill_addition(self, skill):
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "skill_added",
            "skill_id": skill.skill_id,
            "skill_name": skill.name,
            "skill_type": skill.skill_type.value
        }
        self.usage_history.append(record)
    
    async def _record_skill_usage(self, skill):
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "skill_used",
            "skill_id": skill.skill_id,
            "usage_count": skill.metadata.usage_count
        }
        self.usage_history.append(record)
    
    async def _filter_by_complexity(self, candidates, complexity_range):
        min_complexity, max_complexity = complexity_range
        complexity_order = [SkillComplexity.BASIC, SkillComplexity.INTERMEDIATE, 
                          SkillComplexity.ADVANCED, SkillComplexity.EXPERT]
        min_idx = complexity_order.index(min_complexity)
        max_idx = complexity_order.index(max_complexity)
        
        filtered = set()
        for skill_id in candidates:
            skill_complexity = self.skills[skill_id].metadata.complexity
            complexity_idx = complexity_order.index(skill_complexity)
            if min_idx <= complexity_idx <= max_idx:
                filtered.add(skill_id)
        
        return filtered
    
    async def _filter_by_performance(self, candidates, min_performance):
        filtered = set()
        for skill_id in candidates:
            skill = self.skills[skill_id]
            if skill.metadata.performance_metrics:
                avg_performance = np.mean(list(skill.metadata.performance_metrics.values()))
                if avg_performance >= min_performance:
                    filtered.add(skill_id)
            else:
                # Include skills without performance data if min_performance is 0
                if min_performance == 0:
                    filtered.add(skill_id)
        
        return filtered
    
    async def _calculate_relevance_score(self, skill, keywords):
        score = 0.0
        text_content = f"{skill.name} {skill.description} {' '.join(skill.metadata.tags)}"
        text_content = text_content.lower()
        
        for keyword in keywords:
            keyword = keyword.lower()
            if keyword in text_content:
                score += 1.0
            # Additional scoring based on exact matches, partial matches, etc.
        
        # Boost score based on usage and performance
        score += skill.metadata.usage_count * 0.1
        if skill.metadata.performance_metrics:
            avg_performance = np.mean(list(skill.metadata.performance_metrics.values()))
            score += avg_performance
        
        return score
    
    async def _calculate_default_score(self, skill):
        score = skill.metadata.usage_count * 0.5
        if skill.metadata.performance_metrics:
            avg_performance = np.mean(list(skill.metadata.performance_metrics.values()))
            score += avg_performance * 2.0
        
        # Boost recent skills slightly
        days_since_creation = (datetime.now() - skill.metadata.creation_date).days
        if days_since_creation < 30:
            score += 0.5
        
        return score
    
    async def _generate_search_suggestions(self, query, results):
        suggestions = []
        
        if len(results) == 0:
            suggestions.append("Try broader search terms")
            suggestions.append("Check skill types or domains")
        elif len(results) < 3:
            suggestions.append("Try removing some filters")
            suggestions.append("Consider related skill types")
        
        return suggestions
    
    async def _recommend_similar_skills(self, skill_id, limit):
        target_skill = self.skills.get(skill_id)
        if not target_skill:
            return []
        
        similarities = []
        for other_id, other_skill in self.skills.items():
            if other_id != skill_id:
                similarity = await self._calculate_skill_similarity(target_skill, other_skill)
                similarities.append((other_skill, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    async def _recommend_contextual_skills(self, context, limit):
        # Mock contextual recommendation based on context
        context_keywords = context.get("keywords", [])
        domain = context.get("domain", "general")
        
        recommendations = []
        for skill in self.skills.values():
            score = 0.0
            
            # Score based on domain match
            if skill.metadata.domain == domain:
                score += 2.0
            
            # Score based on keyword matches
            for keyword in context_keywords:
                if keyword.lower() in skill.name.lower() or keyword.lower() in skill.description.lower():
                    score += 1.0
            
            if score > 0:
                recommendations.append((skill, score))
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    async def _calculate_skill_similarity(self, skill1, skill2):
        similarity = 0.0
        
        # Type similarity
        if skill1.skill_type == skill2.skill_type:
            similarity += 0.3
        
        # Domain similarity
        if skill1.metadata.domain == skill2.metadata.domain:
            similarity += 0.2
        
        # Tag overlap
        common_tags = set(skill1.metadata.tags) & set(skill2.metadata.tags)
        if skill1.metadata.tags and skill2.metadata.tags:
            tag_similarity = len(common_tags) / max(len(skill1.metadata.tags), len(skill2.metadata.tags))
            similarity += tag_similarity * 0.3
        
        # Description similarity (mock)
        desc_similarity = np.random.uniform(0, 0.2)  # Would use actual NLP similarity
        similarity += desc_similarity
        
        return similarity
    
    async def _build_skill_subtree(self, skill_id):
        skill = self.skills.get(skill_id)
        if not skill:
            return None
        
        subtree = {
            "skill": skill,
            "children": []
        }
        
        for child_id in skill.child_skills:
            child_subtree = await self._build_skill_subtree(child_id)
            if child_subtree:
                subtree["children"].append(child_subtree)
        
        return subtree
    
    async def _build_full_hierarchy(self):
        # Find root skills (skills with no parents)
        root_skills = [skill for skill in self.skills.values() if not skill.parent_skills]
        
        hierarchy = {
            "roots": []
        }
        
        for root_skill in root_skills:
            subtree = await self._build_skill_subtree(root_skill.skill_id)
            hierarchy["roots"].append(subtree)
        
        return hierarchy
    
    async def _invalidate_performance_caches(self, skill_id):
        # Would invalidate caches related to performance
        pass
    
    async def _record_performance_update(self, skill_id, performance_data):
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "performance_updated",
            "skill_id": skill_id,
            "metrics": performance_data
        }
        self.usage_history.append(record)
    
    async def _check_skill_compatibility(self, skills):
        # Mock compatibility check
        return {"compatible": True, "reason": ""}
    
    async def _create_composed_skill(self, skills, composition_type):
        # Create a new skill that composes the input skills
        composed_name = f"Composed_{composition_type}_" + "_".join([s.name for s in skills[:3]])
        
        return Skill(
            skill_id="",  # Will be auto-generated
            name=composed_name,
            description=f"Composition of {len(skills)} skills using {composition_type} strategy",
            skill_type=SkillType.COGNITIVE,  # Default type for composed skills
            implementation={"type": "composed", "components": [s.skill_id for s in skills]},
            metadata=SkillMetadata(
                tags=["composed", composition_type],
                complexity=SkillComplexity.ADVANCED
            )
        )
    
    async def _create_skill_version(self, original_skill, changes):
        # Create new version with changes applied
        new_skill = Skill(
            skill_id=original_skill.skill_id,  # Keep same ID
            name=original_skill.name,
            description=original_skill.description,
            skill_type=original_skill.skill_type,
            implementation=original_skill.implementation,
            parameters=original_skill.parameters.copy(),
            metadata=original_skill.metadata,
            status=original_skill.status,
            parent_skills=original_skill.parent_skills.copy(),
            child_skills=original_skill.child_skills.copy(),
            related_skills=original_skill.related_skills.copy()
        )
        
        # Apply changes
        for key, value in changes.items():
            if hasattr(new_skill, key):
                setattr(new_skill, key, value)
            elif hasattr(new_skill.metadata, key):
                setattr(new_skill.metadata, key, value)
        
        # Increment version
        version_parts = original_skill.metadata.version.split('.')
        if len(version_parts) == 3:
            major, minor, patch = version_parts
            new_skill.metadata.version = f"{major}.{minor}.{int(patch) + 1}"
        
        return new_skill
    
    async def _remove_from_active_indices(self, skill):
        # Remove from search indices
        for index_key, skill_set in self.skill_index.items():
            skill_set.discard(skill.skill_id)
    
    async def _record_skill_archival(self, skill_id, reason):
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "skill_archived",
            "skill_id": skill_id,
            "reason": reason
        }
        self.usage_history.append(record)
    
    # Export/Import methods (mock implementations)
    async def _export_as_json(self, skills):
        # Would serialize skills to JSON
        return {"skills": [{"id": s.skill_id, "name": s.name} for s in skills]}
    
    async def _export_as_yaml(self, skills):
        # Would serialize skills to YAML
        return f"skills: [{', '.join([s.name for s in skills])}]"
    
    async def _import_from_json(self, data):
        # Would deserialize skills from JSON
        return []
    
    async def _import_from_yaml(self, data):
        # Would deserialize skills from YAML
        return []

    async def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive library statistics"""
        
        stats = {
            "total_skills": len(self.skills),
            "active_skills": len([s for s in self.skills.values() if s.status == SkillStatus.ACTIVE]),
            "deprecated_skills": len([s for s in self.skills.values() if s.status == SkillStatus.DEPRECATED]),
            "skill_types": {},
            "domains": {},
            "complexity_distribution": {},
            "most_used_skills": [],
            "recent_additions": []
        }
        
        # Analyze skill types
        for skill in self.skills.values():
            skill_type = skill.skill_type.value
            stats["skill_types"][skill_type] = stats["skill_types"].get(skill_type, 0) + 1
        
        # Analyze domains
        for skill in self.skills.values():
            domain = skill.metadata.domain
            stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
        
        # Analyze complexity
        for skill in self.skills.values():
            complexity = skill.metadata.complexity.value
            stats["complexity_distribution"][complexity] = stats["complexity_distribution"].get(complexity, 0) + 1
        
        # Most used skills
        skills_by_usage = sorted(self.skills.values(), key=lambda s: s.metadata.usage_count, reverse=True)
        stats["most_used_skills"] = [(s.name, s.metadata.usage_count) for s in skills_by_usage[:10]]
        
        # Recent additions
        recent_skills = sorted(self.skills.values(), key=lambda s: s.metadata.creation_date, reverse=True)
        stats["recent_additions"] = [(s.name, s.metadata.creation_date.isoformat()) for s in recent_skills[:10]]
        
        return stats