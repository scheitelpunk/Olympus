"""PostgreSQL storage layer for MORPHEUS system.

Provides complete database interface for storing experiences, materials,
dream sessions, and learned strategies with optimized queries and connection pooling.
"""

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
from psycopg2 import sql
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import asdict
import uuid
import logging
import json
from contextlib import contextmanager
import threading
from functools import wraps

from ..core.types import (
    SensoryExperience, LearnedStrategy, MaterialInteraction,
    DreamSessionConfig, SystemMetrics
)

logger = logging.getLogger(__name__)


def handle_db_errors(func):
    """Decorator to handle database errors gracefully."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except psycopg2.Error as e:
            logger.error(f"Database error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper


class PostgreSQLStorage:
    """PostgreSQL interface for MORPHEUS system with connection pooling and optimization."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        """Initialize PostgreSQL storage with connection pooling.
        
        Args:
            connection_params: Dictionary with connection parameters
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Database user
                - password: Database password
                - pool_size: Connection pool size (default: 20)
                - max_overflow: Maximum overflow connections (default: 10)
        """
        self.connection_params = connection_params.copy()
        
        # Extract pool configuration
        self.min_connections = 1
        self.max_connections = connection_params.get('pool_size', 20)
        
        # Create threaded connection pool for multi-threaded access
        try:
            self.pool = ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                host=connection_params['host'],
                port=connection_params.get('port', 5432),
                database=connection_params['database'],
                user=connection_params['user'],
                password=connection_params['password'],
                application_name='MORPHEUS'
            )
            logger.info(f"PostgreSQL connection pool created: {self.min_connections}-{self.max_connections}")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
        
        # Thread-local storage for connection management
        self._local = threading.local()
        
        # Initialize database schema
        self.init_schema()
        logger.info("PostgreSQLStorage initialized successfully")
        
    @contextmanager
    def get_connection(self):
        """Context manager for getting database connections from pool."""
        conn = None
        try:
            conn = self.pool.getconn()
            if conn:
                # Test connection
                conn.autocommit = False
                yield conn
            else:
                raise psycopg2.Error("Failed to get connection from pool")
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Failed to return connection to pool: {e}")
    
    @handle_db_errors
    def init_schema(self):
        """Create all required tables, indices, and stored procedures."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Enable required extensions
                cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                cur.execute("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";")
                cur.execute("CREATE EXTENSION IF NOT EXISTS \"btree_gin\";")
                
                # Main experiences table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS experiences (
                        id SERIAL PRIMARY KEY,
                        experience_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                        session_id UUID NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        
                        -- Temporal data
                        timestamp FLOAT NOT NULL,
                        duration_ms FLOAT DEFAULT 0,
                        
                        -- Sensory embeddings (using arrays for efficient storage)
                        tactile_embedding FLOAT[],
                        audio_embedding FLOAT[],
                        visual_embedding FLOAT[],
                        fused_embedding FLOAT[],
                        
                        -- Raw sensory data (JSONB for flexibility and indexing)
                        tactile_data JSONB,
                        audio_data JSONB,
                        visual_data JSONB,
                        
                        -- Material information
                        primary_material VARCHAR(100),
                        secondary_material VARCHAR(100),
                        material_interaction JSONB,
                        
                        -- Physics data
                        contact_points JSONB,
                        forces FLOAT[],
                        torques FLOAT[],
                        
                        -- Action & outcome
                        action_type VARCHAR(50),
                        action_params JSONB,
                        success BOOLEAN DEFAULT TRUE,
                        reward FLOAT DEFAULT 0,
                        
                        -- Metadata
                        tags TEXT[],
                        notes TEXT,
                        
                        -- Performance tracking
                        processing_time_ms FLOAT,
                        data_size_bytes INTEGER
                    );
                    
                    -- Trigger for updated_at
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = NOW();
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                    
                    DROP TRIGGER IF EXISTS update_experiences_updated_at ON experiences;
                    CREATE TRIGGER update_experiences_updated_at
                        BEFORE UPDATE ON experiences
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                """)
                
                # Optimized indices for experiences
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_exp_created_at ON experiences(created_at DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_session ON experiences(session_id);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_experience_id ON experiences(experience_id);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiences(timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_materials ON experiences(primary_material, secondary_material);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_success ON experiences(success) WHERE success = true;",
                    "CREATE INDEX IF NOT EXISTS idx_exp_action_type ON experiences(action_type);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_tags ON experiences USING GIN(tags);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_tactile_data ON experiences USING GIN(tactile_data);",
                    "CREATE INDEX IF NOT EXISTS idx_exp_reward ON experiences(reward DESC) WHERE reward > 0;",
                ]
                
                for index in indices:
                    cur.execute(index)
                
                # Material predictions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS material_predictions (
                        id SERIAL PRIMARY KEY,
                        prediction_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        
                        -- Materials
                        material1 VARCHAR(100) NOT NULL,
                        material2 VARCHAR(100) NOT NULL,
                        interaction_type VARCHAR(50) DEFAULT 'contact',
                        
                        -- Predicted properties
                        predicted_friction FLOAT,
                        predicted_restitution FLOAT,
                        predicted_hardness FLOAT,
                        predicted_sound_freq FLOAT,
                        predicted_texture VARCHAR(50),
                        predicted_temperature_feel FLOAT,
                        
                        -- Actual measurements (when available)
                        actual_friction FLOAT,
                        actual_restitution FLOAT,
                        actual_hardness FLOAT,
                        actual_sound_freq FLOAT,
                        actual_texture VARCHAR(50),
                        actual_temperature_feel FLOAT,
                        
                        -- Prediction quality
                        confidence FLOAT DEFAULT 0,
                        error_metrics JSONB,
                        validation_count INTEGER DEFAULT 0,
                        
                        -- Source
                        source_experience_id INTEGER REFERENCES experiences(id) ON DELETE SET NULL,
                        
                        -- Scenario context
                        scenario_params JSONB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_mat_pred_materials 
                        ON material_predictions(material1, material2);
                    CREATE INDEX IF NOT EXISTS idx_mat_pred_confidence 
                        ON material_predictions(confidence DESC);
                    CREATE INDEX IF NOT EXISTS idx_mat_pred_created 
                        ON material_predictions(created_at DESC);
                """)
                
                # Dream sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dream_sessions (
                        id SERIAL PRIMARY KEY,
                        session_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        ended_at TIMESTAMPTZ,
                        
                        -- Configuration
                        config JSONB NOT NULL,
                        
                        -- Input parameters
                        experience_count INTEGER DEFAULT 0,
                        time_range_hours FLOAT DEFAULT 24,
                        source_session_ids UUID[],
                        
                        -- Process metrics
                        replay_count INTEGER DEFAULT 0,
                        variations_generated INTEGER DEFAULT 0,
                        compute_time_seconds FLOAT DEFAULT 0,
                        memory_usage_mb FLOAT DEFAULT 0,
                        
                        -- Results
                        strategies_found INTEGER DEFAULT 0,
                        strategies_stored INTEGER DEFAULT 0,
                        improvements JSONB,
                        consolidated_memories INTEGER DEFAULT 0,
                        
                        -- Quality metrics
                        average_improvement FLOAT DEFAULT 0,
                        best_improvement FLOAT DEFAULT 0,
                        convergence_iterations INTEGER,
                        
                        -- Status
                        status VARCHAR(20) DEFAULT 'running',
                        error_message TEXT,
                        
                        notes TEXT
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_dream_started ON dream_sessions(started_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_dream_status ON dream_sessions(status);
                    CREATE INDEX IF NOT EXISTS idx_dream_improvement ON dream_sessions(best_improvement DESC);
                """)
                
                # Learned strategies table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS learned_strategies (
                        id SERIAL PRIMARY KEY,
                        strategy_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        dream_session_id UUID REFERENCES dream_sessions(session_id) ON DELETE SET NULL,
                        
                        -- Strategy identification
                        name VARCHAR(200) NOT NULL,
                        category VARCHAR(50) DEFAULT 'general',
                        description TEXT,
                        version INTEGER DEFAULT 1,
                        
                        -- The actual strategy data
                        strategy_data JSONB NOT NULL,
                        
                        -- Performance metrics
                        baseline_performance FLOAT DEFAULT 0,
                        improved_performance FLOAT DEFAULT 0,
                        improvement_ratio FLOAT DEFAULT 0,
                        confidence FLOAT DEFAULT 0,
                        statistical_significance FLOAT,
                        
                        -- Applicability
                        applicable_materials TEXT[],
                        applicable_scenarios TEXT[],
                        applicable_conditions JSONB,
                        
                        -- Usage tracking
                        times_used INTEGER DEFAULT 0,
                        times_successful INTEGER DEFAULT 0,
                        success_rate FLOAT DEFAULT 0,
                        last_used TIMESTAMPTZ,
                        
                        -- Quality indicators
                        is_validated BOOLEAN DEFAULT FALSE,
                        validation_count INTEGER DEFAULT 0,
                        peer_rating FLOAT,
                        
                        -- Status
                        is_active BOOLEAN DEFAULT TRUE,
                        deprecation_reason TEXT
                    );
                    
                    DROP TRIGGER IF EXISTS update_strategies_updated_at ON learned_strategies;
                    CREATE TRIGGER update_strategies_updated_at
                        BEFORE UPDATE ON learned_strategies
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                        
                    CREATE INDEX IF NOT EXISTS idx_strategy_category ON learned_strategies(category);
                    CREATE INDEX IF NOT EXISTS idx_strategy_performance ON learned_strategies(improvement_ratio DESC);
                    CREATE INDEX IF NOT EXISTS idx_strategy_confidence ON learned_strategies(confidence DESC);
                    CREATE INDEX IF NOT EXISTS idx_strategy_active ON learned_strategies(is_active) WHERE is_active = TRUE;
                    CREATE INDEX IF NOT EXISTS idx_strategy_materials ON learned_strategies USING GIN(applicable_materials);
                    CREATE INDEX IF NOT EXISTS idx_strategy_usage ON learned_strategies(times_used DESC, success_rate DESC);
                """)
                
                # Sensor calibration table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sensor_calibration (
                        id SERIAL PRIMARY KEY,
                        calibration_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
                        calibrated_at TIMESTAMPTZ DEFAULT NOW(),
                        sensor_type VARCHAR(50) NOT NULL,
                        sensor_id VARCHAR(100),
                        
                        -- Calibration parameters
                        offset FLOAT[],
                        scale FLOAT[],
                        rotation_matrix FLOAT[][],
                        noise_model JSONB,
                        
                        -- Validation metrics
                        validation_error FLOAT DEFAULT 0,
                        validation_samples INTEGER DEFAULT 0,
                        validation_date TIMESTAMPTZ,
                        
                        -- Status
                        is_active BOOLEAN DEFAULT TRUE,
                        confidence FLOAT DEFAULT 1.0,
                        
                        -- Metadata
                        calibration_method VARCHAR(100),
                        environment_conditions JSONB,
                        notes TEXT
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_cal_sensor_type ON sensor_calibration(sensor_type, is_active);
                    CREATE INDEX IF NOT EXISTS idx_cal_calibrated_at ON sensor_calibration(calibrated_at DESC);
                """)
                
                # System metrics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id SERIAL PRIMARY KEY,
                        recorded_at TIMESTAMPTZ DEFAULT NOW(),
                        metric_type VARCHAR(50) NOT NULL,
                        
                        -- Performance metrics
                        perception_count INTEGER DEFAULT 0,
                        dream_count INTEGER DEFAULT 0,
                        strategies_learned INTEGER DEFAULT 0,
                        database_size_mb FLOAT DEFAULT 0,
                        
                        -- Timing metrics
                        average_processing_time_ms FLOAT DEFAULT 0,
                        max_processing_time_ms FLOAT DEFAULT 0,
                        min_processing_time_ms FLOAT DEFAULT 0,
                        
                        -- Resource usage
                        memory_usage_mb FLOAT DEFAULT 0,
                        cpu_usage_percent FLOAT DEFAULT 0,
                        disk_usage_mb FLOAT DEFAULT 0,
                        
                        -- Quality metrics
                        success_rate FLOAT DEFAULT 0,
                        error_rate FLOAT DEFAULT 0,
                        average_confidence FLOAT DEFAULT 0,
                        
                        -- Additional metrics
                        custom_metrics JSONB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_metrics_recorded ON system_metrics(recorded_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_metrics_type ON system_metrics(metric_type, recorded_at DESC);
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
    
    @handle_db_errors
    def store_experience(self, experience: SensoryExperience) -> int:
        """Store a complete sensory experience.
        
        Args:
            experience: SensoryExperience object to store
            
        Returns:
            Database ID of stored experience
        """
        start_time = datetime.now()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Convert experience to storage format
                experience_data = experience.to_dict()
                
                # Calculate data size for metrics
                data_size = len(json.dumps(experience_data, default=str))
                
                cur.execute("""
                    INSERT INTO experiences (
                        experience_id, session_id, timestamp, duration_ms,
                        tactile_embedding, audio_embedding, visual_embedding, fused_embedding,
                        tactile_data, audio_data, visual_data,
                        primary_material, secondary_material, material_interaction,
                        contact_points, forces, torques,
                        action_type, action_params, success, reward,
                        tags, notes, processing_time_ms, data_size_bytes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id;
                """, (
                    experience.experience_id,
                    experience.session_id,
                    experience.timestamp,
                    experience.duration_ms,
                    experience.tactile_embedding.tolist() if experience.tactile_embedding is not None else None,
                    experience.audio_embedding.tolist() if experience.audio_embedding is not None else None,
                    experience.visual_embedding.tolist() if experience.visual_embedding is not None else None,
                    experience.fused_embedding.tolist() if experience.fused_embedding is not None else None,
                    Json(experience_data.get('tactile_data')),
                    Json(experience_data.get('audio_data')),
                    Json(experience_data.get('visual_data')),
                    experience.primary_material,
                    experience.secondary_material,
                    Json(experience.material_interaction),
                    Json([cp.to_dict() for cp in experience.contact_points] if experience.contact_points else None),
                    experience.forces,
                    experience.torques,
                    experience.action_type.name if experience.action_type else None,
                    Json(experience.action_parameters),
                    experience.success,
                    experience.reward,
                    experience.tags,
                    experience.notes,
                    (datetime.now() - start_time).total_seconds() * 1000,
                    data_size
                ))
                
                exp_id = cur.fetchone()[0]
                conn.commit()
                
                logger.debug(f"Stored experience {experience.experience_id} with ID {exp_id}")
                return exp_id
    
    @handle_db_errors
    def get_experiences(self,
                       session_id: Optional[str] = None,
                       hours: Optional[float] = None,
                       limit: int = 1000,
                       offset: int = 0,
                       material: Optional[str] = None,
                       success_only: bool = False,
                       action_type: Optional[str] = None,
                       min_reward: Optional[float] = None,
                       tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve experiences with flexible filtering.
        
        Args:
            session_id: Filter by session ID
            hours: Get experiences from last N hours
            limit: Maximum number of experiences to return
            offset: Offset for pagination
            material: Filter by primary material
            success_only: Only return successful experiences
            action_type: Filter by action type
            min_reward: Minimum reward threshold
            tags: Filter by tags (must contain all specified tags)
            
        Returns:
            List of experience dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build dynamic query
                conditions = []
                params = []
                
                if session_id:
                    conditions.append("session_id = %s")
                    params.append(session_id)
                    
                if hours:
                    conditions.append("created_at > NOW() - INTERVAL '%s hours'")
                    params.append(hours)
                    
                if material:
                    conditions.append("(primary_material = %s OR secondary_material = %s)")
                    params.extend([material, material])
                    
                if success_only:
                    conditions.append("success = TRUE")
                    
                if action_type:
                    conditions.append("action_type = %s")
                    params.append(action_type)
                    
                if min_reward is not None:
                    conditions.append("reward >= %s")
                    params.append(min_reward)
                    
                if tags:
                    conditions.append("tags @> %s")
                    params.append(tags)
                
                # Construct final query
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                query = f"""
                    SELECT * FROM experiences
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                
                params.extend([limit, offset])
                
                cur.execute(query, params)
                results = cur.fetchall()
                
                logger.debug(f"Retrieved {len(results)} experiences")
                return [dict(row) for row in results]
    
    @handle_db_errors
    def get_recent_experiences(self, 
                              hours: float = 24,
                              limit: int = 1000,
                              **filters) -> List[Dict[str, Any]]:
        """Get recent experiences (convenience method)."""
        return self.get_experiences(hours=hours, limit=limit, **filters)
    
    @handle_db_errors
    def store_dream_session(self, session_data: Dict[str, Any]) -> str:
        """Store dream session results.
        
        Args:
            session_data: Dictionary containing dream session data
            
        Returns:
            Session ID of stored dream session
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dream_sessions (
                        config, experience_count, time_range_hours, source_session_ids,
                        replay_count, variations_generated, compute_time_seconds, memory_usage_mb,
                        strategies_found, strategies_stored, improvements, consolidated_memories,
                        average_improvement, best_improvement, convergence_iterations,
                        status, notes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING session_id;
                """, (
                    Json(session_data.get('config', {})),
                    session_data.get('experience_count', 0),
                    session_data.get('time_range_hours', 24),
                    session_data.get('source_session_ids', []),
                    session_data.get('replay_count', 0),
                    session_data.get('variations_generated', 0),
                    session_data.get('compute_time_seconds', 0),
                    session_data.get('memory_usage_mb', 0),
                    session_data.get('strategies_found', 0),
                    session_data.get('strategies_stored', 0),
                    Json(session_data.get('improvements', {})),
                    session_data.get('consolidated_memories', 0),
                    session_data.get('average_improvement', 0),
                    session_data.get('best_improvement', 0),
                    session_data.get('convergence_iterations'),
                    session_data.get('status', 'completed'),
                    session_data.get('notes')
                ))
                
                session_id = cur.fetchone()[0]
                
                # Update end time
                cur.execute("""
                    UPDATE dream_sessions 
                    SET ended_at = NOW()
                    WHERE session_id = %s
                """, (session_id,))
                
                conn.commit()
                
                logger.info(f"Stored dream session {session_id}")
                return str(session_id)
    
    @handle_db_errors
    def store_learned_strategy(self, strategy: LearnedStrategy) -> int:
        """Store a learned strategy.
        
        Args:
            strategy: LearnedStrategy object to store
            
        Returns:
            Database ID of stored strategy
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO learned_strategies (
                        strategy_id, name, category, description,
                        strategy_data, baseline_performance, improved_performance,
                        improvement_ratio, confidence, applicable_materials,
                        applicable_scenarios, times_used, success_rate
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id;
                """, (
                    strategy.strategy_id,
                    strategy.name,
                    strategy.category,
                    "",  # description - can be added later
                    Json(strategy.strategy_data),
                    strategy.baseline_performance,
                    strategy.improved_performance,
                    strategy.improvement_ratio,
                    strategy.confidence,
                    strategy.applicable_materials,
                    strategy.applicable_scenarios,
                    strategy.times_used,
                    strategy.success_rate
                ))
                
                strategy_id = cur.fetchone()[0]
                conn.commit()
                
                logger.debug(f"Stored strategy '{strategy.name}' with ID {strategy_id}")
                return strategy_id
    
    @handle_db_errors
    def get_learned_strategies(self,
                              category: Optional[str] = None,
                              min_confidence: float = 0.0,
                              min_improvement: float = 0.0,
                              active_only: bool = True,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get learned strategies with filtering.
        
        Args:
            category: Filter by strategy category
            min_confidence: Minimum confidence threshold
            min_improvement: Minimum improvement ratio
            active_only: Only return active strategies
            limit: Maximum number of strategies to return
            
        Returns:
            List of strategy dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                conditions = ["confidence >= %s", "improvement_ratio >= %s"]
                params = [min_confidence, min_improvement]
                
                if category:
                    conditions.append("category = %s")
                    params.append(category)
                    
                if active_only:
                    conditions.append("is_active = TRUE")
                
                where_clause = "WHERE " + " AND ".join(conditions)
                query = f"""
                    SELECT * FROM learned_strategies
                    {where_clause}
                    ORDER BY improvement_ratio DESC, confidence DESC
                    LIMIT %s
                """
                
                params.append(limit)
                cur.execute(query, params)
                results = cur.fetchall()
                
                logger.debug(f"Retrieved {len(results)} strategies")
                return [dict(row) for row in results]
    
    @handle_db_errors
    def update_strategy_usage(self, strategy_id: str, success: bool):
        """Update strategy usage statistics.
        
        Args:
            strategy_id: Strategy ID to update
            success: Whether the strategy was successful
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE learned_strategies
                    SET times_used = times_used + 1,
                        times_successful = times_successful + %s,
                        success_rate = (times_successful::FLOAT + %s) / (times_used + 1),
                        last_used = NOW()
                    WHERE strategy_id = %s
                """, (1 if success else 0, 1 if success else 0, strategy_id))
                
                conn.commit()
                logger.debug(f"Updated usage for strategy {strategy_id}")
    
    @handle_db_errors
    def cleanup_old_data(self, 
                        retention_days: int = 30,
                        keep_important: bool = True) -> Dict[str, int]:
        """Clean up old data to manage storage.
        
        Args:
            retention_days: Days to retain data
            keep_important: Keep important experiences (successful, high reward)
            
        Returns:
            Dictionary with cleanup statistics
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cleanup_stats = {}
                
                # Clean old experiences
                conditions = "created_at < NOW() - INTERVAL '%s days'"
                params = [retention_days]
                
                if keep_important:
                    conditions += " AND success = FALSE AND reward < 0.5"
                
                # Don't delete experiences referenced by predictions
                conditions += """
                    AND id NOT IN (
                        SELECT DISTINCT source_experience_id 
                        FROM material_predictions 
                        WHERE source_experience_id IS NOT NULL
                    )
                """
                
                cur.execute(f"DELETE FROM experiences WHERE {conditions}", params)
                cleanup_stats['experiences_deleted'] = cur.rowcount
                
                # Clean old dream sessions (keep metadata)
                cur.execute("""
                    UPDATE dream_sessions 
                    SET improvements = NULL, notes = NULL
                    WHERE started_at < NOW() - INTERVAL '%s days'
                """, (retention_days * 2,))  # Keep dream metadata longer
                cleanup_stats['dream_sessions_cleaned'] = cur.rowcount
                
                # Clean old sensor calibration data
                cur.execute("""
                    DELETE FROM sensor_calibration 
                    WHERE calibrated_at < NOW() - INTERVAL '%s days' 
                    AND is_active = FALSE
                """, (retention_days,))
                cleanup_stats['calibrations_deleted'] = cur.rowcount
                
                # Clean old system metrics
                cur.execute("""
                    DELETE FROM system_metrics 
                    WHERE recorded_at < NOW() - INTERVAL '%s days'
                """, (retention_days // 2,))  # Keep metrics for shorter time
                cleanup_stats['metrics_deleted'] = cur.rowcount
                
                # Vacuum analyze for performance
                cur.execute("VACUUM ANALYZE;")
                
                conn.commit()
                
                total_deleted = sum(cleanup_stats.values())
                logger.info(f"Cleanup completed: {total_deleted} records removed")
                
                return cleanup_stats
    
    @handle_db_errors
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics.
        
        Returns:
            Dictionary with database statistics
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                stats = {}
                
                # Table sizes
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                        pg_relation_size(schemaname||'.'||tablename) as table_size,
                        pg_total_relation_size(schemaname||'.'||tablename) - 
                        pg_relation_size(schemaname||'.'||tablename) as index_size
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename IN ('experiences', 'learned_strategies', 'dream_sessions')
                    ORDER BY size_bytes DESC;
                """)
                
                table_stats = {}
                total_size = 0
                for row in cur.fetchall():
                    table_name = row['tablename']
                    table_stats[table_name] = {
                        'size_mb': row['size_bytes'] / 1024 / 1024,
                        'table_size_mb': row['table_size'] / 1024 / 1024,
                        'index_size_mb': row['index_size'] / 1024 / 1024
                    }
                    total_size += row['size_bytes']
                
                stats['tables'] = table_stats
                stats['total_size_mb'] = total_size / 1024 / 1024
                
                # Record counts
                tables = ['experiences', 'learned_strategies', 'dream_sessions', 'material_predictions']
                record_counts = {}
                
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    record_counts[table] = cur.fetchone()[0]
                
                stats['record_counts'] = record_counts
                
                # Recent activity (last 24 hours)
                cur.execute("""
                    SELECT 
                        COUNT(*) as recent_experiences,
                        COUNT(CASE WHEN success = TRUE THEN 1 END) as successful_experiences,
                        AVG(reward) as avg_reward,
                        COUNT(DISTINCT session_id) as active_sessions
                    FROM experiences 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                activity = cur.fetchone()
                stats['recent_activity'] = dict(activity)
                
                # Connection pool stats
                stats['connection_pool'] = {
                    'total_connections': self.max_connections,
                    'min_connections': self.min_connections
                }
                
                return stats
    
    @handle_db_errors
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system performance metrics.
        
        Args:
            metrics: SystemMetrics object to store
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                metrics_dict = metrics.to_dict()
                
                cur.execute("""
                    INSERT INTO system_metrics (
                        metric_type, perception_count, dream_count, strategies_learned,
                        average_processing_time_ms, memory_usage_mb, success_rate,
                        custom_metrics
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    'system_snapshot',
                    metrics.perception_count,
                    metrics.dream_count,
                    metrics.strategies_learned,
                    metrics.average_processing_time,
                    metrics.memory_usage_mb,
                    metrics.success_rate,
                    Json(metrics_dict)
                ))
                
                conn.commit()
                logger.debug("Stored system metrics")
    
    def close(self):
        """Close all database connections."""
        if hasattr(self, 'pool') and self.pool:
            try:
                self.pool.closeall()
                logger.info("Database connections closed")
            except Exception as e:
                logger.error(f"Error closing database connections: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup