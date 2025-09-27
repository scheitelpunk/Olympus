# MORPHEUS Database Schema Design

## Overview

The MORPHEUS system uses PostgreSQL as its primary data store, designed for high-performance sensory data storage, experience tracking, and learned strategy management. The schema supports flexible data structures while maintaining query performance through strategic indexing.

## Core Design Principles

1. **Flexible Schema Evolution**: JSONB for extensible data structures
2. **High-Performance Queries**: Optimized indices for temporal and material-based searches
3. **Scalable Storage**: Efficient handling of large embedding vectors
4. **Data Integrity**: Foreign key relationships with cascade handling
5. **Analytics Ready**: Structure optimized for learning analytics and reporting

## Database Configuration

### Connection Settings
```yaml
database:
  host: "localhost"
  port: 5432
  database: "morpheus"
  user: "morpheus_user"
  password: "morpheus_pass"
  
connection_pool:
  min_connections: 1
  max_connections: 20
  connection_timeout: 30s
```

### Performance Optimizations
```sql
-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Performance settings
shared_preload_libraries = 'pg_stat_statements'
work_mem = '256MB'
maintenance_work_mem = '512MB' 
effective_cache_size = '2GB'
random_page_cost = 1.1
```

## Core Tables

### 1. experiences

**Purpose**: Central repository for all sensory experiences

```sql
CREATE TABLE experiences (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    session_id UUID NOT NULL,
    
    -- Temporal data
    timestamp FLOAT NOT NULL,
    duration_ms FLOAT,
    
    -- Sensory embeddings (fixed-size arrays for performance)
    tactile_embedding FLOAT[64],
    audio_embedding FLOAT[32], 
    visual_embedding FLOAT[128],
    fused_embedding FLOAT[128],
    
    -- Raw sensory data (flexible JSONB structure)
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
    success BOOLEAN,
    reward FLOAT,
    
    -- Metadata
    tags TEXT[],
    notes TEXT
);
```

**Key Indices**:
```sql
-- Primary access patterns
CREATE INDEX idx_exp_timestamp ON experiences(created_at DESC);
CREATE INDEX idx_exp_session ON experiences(session_id);
CREATE INDEX idx_exp_material ON experiences(primary_material, secondary_material);
CREATE INDEX idx_exp_success ON experiences(success);
CREATE INDEX idx_exp_tags ON experiences USING GIN(tags);

-- Composite indices for common queries
CREATE INDEX idx_exp_material_success ON experiences(primary_material, success, created_at DESC);
CREATE INDEX idx_exp_session_time ON experiences(session_id, created_at DESC);
```

**Sample Data Structure**:
```json
{
  "tactile_data": {
    "contact_points": [
      {"position": [0.1, 0.2, 0.3], "force": 5.2, "normal": [0, 0, 1]}
    ],
    "total_force": 5.2,
    "contact_area": 0.001,
    "pressure": 5200.0,
    "texture": "smooth",
    "hardness": 0.8,
    "temperature": 22.5,
    "vibration": [0.1, 0.05, 0.02, ...]
  },
  "audio_data": {
    "sources": [
      {"frequency": 440.0, "amplitude": 0.8, "position": [1.0, 0.5, 0.2]}
    ],
    "doppler_shifts": [0.02],
    "reverb_time": 0.3,
    "spatial_signature": [0.2, 0.8, 0.1, ...]
  }
}
```

### 2. material_predictions

**Purpose**: Track material property predictions vs actual measurements

```sql
CREATE TABLE material_predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Materials being predicted
    material1 VARCHAR(100) NOT NULL,
    material2 VARCHAR(100) NOT NULL,
    
    -- Predicted properties
    predicted_friction FLOAT,
    predicted_restitution FLOAT,
    predicted_hardness FLOAT,
    predicted_sound_freq FLOAT,
    predicted_texture VARCHAR(50),
    
    -- Actual measurements (when available)
    actual_friction FLOAT,
    actual_restitution FLOAT,
    actual_hardness FLOAT,
    actual_sound_freq FLOAT,
    actual_texture VARCHAR(50),
    
    -- Prediction quality metrics
    confidence FLOAT,
    error_metrics JSONB,
    
    -- Source experience reference
    source_experience_id INTEGER REFERENCES experiences(id)
);
```

**Indices**:
```sql
CREATE INDEX idx_mat_pred_materials ON material_predictions(material1, material2);
CREATE INDEX idx_mat_pred_confidence ON material_predictions(confidence DESC);
CREATE INDEX idx_mat_pred_created ON material_predictions(created_at DESC);
```

### 3. dream_sessions

**Purpose**: Record dream session metadata and outcomes

```sql
CREATE TABLE dream_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Configuration used
    config JSONB NOT NULL,
    
    -- Input data scope  
    experience_count INTEGER,
    time_range_hours FLOAT,
    
    -- Processing metrics
    replay_count INTEGER,
    variations_generated INTEGER,
    compute_time_seconds FLOAT,
    
    -- Results achieved
    strategies_found INTEGER,
    improvements JSONB,
    consolidated_memories INTEGER,
    
    -- Quality metrics
    average_improvement FLOAT,
    best_improvement FLOAT,
    
    notes TEXT
);
```

**Sample Configuration**:
```json
{
  "dream_config": {
    "replay_speed": 10.0,
    "variation_factor": 0.2,
    "exploration_rate": 0.3,
    "consolidation_threshold": 0.8,
    "min_improvement": 0.1,
    "max_iterations": 1000,
    "parallel_dreams": 4
  }
}
```

### 4. learned_strategies

**Purpose**: Store discovered strategies with performance tracking

```sql
CREATE TABLE learned_strategies (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    dream_session_id UUID REFERENCES dream_sessions(id),
    
    -- Strategy identification
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50), -- 'tactile', 'audio', 'motion', 'material'
    
    -- The actual strategy data
    strategy_data JSONB NOT NULL,
    
    -- Performance metrics
    baseline_performance FLOAT,
    improved_performance FLOAT,
    improvement_ratio FLOAT,
    confidence FLOAT,
    
    -- Applicability rules
    applicable_materials TEXT[],
    applicable_scenarios TEXT[],
    
    -- Usage tracking
    times_used INTEGER DEFAULT 0,
    last_used TIMESTAMPTZ,
    success_rate FLOAT
);
```

**Indices**:
```sql
CREATE INDEX idx_strategy_category ON learned_strategies(category);
CREATE INDEX idx_strategy_performance ON learned_strategies(improvement_ratio DESC);
CREATE INDEX idx_strategy_confidence ON learned_strategies(confidence DESC);
CREATE INDEX idx_strategy_materials ON learned_strategies USING GIN(applicable_materials);
```

**Strategy Data Structure**:
```json
{
  "strategy_type": "force_optimization",
  "original_approach": {
    "force_level": 10.0,
    "contact_duration": 2.0,
    "success_rate": 0.6
  },
  "optimized_approach": {
    "force_level": 7.5,
    "contact_duration": 1.5,
    "success_rate": 0.85
  },
  "key_insights": [
    "Lower force reduces material stress",
    "Shorter duration maintains effectiveness"
  ],
  "conditions": {
    "material_hardness_range": [0.3, 0.7],
    "contact_area_min": 0.001
  }
}
```

### 5. sensor_calibration

**Purpose**: Track sensor calibration data and validation

```sql
CREATE TABLE sensor_calibration (
    id SERIAL PRIMARY KEY,
    calibrated_at TIMESTAMPTZ DEFAULT NOW(),
    sensor_type VARCHAR(50) NOT NULL, -- 'tactile', 'audio', 'visual'
    
    -- Calibration parameters
    offset FLOAT[],
    scale FLOAT[],
    noise_model JSONB,
    
    -- Validation metrics
    validation_error FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Calibration metadata
    calibration_method VARCHAR(100),
    validation_samples INTEGER,
    notes TEXT
);
```

## Supporting Tables

### 6. sessions

**Purpose**: Track system sessions and user interactions

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Session metadata
    session_type VARCHAR(50), -- 'perception', 'dream', 'mixed'
    user_id VARCHAR(100),
    
    -- Performance tracking
    total_experiences INTEGER DEFAULT 0,
    total_dreams INTEGER DEFAULT 0,
    total_strategies INTEGER DEFAULT 0,
    
    -- Session configuration
    config JSONB,
    
    notes TEXT
);
```

### 7. system_metrics

**Purpose**: Track system performance and health metrics

```sql
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Performance metrics
    avg_perception_latency_ms FLOAT,
    avg_dream_duration_seconds FLOAT,
    experiences_per_second FLOAT,
    
    -- Resource usage
    cpu_usage_percent FLOAT,
    memory_usage_mb FLOAT,
    disk_usage_mb FLOAT,
    
    -- Database metrics
    active_connections INTEGER,
    query_response_time_ms FLOAT,
    
    -- Learning metrics
    strategies_learned_per_hour FLOAT,
    average_improvement_ratio FLOAT,
    
    notes TEXT
);
```

## Query Optimization Patterns

### 1. Recent Experience Retrieval
```sql
-- Optimized for dream session preparation
SELECT * FROM experiences 
WHERE created_at > NOW() - INTERVAL '24 hours'
  AND success = TRUE
  AND primary_material IN ('steel', 'rubber', 'plastic')
ORDER BY created_at DESC 
LIMIT 1000;
```

### 2. Material Performance Analysis
```sql
-- Material success rate analysis
SELECT 
    primary_material,
    COUNT(*) as total_attempts,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
    AVG(reward) as average_reward,
    AVG(ARRAY_LENGTH(forces, 1)) as avg_contact_points
FROM experiences 
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY primary_material
ORDER BY average_reward DESC;
```

### 3. Strategy Effectiveness Tracking
```sql
-- Strategy performance over time
SELECT 
    s.name,
    s.category,
    s.improvement_ratio,
    s.times_used,
    s.success_rate,
    d.started_at as discovery_date
FROM learned_strategies s
JOIN dream_sessions d ON s.dream_session_id = d.id
WHERE s.confidence > 0.7
  AND s.times_used > 0
ORDER BY s.improvement_ratio DESC;
```

## Data Retention and Archival

### Retention Policies

```sql
-- Archive old experiences (keep learning-relevant ones)
CREATE OR REPLACE FUNCTION archive_old_experiences() 
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM experiences 
    WHERE created_at < NOW() - INTERVAL '30 days'
    AND id NOT IN (
        SELECT DISTINCT source_experience_id 
        FROM material_predictions 
        WHERE source_experience_id IS NOT NULL
    );
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
pg_dump -h localhost -U morpheus_user morpheus | gzip > "morpheus_backup_$(date +%Y%m%d).sql.gz"

# Keep 30 days of backups
find /backup/path -name "morpheus_backup_*.sql.gz" -mtime +30 -delete
```

## Performance Monitoring

### Key Metrics to Track

1. **Query Performance**: Average response times for common queries
2. **Storage Growth**: Table sizes and growth rates
3. **Index Effectiveness**: Index hit ratios and scan types
4. **Connection Usage**: Pool utilization and wait times

### Monitoring Queries

```sql
-- Table size monitoring
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY size_bytes DESC;

-- Index usage statistics
SELECT 
    indexrelname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;
```

This database schema provides a robust foundation for the MORPHEUS system, supporting both high-performance operation and analytical insights while maintaining flexibility for future enhancements.