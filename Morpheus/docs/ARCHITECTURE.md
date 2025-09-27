# MORPHEUS System Architecture

## Overview

MORPHEUS (Multi-modal Optimization through Replay, Prediction, and Haptic-Environmental Understanding System) implements a sophisticated robotics architecture combining real-time sensory processing with dream-based learning optimization.

## Core Design Principles

### 1. Modular Architecture
- **Loose Coupling**: Components communicate through well-defined interfaces
- **High Cohesion**: Related functionality grouped into focused modules
- **Dependency Injection**: Configurable component dependencies
- **Plugin Architecture**: Extensible processor and network plugins

### 2. Multi-Modal Processing
- **Sensory Fusion**: Neural attention-based fusion of tactile, audio, visual
- **Material Awareness**: Physics-based material property integration
- **Real-time Processing**: <10ms latency for critical perception paths
- **Parallel Processing**: Multi-threaded sensory stream handling

### 3. Dream-Based Learning
- **Experience Replay**: Systematic replay of stored experiences
- **Strategy Optimization**: Neural strategy discovery and refinement
- **Knowledge Consolidation**: Integration of learned patterns
- **Continuous Improvement**: Autonomous performance enhancement

## System Components

### Core Layer (`morpheus.core`)

#### Orchestrator
Central system coordinator managing:
- Session lifecycle with UUID tracking
- Component initialization and cleanup
- Multi-modal perception orchestration
- Dream cycle scheduling and execution
- Performance monitoring and metrics

#### Configuration System
Hierarchical configuration management:
- YAML-based configuration with validation
- Environment variable support
- Runtime configuration updates
- Component-specific configuration sections

#### Type System
Comprehensive data structures:
- Sensory signature dataclasses
- Experience and observation types
- Configuration schemas with Pydantic
- Vector embedding representations

### Perception Layer (`morpheus.perception`)

#### Tactile Processor
Advanced haptic processing:
- Multi-contact point analysis
- Vibration spectrum analysis (32-bin FFT)
- Material-aware texture classification
- Pressure and deformation computation
- 64-dimensional tactile embeddings

```python
class TactileProcessor:
    def process_contacts(self, body_id: int, material: str) -> TactileSignature:
        # Contact point extraction from PyBullet
        # Force history analysis
        # Vibration spectrum computation
        # Material-based property prediction
```

#### Audio Spatial Processor
3D spatial audio processing:
- HRTF-based spatial localization
- Multi-source audio fusion (up to 10 sources)
- Doppler effect computation
- Material-based acoustic modeling
- 32-dimensional audio embeddings

```python
class AudioProcessor:
    def process_spatial_audio(self, position: np.ndarray) -> AudioSignature:
        # 3D spatial audio localization
        # Multi-source fusion
        # Doppler shift calculation
        # Acoustic material inference
```

#### Sensory Fusion Network
Neural multi-modal fusion:
- Transformer-based attention fusion
- Quality-adaptive modality weighting  
- Temporal LSTM processing
- Uncertainty quantification
- 128-dimensional fused embeddings

```python
class SensoryFusionNetwork(nn.Module):
    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        # Multi-head attention fusion
        # Quality-based weighting
        # Temporal processing
        # Uncertainty estimation
```

### Dream Simulation Layer (`morpheus.dream_sim`)

#### Dream Orchestrator
Experience replay and optimization:
- Parallel dream session processing (2-8 threads)
- Experience variation generation
- Strategy discovery and evaluation
- Knowledge consolidation algorithms
- Performance improvement tracking

```python
class DreamOrchestrator:
    def dream(self, duration: float) -> Dict[str, Any]:
        # Parallel experience processing
        # Strategy optimization
        # Performance evaluation
        # Knowledge consolidation
```

#### Strategy Optimizer
Neural strategy learning:
- Multi-objective optimization
- Strategy similarity analysis
- Performance-based ranking
- Automated strategy merging
- Continuous strategy refinement

### Predictive Layer (`morpheus.predictive`)

#### Forward Model
State prediction system:
- Multi-step prediction (1-50 steps)
- Physics-informed neural networks
- Conservation law enforcement
- Material property conditioning
- Uncertainty quantification

```python
class SensoryPredictor:
    def predict_trajectory(self, state: np.ndarray, actions: List) -> Prediction:
        # Multi-step forward prediction
        # Physics constraint enforcement
        # Uncertainty estimation
        # Material property integration
```

#### Material Predictor
Material interaction modeling:
- Physics-based property computation
- Interaction outcome prediction
- Confidence estimation
- Learning from experience feedback

### Storage Layer (`morpheus.storage`)

#### PostgreSQL Interface
Production database management:
- Connection pooling (1-20 connections)
- Optimized schema with JSONB flexibility
- Multi-modal data storage
- Experience indexing and retrieval
- Automated cleanup and maintenance

```sql
-- Core experience table
CREATE TABLE experiences (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    fused_embedding FLOAT[],
    tactile_data JSONB,
    audio_data JSONB,
    primary_material VARCHAR(100),
    success BOOLEAN,
    reward FLOAT
);

-- Strategic indices for performance
CREATE INDEX idx_exp_timestamp ON experiences(created_at DESC);
CREATE INDEX idx_exp_material ON experiences(primary_material);
```

### Integration Layer (`morpheus.integration`)

#### Material Bridge
GASM-Robotics material integration:
- Direct material property loading
- Physics parameter computation
- Interaction prediction modeling
- Material recognition from sensors

```python
class MaterialBridge:
    def compute_interaction(self, mat1: str, mat2: str) -> Dict:
        # Combined friction/restitution
        # Contact stiffness computation
        # Audio frequency prediction
        # Thermal interaction modeling
```

#### GASM Bridge
Real-time GASM-Robotics synchronization:
- 100Hz+ state synchronization
- Bidirectional command/state exchange
- Coordinate frame transformation
- Visual integration and capture

## Data Flow Architecture

### Perception Pipeline
```
Sensor Data → Processor → Embedding → Fusion → Storage
     ↓             ↓          ↓         ↓        ↓
  Contact      Tactile    64-dim    128-dim   Database
   Points      Analysis   Vector    Vector    Record
```

### Dream Learning Pipeline
```
Experiences → Replay → Variations → Evaluation → Strategies
     ↓           ↓         ↓           ↓          ↓
  Database   Parallel   Generated   Neural     Consolidated
  Records    Threads    Scenarios   Scoring    Knowledge
```

### Prediction Pipeline
```
Current State → Forward Model → Predicted State → Uncertainty
     ↓              ↓               ↓              ↓
  128-dim        Neural           128-dim        Confidence
  Vector         Network          Vector         Score
```

## Neural Network Architectures

### Sensory Fusion Network
- **Input**: Multi-modal embeddings (64+32+128 dims)
- **Architecture**: Transformer with 8 attention heads
- **Hidden**: 256-dimensional hidden representations
- **Output**: 128-dimensional fused embedding
- **Training**: Contrastive learning with uncertainty

### Forward Prediction Model
- **Input**: State (128-dim) + Action (7-dim)
- **Architecture**: Physics-informed MLP with skip connections
- **Hidden**: 256-dimensional with ReLU activations
- **Output**: Next state (128-dim) + uncertainty
- **Constraints**: Physics conservation laws

### Strategy Discovery Network
- **Input**: Experience sequences (variable length)
- **Architecture**: LSTM encoder with attention decoder
- **Hidden**: 512-dimensional recurrent states
- **Output**: Strategy representations and scores
- **Learning**: Reinforcement learning with experience replay

## Performance Architecture

### Real-time Requirements
- **Perception Latency**: <10ms for sensory fusion
- **Database Throughput**: >5000 writes/second
- **Memory Usage**: <4GB total system memory
- **CPU Usage**: <80% on 4-core systems

### Scaling Characteristics
- **Horizontal**: Multi-process dream sessions
- **Vertical**: GPU acceleration for neural networks
- **Storage**: Automatic data cleanup and archival
- **Network**: Distributed GASM synchronization

### Quality Assurance
- **Test Coverage**: >95% code coverage
- **Performance Tests**: Latency and throughput validation
- **Integration Tests**: End-to-end system validation
- **Stress Tests**: Extended operation validation

## Security Architecture

### Data Protection
- Database connection encryption
- Sensitive parameter environment variables
- Input validation and sanitization
- Error handling without information leakage

### System Isolation
- Containerized deployment with Docker
- Network isolation for database access
- File system permission restrictions
- Process isolation for parallel operations

## Deployment Architecture

### Container Strategy
```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: morpheus
  
  morpheus:
    build: .
    depends_on:
      - postgres
    environment:
      - DATABASE_HOST=postgres
```

### Production Configuration
- Multi-environment configuration support
- Secrets management integration
- Health check endpoints
- Monitoring and alerting integration

This architecture provides a robust, scalable, and maintainable foundation for advanced robotics perception and learning applications.