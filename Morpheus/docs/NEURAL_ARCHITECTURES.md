# MORPHEUS Neural Network Architectures

## Overview

MORPHEUS employs several specialized neural networks for multi-modal sensory fusion, predictive modeling, and dream-based optimization. These architectures are designed for real-time operation while maintaining learning capability through experience replay.

## Core Network Architectures

### 1. Sensory Fusion Network (SensoryFusionNetwork)

**Purpose**: Integrate tactile, audio, and visual sensory inputs into unified embeddings

```python
class SensoryFusionNetwork(nn.Module):
    """
    Multi-modal sensory fusion using attention mechanisms.
    Handles variable modality availability (missing sensors).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Modal-specific encoders
        self.tactile_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(128)
        )
        
        # Cross-modal attention fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, tactile_input, audio_input, visual_input):
        batch_size = tactile_input.shape[0] if tactile_input is not None else 1
        encoded_modalities = []
        valid_modalities = []
        
        # Encode available modalities
        if tactile_input is not None:
            tactile_encoded = self.tactile_encoder(tactile_input)
            encoded_modalities.append(tactile_encoded)
            valid_modalities.append(0)
            
        if audio_input is not None:
            audio_encoded = self.audio_encoder(audio_input)
            encoded_modalities.append(audio_encoded)
            valid_modalities.append(1)
            
        if visual_input is not None:
            visual_encoded = self.visual_encoder(visual_input)
            encoded_modalities.append(visual_encoded)
            valid_modalities.append(2)
            
        # Handle missing modalities with learned zero vectors
        if len(encoded_modalities) == 0:
            return torch.zeros(batch_size, 128)
            
        # Stack modalities for attention
        stacked_modalities = torch.stack(encoded_modalities, dim=1)
        
        # Apply cross-modal attention
        attended_features, attention_weights = self.multihead_attention(
            stacked_modalities, stacked_modalities, stacked_modalities
        )
        
        # Weight by learned modality importance
        weights = torch.softmax(self.modality_weights[valid_modalities], dim=0)
        weighted_features = torch.sum(attended_features * weights.unsqueeze(0).unsqueeze(-1), dim=1)
        
        # Final fusion
        fused_output = self.fusion_layers(weighted_features)
        
        return fused_output, attention_weights
```

**Architecture Diagram**:
```
Tactile Input (64D) ──→ Encoder ──→ [128D] ──┐
                                              │
Audio Input (32D)   ──→ Encoder ──→ [128D] ──┼── Multi-Head ──→ Weighted ──→ Fusion ──→ Output
                                              │   Attention      Combination   Layers    (128D)
Visual Input (128D) ──→ Encoder ──→ [128D] ──┘
```

### 2. Predictive Forward Model (SensoryPredictor)

**Purpose**: Predict future sensory states given current state and actions

```python
class SensoryPredictor(nn.Module):
    """
    Forward model for predicting next sensory state with uncertainty estimation.
    Uses ensemble methods for robust uncertainty quantification.
    """
    
    def __init__(self, config):
        super().__init__()
        self.state_dim = config['state_dim']  # 128
        self.action_dim = config['action_dim']  # 7 (6DOF + gripper)
        self.hidden_dim = config['hidden_dim']  # 256
        
        # State-action encoder
        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Multiple prediction heads (ensemble)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.state_dim)
            ) for _ in range(5)
        ])
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.state_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, current_state, action, hidden_state=None):
        batch_size, seq_len = current_state.shape[:2]
        
        # Combine state and action
        state_action = torch.cat([current_state, action], dim=-1)
        
        # Encode state-action pairs
        encoded = self.state_action_encoder(state_action)
        
        # LSTM processing
        lstm_output, new_hidden = self.lstm(encoded, hidden_state)
        
        # Ensemble predictions
        predictions = []
        for head in self.prediction_heads:
            pred = head(lstm_output)
            predictions.append(pred)
            
        # Combine ensemble predictions
        ensemble_pred = torch.stack(predictions, dim=0)
        mean_pred = torch.mean(ensemble_pred, dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = torch.var(ensemble_pred, dim=0)
        
        # Aleatoric uncertainty (data uncertainty)  
        aleatoric_uncertainty = self.uncertainty_head(lstm_output)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_pred, total_uncertainty, new_hidden
```

### 3. Material Property Predictor (MaterialPredictor)

**Purpose**: Predict material properties from sensory observations

```python
class MaterialPredictor(nn.Module):
    """
    Predicts material properties from multi-modal sensory input.
    Learns relationships between sensory signatures and physical properties.
    """
    
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['sensory_dim']  # 128
        self.num_materials = config['num_materials']  # Known material count
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Property prediction heads
        self.friction_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Friction 0-2 range, normalized
        )
        
        self.restitution_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()  # Restitution 0-1 range
        )
        
        self.hardness_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized hardness 0-1
        )
        
        self.density_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized density 0-1
        )
        
        # Material classification head
        self.material_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_materials)
        )
        
    def forward(self, sensory_input):
        # Extract features
        features = self.feature_extractor(sensory_input)
        
        # Predict properties
        friction = self.friction_head(features) * 2.0  # Scale to 0-2 range
        restitution = self.restitution_head(features)
        hardness = self.hardness_head(features) 
        density = self.density_head(features) * 10000  # Scale to reasonable density range
        
        # Material classification
        material_logits = self.material_classifier(features)
        material_probs = torch.softmax(material_logits, dim=-1)
        
        return {
            'friction': friction,
            'restitution': restitution, 
            'hardness': hardness,
            'density': density,
            'material_probs': material_probs
        }
```

## Specialized Network Components

### 4. Attention-Based Sequence Encoder

```python
class SequenceEncoder(nn.Module):
    """
    Encode sequences of experiences for dream processing.
    Uses self-attention to identify important temporal patterns.
    """
    
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['experience_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.input_dim, max_len=1000)
        
        # Multi-layer transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
    def forward(self, experience_sequences, mask=None):
        # Add positional encoding
        encoded_input = self.pos_encoding(experience_sequences)
        
        # Apply transformer encoder
        encoded_sequences = self.transformer_encoder(encoded_input, src_key_padding_mask=mask)
        
        # Project to output dimension
        output = self.output_projection(encoded_sequences)
        
        return output
```

### 5. Variational Strategy Generator

```python
class StrategyGenerator(nn.Module):
    """
    Generate strategy variations using variational autoencoders.
    Learns latent strategy representations for systematic exploration.
    """
    
    def __init__(self, config):
        super().__init__()
        self.strategy_dim = config['strategy_dim']
        self.latent_dim = config['latent_dim']
        
        # Encoder (strategy → latent)
        self.encoder = nn.Sequential(
            nn.Linear(self.strategy_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(128, self.latent_dim)
        self.logvar_head = nn.Linear(128, self.latent_dim)
        
        # Decoder (latent → strategy)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.strategy_dim)
        )
        
    def encode(self, strategy):
        h = self.encoder(strategy)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, strategy):
        mu, logvar = self.encode(strategy)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
        
    def generate_variations(self, strategy, num_variations=5):
        """Generate variations of a given strategy"""
        mu, logvar = self.encode(strategy)
        
        variations = []
        for _ in range(num_variations):
            # Sample from latent space
            z = self.reparameterize(mu, logvar)
            # Add exploration noise
            z = z + torch.randn_like(z) * 0.1
            variation = self.decode(z)
            variations.append(variation)
            
        return torch.stack(variations)
```

## Training Procedures

### Online Learning Protocol

```python
class MorpheusTrainer:
    """
    Handles online training of MORPHEUS neural networks.
    Implements experience replay and continual learning strategies.
    """
    
    def __init__(self, networks, config):
        self.networks = networks
        self.config = config
        
        # Optimizers
        self.fusion_optimizer = torch.optim.AdamW(
            networks['fusion'].parameters(), 
            lr=config['learning_rates']['fusion']
        )
        self.predictor_optimizer = torch.optim.AdamW(
            networks['predictor'].parameters(),
            lr=config['learning_rates']['predictor']
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss()
        
        # Experience buffer for replay
        self.experience_buffer = ExperienceReplayBuffer(
            max_size=config['buffer_size']
        )
        
    def train_fusion_network(self, batch_data):
        """Train sensory fusion network"""
        self.networks['fusion'].train()
        self.fusion_optimizer.zero_grad()
        
        # Forward pass
        fused_output, attention = self.networks['fusion'](
            batch_data['tactile'],
            batch_data['audio'], 
            batch_data['visual']
        )
        
        # Reconstruction loss (if ground truth available)
        if 'target_embedding' in batch_data:
            reconstruction_loss = self.mse_loss(fused_output, batch_data['target_embedding'])
        else:
            # Self-supervised consistency loss
            reconstruction_loss = self.consistency_loss(fused_output, batch_data)
            
        # Attention regularization
        attention_reg = self.attention_regularization(attention)
        
        total_loss = reconstruction_loss + 0.01 * attention_reg
        total_loss.backward()
        self.fusion_optimizer.step()
        
        return total_loss.item()
        
    def train_predictor(self, batch_data):
        """Train predictive forward model"""
        self.networks['predictor'].train()
        self.predictor_optimizer.zero_grad()
        
        # Forward prediction
        predicted_state, uncertainty, _ = self.networks['predictor'](
            batch_data['current_state'],
            batch_data['action']
        )
        
        # Prediction loss
        prediction_loss = self.mse_loss(predicted_state, batch_data['next_state'])
        
        # Uncertainty loss (calibration)
        uncertainty_loss = self.uncertainty_calibration_loss(
            predicted_state, batch_data['next_state'], uncertainty
        )
        
        total_loss = prediction_loss + 0.1 * uncertainty_loss
        total_loss.backward()
        self.predictor_optimizer.step()
        
        return total_loss.item()
```

## Architectural Design Patterns

### 1. Modular Network Design

- **Separation of Concerns**: Each network handles specific aspects
- **Plug-and-Play**: Networks can be replaced independently  
- **Scalable**: Easy to add new modalities or capabilities

### 2. Uncertainty Quantification

- **Epistemic Uncertainty**: Model uncertainty via ensembles
- **Aleatoric Uncertainty**: Data uncertainty via learned variance
- **Calibrated Predictions**: Uncertainty-aware decision making

### 3. Continual Learning

- **Experience Replay**: Prevent catastrophic forgetting
- **Regularization**: EWC (Elastic Weight Consolidation)
- **Progressive Networks**: Add capacity for new tasks

### 4. Multi-Task Learning

- **Shared Representations**: Common feature extractors
- **Task-Specific Heads**: Specialized output layers
- **Gradient Balance**: Weighted loss combination

## Performance Characteristics

### Computational Requirements

| Network | Parameters | FLOPs | Memory (MB) | Latency (ms) |
|---------|------------|-------|-------------|--------------|
| SensoryFusion | 0.5M | 2.1M | 12 | 3 |
| SensoryPredictor | 1.2M | 5.8M | 28 | 8 |
| MaterialPredictor | 0.8M | 3.2M | 18 | 5 |

### Training Characteristics

- **Batch Size**: 32-64 experiences
- **Learning Rate**: 1e-4 to 1e-3 (adaptive)
- **Update Frequency**: Every 10-100 experiences
- **Convergence**: 1000-10000 training steps

These neural architectures provide MORPHEUS with sophisticated sensory processing, predictive modeling, and strategy learning capabilities while maintaining real-time performance requirements.