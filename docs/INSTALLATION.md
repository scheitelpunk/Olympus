# OLYMPUS Installation Guide

## Complete Setup Instructions for Project OLYMPUS

This guide provides comprehensive instructions for installing and configuring Project OLYMPUS in various environments, from development setups to production deployments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Development Environment Setup](#development-environment-setup)
4. [Production Installation](#production-installation)
5. [Docker Deployment](#docker-deployment)
6. [Configuration](#configuration)
7. [Verification & Testing](#verification--testing)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10/11 with WSL2
- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space (50GB recommended)
- **Network**: Internet connection for package downloads

### Recommended Requirements

- **Operating System**: Ubuntu 22.04 LTS or Rocky Linux 9
- **Python**: 3.11+
- **RAM**: 32GB or higher
- **Storage**: 100GB+ SSD storage
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (for ML workloads)
- **Network**: High-speed internet (for real-time operations)

### Hardware Compatibility

#### Supported Robotic Platforms
- Industrial robotic arms (Universal Robots, KUKA, ABB)
- Mobile robots (TurtleBot, Clearpath Robotics)
- Drone platforms (PX4, ArduPilot compatible)
- Custom robotic systems with ROS/ROS2 support

#### Sensor Requirements
- LiDAR sensors for spatial awareness
- RGB-D cameras for vision processing
- Force/torque sensors for haptic feedback
- IMU sensors for motion tracking

---

## Quick Installation

### One-Line Installation (Recommended for Development)

```bash
# Download and run the installation script
curl -fsSL https://raw.githubusercontent.com/olympus-ai/olympus/main/install.sh | bash
```

### Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/olympus-ai/olympus.git
cd olympus

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Install OLYMPUS
pip install -e .

# 5. Initialize system
olympus init --config configs/development.yaml
```

---

## Development Environment Setup

### Prerequisites Installation

#### Ubuntu/Debian
```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libomp-dev

# Install CUDA (for GPU support)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3
```

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 cmake git curl wget
brew install --cask docker

# Install additional ML libraries
brew install libomp openblas
```

#### Windows (WSL2)
```powershell
# Enable WSL2 and install Ubuntu
wsl --install -d Ubuntu-22.04

# Then follow Ubuntu instructions inside WSL2
```

### Python Environment Setup

```bash
# Create isolated Python environment
python3.11 -m venv olympus-env
source olympus-env/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Development Tools Installation

```bash
# Install additional development tools
pip install \
    jupyter \
    ipython \
    pytest-xvfb \
    pytest-benchmark \
    memory-profiler \
    line-profiler

# Install code quality tools
pip install \
    black \
    isort \
    flake8 \
    mypy \
    pylint \
    bandit
```

### IDE Configuration

#### VS Code Setup
```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-python.mypy-type-checker
code --install-extension ms-toolsai.jupyter
code --install-extension redhat.vscode-yaml

# Copy VS Code configuration
cp .vscode/settings.json.example .vscode/settings.json
```

#### PyCharm Configuration
```bash
# Configure PyCharm interpreter
# File -> Settings -> Project -> Python Interpreter
# Select the olympus-env/bin/python interpreter

# Enable code formatting
# File -> Settings -> Tools -> External Tools
# Configure Black formatter and isort
```

---

## Production Installation

### System Hardening

```bash
# Create dedicated olympus user
sudo useradd -r -s /bin/false olympus
sudo mkdir -p /opt/olympus
sudo chown olympus:olympus /opt/olympus

# Set up systemd service
sudo cp scripts/olympus.service /etc/systemd/system/
sudo systemctl enable olympus
```

### Production Dependencies

```bash
# Install production-specific packages
sudo apt install -y \
    nginx \
    postgresql-14 \
    redis-server \
    supervisor \
    certbot \
    fail2ban

# Configure PostgreSQL
sudo -u postgres psql << EOF
CREATE DATABASE olympus_prod;
CREATE USER olympus WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE olympus_prod TO olympus;
\q
EOF
```

### Application Installation

```bash
# Switch to olympus user
sudo -u olympus bash
cd /opt/olympus

# Clone and install application
git clone https://github.com/olympus-ai/olympus.git .
python3.11 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Configure production settings
cp configs/production.yaml.example configs/production.yaml
# Edit production.yaml with your settings

# Initialize production database
olympus db init --config configs/production.yaml
```

### SSL/TLS Setup

```bash
# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Configure nginx for OLYMPUS
sudo cp configs/nginx/olympus.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/olympus.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## Docker Deployment

### Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/olympus-ai/olympus.git
cd olympus

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Verify installation
docker-compose logs olympus
docker-compose ps
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  olympus:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - OLYMPUS_CONFIG=/app/configs/production.yaml
      - DATABASE_URL=postgresql://olympus:${DB_PASSWORD}@postgres:5432/olympus
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - olympus_data:/app/data
      - olympus_logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "olympus", "health", "--quick"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: olympus
      POSTGRES_USER: olympus
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./configs/nginx:/etc/nginx/conf.d
      - certbot_conf:/etc/letsencrypt
      - certbot_www:/var/www/certbot
    depends_on:
      - olympus
    restart: unless-stopped

volumes:
  olympus_data:
  olympus_logs:
  postgres_data:
  redis_data:
  certbot_conf:
  certbot_www:
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace olympus

# Apply manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get pods -n olympus
kubectl get services -n olympus
```

---

## Configuration

### Environment Variables

```bash
# Core Configuration
export OLYMPUS_ENV=production
export OLYMPUS_CONFIG=/opt/olympus/configs/production.yaml
export OLYMPUS_LOG_LEVEL=INFO

# Database Configuration
export DATABASE_URL=postgresql://olympus:password@localhost:5432/olympus_prod
export REDIS_URL=redis://localhost:6379/0

# Security Configuration
export SECRET_KEY="your-secret-key-here"
export JWT_SECRET="your-jwt-secret-here"
export ENCRYPTION_KEY="your-encryption-key-here"

# Hardware Configuration
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Safety Configuration
export SAFETY_MODE=strict
export ETHICS_VALIDATION=required
export HUMAN_OVERRIDE=enabled
```

### Configuration File Structure

```yaml
# configs/production.yaml
olympus:
  # System Configuration
  system:
    instance_name: "olympus-prod-001"
    environment: "production"
    debug_mode: false
    log_level: "INFO"
    
  # Database Configuration
  database:
    url: "${DATABASE_URL}"
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    
  # Redis Configuration
  redis:
    url: "${REDIS_URL}"
    max_connections: 100
    
  # Security Configuration
  security:
    secret_key: "${SECRET_KEY}"
    jwt_secret: "${JWT_SECRET}"
    encryption_key: "${ENCRYPTION_KEY}"
    session_timeout: 3600
    
  # ASIMOV Ethical Framework
  asimov:
    integrity_check_interval: 100  # milliseconds
    law_validation_strict: true
    human_override_enabled: true
    emergency_stop_enabled: true
    audit_logging: true
    
  # Safety Layer Configuration
  safety:
    strict_mode: true
    physics_limits:
      max_force: 20.0  # Newtons
      max_speed: 1.0   # m/s
      max_acceleration: 2.0  # m/s²
    spatial_limits:
      workspace_bounds: [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]]
      min_obstacle_distance: 0.15  # meters
    human_safety:
      min_safe_distance: 0.5  # meters
      warning_distance: 1.0   # meters
      
  # Module Configuration
  modules:
    nexus:
      enabled: true
      max_swarm_size: 100
      consensus_threshold: 0.67
      communication_range: 1000.0  # meters
      
    atlas:
      enabled: true
      transfer_validation_level: "high"
      sim2real_bridge_enabled: true
      
    prometheus:
      enabled: true
      health_check_interval: 5  # seconds
      predictive_maintenance: true
      self_repair_enabled: true
      
  # Integration Configuration
  integrations:
    gasm:
      enabled: true
      spatial_resolution: 0.01  # meters
      physics_timestep: 0.001   # seconds
      
    morpheus:
      enabled: true
      dream_simulation: true
      scenario_planning: true
      
  # Monitoring Configuration
  monitoring:
    metrics_enabled: true
    prometheus_endpoint: true
    grafana_dashboard: true
    alert_manager: true
```

### Hardware Configuration

```yaml
# configs/hardware.yaml
hardware:
  # Robotic Platforms
  robots:
    - id: "robot_001"
      type: "universal_robot_ur5e"
      ip_address: "192.168.1.100"
      control_mode: "position"
      safety_limits:
        max_velocity: 1.0
        max_acceleration: 2.0
        
    - id: "robot_002"
      type: "mobile_base_turtlebot3"
      ip_address: "192.168.1.101"
      control_mode: "velocity"
      
  # Sensors
  sensors:
    - id: "lidar_001"
      type: "velodyne_vlp16"
      ip_address: "192.168.1.200"
      frequency: 10  # Hz
      
    - id: "camera_001"
      type: "realsense_d435i"
      device_path: "/dev/video0"
      resolution: [640, 480]
      fps: 30
      
  # Actuators
  actuators:
    - id: "gripper_001"
      type: "robotiq_2f_140"
      control_interface: "modbus_tcp"
      ip_address: "192.168.1.110"
```

---

## Verification & Testing

### Basic System Verification

```bash
# Test installation
olympus --version
olympus health --comprehensive

# Test ethical framework
olympus ethics test --all-laws

# Test safety systems
olympus safety test --all-filters

# Test module integration
olympus modules test --all
```

### Hardware Integration Testing

```bash
# Test robotic platform connectivity
olympus hardware test --robots

# Test sensor integration
olympus hardware test --sensors

# Test end-to-end system
olympus integration-test --full-system
```

### Performance Benchmarking

```bash
# Run performance benchmarks
olympus benchmark --suite comprehensive

# Test load capacity
olympus load-test --concurrent-actions 100

# Memory and CPU profiling
olympus profile --duration 300  # 5 minutes
```

### Safety Validation

```bash
# Run safety test suite
olympus safety validate --all

# Test emergency procedures
olympus safety test-emergency --scenarios all

# Verify ethical compliance
olympus ethics validate --comprehensive
```

---

## Troubleshooting

### Common Installation Issues

#### Python Version Conflicts
```bash
# Issue: Wrong Python version
# Solution: Use specific Python version
python3.11 -m venv venv
source venv/bin/activate
which python  # Should show venv path
```

#### CUDA Installation Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install CUDA if missing
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Dependency Conflicts
```bash
# Clear pip cache
pip cache purge

# Reinstall with no cache
pip install --no-cache-dir -r requirements.txt

# Use conda for complex dependencies
conda create -n olympus python=3.11
conda activate olympus
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Runtime Issues

#### Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U olympus -d olympus_prod

# Reset password if needed
sudo -u postgres psql
\password olympus
```

#### Permission Errors
```bash
# Fix file permissions
sudo chown -R olympus:olympus /opt/olympus
sudo chmod -R 755 /opt/olympus

# Fix log directory permissions
sudo mkdir -p /var/log/olympus
sudo chown olympus:olympus /var/log/olympus
```

#### Memory Issues
```bash
# Check memory usage
free -h
olympus monitor --memory

# Optimize for low memory
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Network and Connectivity

#### Robot Connection Issues
```bash
# Test network connectivity
ping 192.168.1.100  # Robot IP
telnet 192.168.1.100 502  # Modbus port

# Check firewall settings
sudo ufw status
sudo ufw allow from 192.168.1.0/24

# Debug network issues
sudo tcpdump -i any host 192.168.1.100
```

#### Sensor Integration Issues
```bash
# Check USB devices
lsusb

# Check video devices
ls /dev/video*

# Test camera
v4l2-ctl --list-devices
olympus sensor test --camera camera_001
```

### Performance Issues

#### Slow Response Times
```bash
# Profile performance
olympus profile --duration 60

# Check system resources
htop
iotop

# Optimize database
olympus db optimize
olympus db vacuum --full
```

#### High Memory Usage
```bash
# Monitor memory usage
olympus monitor --memory --duration 300

# Clear caches
olympus cache clear --all

# Adjust memory limits
export OLYMPUS_MAX_MEMORY=8GB
```

### Safety System Issues

#### Emergency Stop Not Responding
```bash
# Test emergency stop
olympus safety emergency-stop --test

# Check safety system status
olympus safety status --detailed

# Reset safety systems
olympus safety reset --authorize "emergency_reset_code"
```

#### Ethical Validation Failures
```bash
# Verify Asimov kernel integrity
olympus ethics verify-integrity

# Check ethical evaluation logs
olympus logs ethics --tail 100

# Reset ethical framework
olympus ethics reset --confirm
```

### Log Analysis

```bash
# View system logs
olympus logs system --level ERROR --tail 50

# View safety logs
olympus logs safety --follow

# View audit logs
olympus logs audit --since "1 hour ago"

# Export logs for analysis
olympus logs export --format json --output /tmp/olympus_logs.json
```

### Getting Help

```bash
# Built-in diagnostics
olympus diagnose --full

# Generate support bundle
olympus support-bundle --output /tmp/olympus_support.tar.gz

# Check documentation
olympus docs --open

# Community support
olympus community --forum
```

---

## Post-Installation Tasks

### Security Hardening

```bash
# Change default passwords
olympus auth change-password --user admin

# Generate new API keys
olympus auth generate-keys --rotate-existing

# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # OLYMPUS API
sudo ufw allow 443/tcp   # HTTPS
```

### Monitoring Setup

```bash
# Configure log rotation
sudo cp configs/logrotate/olympus /etc/logrotate.d/

# Set up monitoring alerts
olympus monitoring setup-alerts --email admin@example.com

# Configure backups
olympus backup configure --destination s3://olympus-backups/
```

### Performance Optimization

```bash
# Tune database performance
olympus db tune --environment production

# Optimize cache settings
olympus cache optimize --memory-limit 2GB

# Configure load balancing
olympus cluster setup --nodes 3
```

---

## Validation Checklist

### Installation Verification

- [ ] OLYMPUS command-line interface working
- [ ] All required Python packages installed
- [ ] Database connection established
- [ ] Redis cache accessible
- [ ] Configuration files loaded correctly

### Safety System Verification

- [ ] ASIMOV kernel integrity verified
- [ ] Emergency stop system responsive
- [ ] Safety filters operational
- [ ] Human detection systems active
- [ ] Audit logging functional

### Hardware Integration Verification

- [ ] Robot platforms connected and responsive
- [ ] Sensor data streaming correctly
- [ ] Actuator control verified
- [ ] Network communication stable
- [ ] Real-time performance adequate

### Security Verification

- [ ] SSL/TLS certificates valid
- [ ] Authentication system working
- [ ] API access controls enforced
- [ ] Audit logs secure and tamper-proof
- [ ] Encryption keys properly managed

---

## Next Steps

After successful installation:

1. **Review Configuration**: Customize settings for your specific use case
2. **Run Tests**: Execute the full test suite to verify functionality
3. **Setup Monitoring**: Configure alerts and monitoring dashboards
4. **Train Operators**: Ensure all users understand safety procedures
5. **Plan Maintenance**: Schedule regular updates and health checks

For deployment guidance, see [DEPLOYMENT.md](DEPLOYMENT.md).  
For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).  
For testing procedures, see [TESTING.md](TESTING.md).

---

**Note**: Always prioritize safety during installation and testing. Never bypass safety systems or ethical validations, even in development environments.