# GASM-Roboting API

A comprehensive FastAPI application providing geometric assembly state machine (GASM) capabilities with spatial agent control and advanced natural language processing.

## 🚀 Features

- **Core API Processing**: Text processing with GASM-enhanced LLM capabilities
- **Spatial Agent Control**: 6-DOF pose management and motion planning
- **GASM Integration**: Natural language to geometric constraint conversion
- **Authentication & Authorization**: JWT and API key based security
- **Rate Limiting**: Configurable request rate limiting with Redis support
- **Comprehensive Documentation**: Interactive OpenAPI/Swagger documentation
- **Production Ready**: Docker deployment with monitoring and logging

## 📁 Project Structure

```
src/api/
├── __init__.py                 # API package initialization
├── main.py                     # FastAPI application entry point
├── v1/                         # API version 1
│   ├── __init__.py
│   ├── endpoints.py           # Core text processing endpoints
│   ├── spatial_endpoints.py   # Spatial agent control endpoints
│   └── gasm_endpoints.py      # GASM-specific processing endpoints
├── middleware/                 # Custom middleware components
│   ├── __init__.py
│   ├── auth.py                # Authentication middleware
│   ├── rate_limit.py          # Rate limiting middleware
│   ├── cors.py                # CORS configuration
│   ├── logging.py             # Request logging middleware
│   └── errors.py              # Error handling middleware
├── models/                     # Pydantic data models
│   ├── __init__.py
│   ├── base.py                # Base response models
│   ├── auth.py                # Authentication models
│   ├── spatial.py             # Spatial agent models
│   └── gasm.py                # GASM processing models
├── tests/                      # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest configuration
│   ├── test_core_endpoints.py # Core API tests
│   └── test_spatial_endpoints.py # Spatial API tests
├── docker-compose.yml          # Docker deployment configuration
├── Dockerfile                  # Container build configuration
└── README.md                   # This file
```

## 🛠 Installation & Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Redis (for rate limiting and caching)
- PostgreSQL (for persistent data storage)

### Local Development

1. **Clone the repository and navigate to the API directory:**
   ```bash
   cd /mnt/c/dev/coding/GASM-Roboting/src/api
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export ENVIRONMENT=development
   export DEBUG=true
   export GASM_JWT_SECRET=your_jwt_secret_key
   ```

5. **Run the development server:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

### Docker Deployment

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Monitoring: http://localhost:3000 (Grafana)

## 📖 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Core Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/` | GET | Service information | No |
| `/health` | GET | Health check | No |
| `/api/v1/core/process` | POST | Process text with GASM | Yes |
| `/api/v1/core/analyze` | POST | Geometric analysis | Yes |
| `/api/v1/core/batch` | POST | Batch text processing | Yes |

### Spatial Agent Endpoints

| Endpoint | Method | Description | Permission |
|----------|--------|-------------|------------|
| `/api/v1/spatial/status` | GET | Agent status | Read |
| `/api/v1/spatial/pose` | POST | Set/move to pose | Spatial Control |
| `/api/v1/spatial/plan` | POST | Motion planning | Spatial Control |
| `/api/v1/spatial/constraints` | POST | Manage constraints | Spatial Control |
| `/api/v1/spatial/process` | POST | Natural language control | GASM Process |
| `/api/v1/spatial/metrics` | POST | Calculate metrics | Read |

### GASM Processing Endpoints

| Endpoint | Method | Description | Permission |
|----------|--------|-------------|------------|
| `/api/v1/gasm/status` | GET | GASM system status | Read |
| `/api/v1/gasm/process` | POST | Process spatial instruction | GASM Process |
| `/api/v1/gasm/analyze` | POST | Spatial scene analysis | GASM Process |
| `/api/v1/gasm/constraints` | GET | Supported constraints | Read |
| `/api/v1/gasm/examples` | GET | Processing examples | Read |

## 🔐 Authentication

The API supports two authentication methods:

### JWT Bearer Token
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/core/info
```

### API Key
```bash
curl -H "X-API-Key: <api_key>" http://localhost:8000/api/v1/core/info
```

### Permission Scopes
- `read`: Basic read access
- `write`: Write access to non-critical endpoints  
- `admin`: Full administrative access
- `spatial:control`: Spatial agent control
- `gasm:process`: GASM processing capabilities
- `system:monitor`: System monitoring access

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/test_core_endpoints.py

# With coverage
pytest --cov=src.api --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Authentication Tests**: Security and permissions
- **Validation Tests**: Request/response validation
- **Error Handling Tests**: Error scenarios

## 🚀 Deployment

### Production Environment Variables

```bash
ENVIRONMENT=production
DEBUG=false
GASM_JWT_SECRET=<secure_jwt_secret>
POSTGRES_PASSWORD=<secure_password>
REDIS_URL=redis://redis:6379
ALLOWED_ORIGINS=https://your-domain.com
```

### Monitoring & Observability

- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Metrics visualization (port 3000)
- **Structured Logging**: JSON formatted logs
- **Health Checks**: Automated health monitoring
- **Performance Metrics**: Request timing and throughput

### Production Checklist

- [ ] Set secure JWT secret
- [ ] Configure HTTPS with SSL certificates
- [ ] Set up database backups
- [ ] Configure log aggregation
- [ ] Set up monitoring alerts
- [ ] Review rate limiting settings
- [ ] Audit authentication configuration

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | development | No |
| `DEBUG` | Enable debug mode | false | No |
| `GASM_JWT_SECRET` | JWT signing secret | Generated | Production |
| `ALLOWED_ORIGINS` | CORS allowed origins | * | Production |
| `REDIS_URL` | Redis connection URL | None | Rate Limiting |
| `DATABASE_URL` | Database connection URL | SQLite | Production |

### Feature Flags

The API includes several optional features that can be enabled/disabled:

- **Spatial Components**: Requires spatial agent libraries
- **GASM Processing**: Requires GASM core libraries  
- **Redis Rate Limiting**: Falls back to in-memory
- **Database Persistence**: Falls back to in-memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

### Code Style
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black src/api/
isort src/api/

# Check style
flake8 src/api/
mypy src/api/
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🆘 Support

- **Documentation**: http://localhost:8000/docs
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 🎯 Migration Notes

This API represents a complete migration and enhancement of the original `fastapi_endpoint.py` with the following improvements:

### ✅ **Completed Migrations**

1. **✅ Directory Structure**: Organized into proper packages
2. **✅ Enhanced Endpoints**: All original functionality migrated and expanded
3. **✅ Authentication**: JWT and API key support with permissions
4. **✅ Rate Limiting**: Configurable with Redis support
5. **✅ Spatial Agents**: New spatial control capabilities
6. **✅ GASM Integration**: Enhanced natural language processing
7. **✅ Error Handling**: Comprehensive error responses
8. **✅ Documentation**: Interactive OpenAPI documentation
9. **✅ Testing**: Comprehensive test suite
10. **✅ Deployment**: Docker and production configuration

### 🔧 **Backward Compatibility**

The new API maintains backward compatibility with existing endpoints while adding new capabilities:

- All original `/process`, `/analyze`, `/batch` endpoints are preserved
- Enhanced with better error handling and validation
- Additional metadata and debugging information
- Improved performance and reliability

### 🚀 **New Capabilities**

- **Spatial Agent Control**: 6-DOF pose management and motion planning
- **Natural Language Spatial Commands**: Convert text to robot actions
- **Advanced Authentication**: Role-based access control
- **Production Monitoring**: Health checks and metrics
- **Comprehensive Testing**: Automated test coverage
- **Container Deployment**: Docker and orchestration support