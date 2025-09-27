# 🛠️ GASM Development Tools

This directory contains development and testing tools for the GASM project. **These files are NOT deployed to Hugging Face Space** but are essential for local development, testing, and maintenance.

## 📁 Contents

### **Weight Management**
- **`manage_weights.py`** - CLI tool for GASM weight management
- **`test_weight_persistence.py`** - Comprehensive test suite for weight persistence
- **`WEIGHT_PERSISTENCE_README.md`** - Detailed technical documentation

## 🚀 Usage

### Weight Management CLI
```bash
# From project root
python dev-tools/manage_weights.py status
python dev-tools/manage_weights.py generate --force
python dev-tools/manage_weights.py remove
```

### Testing Weight Persistence
```bash
# From project root  
python dev-tools/test_weight_persistence.py
```

### Read Technical Documentation
```bash
# View detailed weight persistence documentation
cat dev-tools/WEIGHT_PERSISTENCE_README.md
```

## 🎯 Purpose

These tools provide:
- **🔧 Development Support** - CLI tools for weight management during development
- **🧪 Testing Framework** - Comprehensive validation of weight persistence logic
- **📚 Documentation** - Detailed technical specifications and troubleshooting

## ⚠️ Important Notes

- **Not deployed to HF Space** - These are development-only tools
- **Local development only** - Require local Python environment
- **Git ignored** - Won't be committed to production branches
- **Dependencies** - May require additional packages not in main requirements.txt

## 🔗 Related Files

**Production files (deployed to HF Space):**
- `../utils_weights.py` - Core weight persistence utilities
- `../app.py` - Main application with weight persistence integration
- `../README.md` - Public documentation with weight persistence info

---

**Note**: This directory and its contents are excluded from HF Space deployment via `.gitignore`