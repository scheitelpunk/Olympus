# 🪟 GASM-Roboting Windows Setup Guide

## Problem: spaCy Installation Fehler
Das spaCy-Paket hat C-Dependencies, die unter Windows schwer zu kompilieren sind.

## ✅ Lösung: Minimale Installation

### Schritt 1: Virtual Environment aktivieren
```powershell
# Stellen Sie sicher, dass Ihr venv aktiviert ist:
venv\Scripts\activate

# Prüfen Sie die Python-Version:
python --version
```

### Schritt 2: Minimale Requirements installieren
```powershell
# Installieren Sie nur die Kern-Abhängigkeiten:
pip install -r requirements-minimal.txt
```

### Schritt 3: System testen
```powershell
# Testen Sie das System:
python run_quick_demo.py demo
```

## 📦 Was wird installiert (Minimal):
- **numpy** - Mathematische Operationen
- **matplotlib** - 2D Visualisierung  
- **scipy** - Wissenschaftliche Berechnungen
- **fastapi/uvicorn** - Web-API
- **pytest** - Testing

## 🚀 Sofort Verfügbare Features:
1. **2D Spatial Agent** - Funktioniert ohne zusätzliche Dependencies
2. **GASM Bridge** - Dummy-Implementation für Constraints  
3. **SE(3) Utilities** - Mathematische Operationen
4. **Metrics System** - Pose-Fehler und Scoring
5. **Rule-based Planner** - Bewegungsplanung

## 🎯 Optionale Dependencies (bei Bedarf):
```powershell
# PyTorch (CPU-Version für Windows):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Computer Vision:
pip install opencv-python pillow

# 3D Physics Simulation:
pip install pybullet

# NLP (falls spaCy-Probleme weiterhin bestehen):
pip install transformers
```

## 🏃‍♂️ Sofort loslegen:
```powershell
# 2D Demo (funktioniert sofort):
python src/spatial_agent/agent_loop_2d.py --text "box above robot" --no_visualization --steps 10

# Test aller Module:
python run_quick_demo.py test
```

## ❌ Falls Probleme auftreten:
1. **"No module named 'torch'"** - Das ist OK, Fallbacks funktionieren
2. **"No module named 'cv2'"** - Das ist OK, Vision-Fallbacks aktiv
3. **"No module named 'pybullet'"** - Das ist OK, 2D-Demo funktioniert

## 🎉 Das System ist designed um ohne diese optionalen Dependencies zu funktionieren!

Die Spatial Agent Architektur hat robuste Fallback-Mechanismen und funktioniert auch mit minimalen Dependencies einwandfrei.