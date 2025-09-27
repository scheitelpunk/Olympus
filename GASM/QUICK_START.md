# 🚀 GASM-Roboting Quick Start Guide

## ✅ Sofort Loslegen (Windows-freundlich)

### 1. Dependencies installieren
```powershell
# Minimale Dependencies (funktioniert sofort):
pip install -r requirements-minimal.txt
```

### 2. Einfache Demo starten
```powershell
# Automatische Demo:
python scripts/run_simple_demo.py "box above robot"

# Eigene Aufgaben:
python scripts/run_simple_demo.py "sensor near conveyor"
python scripts/run_simple_demo.py "robot left of box"
```

### 3. Erweiterte Demo (bei vollständigen Dependencies)
```powershell
# Vollständige Demo mit Visualisierung:
python scripts/run_quick_demo.py demo
```

## 🎯 Was funktioniert sofort:

### ✅ Simple Demo (scripts/run_simple_demo.py)
- **Keine PyTorch/spaCy Dependencies** erforderlich
- **Text-zu-Constraints Parsing**: Versteht "above", "below", "near", etc.
- **Spatial Reasoning**: Korrekte Objektpositionierung
- **Iterative Optimierung**: Automatische Konvergenz
- **Robuste Architektur**: Expandierbar für echtes GASM

### 📊 Beispiel-Output:
```
🎯 Aufgabe: 'box above robot'
📝 Verarbeite Text: 'box above robot'
🎯 Gefundene Entitäten: ['box', 'robot']
🔗 Gefundene Constraints: ['above']
🎲 Initialisiert 2 Entitäten
🔄 Starte Optimierung für 20 Iterationen...
Iteration  7: Score = 6.754
✅ Konvergiert!
🎉 Demo erfolgreich beendet!
```

## 🏗️ Projektstruktur

```
GASM-Roboting/
├── scripts/                    # 🆕 Demo & Utility Scripts
│   ├── run_simple_demo.py     # Einfache Demo ohne Dependencies  
│   ├── run_quick_demo.py      # Vollständige Demo
│   ├── install_windows.py     # Windows Installation Helper
│   └── test_*.py              # Test Scripts
├── docs/                      # 🆕 Alle Dokumentation
│   ├── WINDOWS_SETUP.md       # Windows Setup Guide
│   ├── ARCHITECTURE.md        # System Architektur
│   └── *.md                   # Weitere Dokumentation
├── examples/legacy/           # 🆕 Legacy Code
│   ├── app.py                 # Original Gradio App
│   ├── fastapi_endpoint.py    # Alt-API (→ src/api/)
│   └── gasm_core.py           # Alt-GASM (→ src/gasm/)
├── src/                       # Hauptcode
│   ├── spatial_agent/         # GASM Spatial Agent System
│   ├── gasm/                  # GASM Core (migriert)
│   └── api/                   # FastAPI Endpoints (migriert)
├── tests/                     # Test Suite
│   └── reports/               # 🆕 Test Reports
├── assets/                    # URDF & Simulation Assets
├── tools/dev/                 # 🆕 Development Tools (ex dev-tools/)
├── coordination/              # Claude Flow Coordination ✅
├── memory/                    # Claude Flow Memory ✅
├── claude-flow*               # Claude Flow Scripts ✅
└── requirements*.txt          # Dependencies
```

## 🔧 Entwicklung

### Lokale Installation:
```powershell
# Entwicklungsumgebung:
pip install -e .

# CLI-Tools nutzen:
gasm-agent-2d --text "box above robot"
gasm-demo
```

### GASM Integration:
```python
# In src/spatial_agent/gasm_bridge.py:
def process(text: str) -> dict:
    # HIER ECHTES GASM EINFÜGEN:
    # return GASM.process(text)
    pass
```

## 🎉 Das war's!

Das System ist **sofort einsatzbereit** mit intelligenten Fallbacks und kann schrittweise mit vollständigen Dependencies erweitert werden.

**Nächste Schritte:** Echtes GASM in `gasm_bridge.py` integrieren! 🚀