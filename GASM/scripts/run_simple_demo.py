#!/usr/bin/env python3
"""
Einfache GASM-Demo ohne PyTorch Dependencies
Funktioniert sofort mit den minimal requirements
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

class SimpleGASMBridge:
    """Einfache GASM-Bridge ohne externe Dependencies"""
    
    def process(self, text: str):
        """Verarbeitet Text zu räumlichen Constraints"""
        print(f"📝 Verarbeite Text: '{text}'")
        
        # Parse Entitäten
        entities = []
        if "box" in text.lower():
            entities.append("box")
        if "robot" in text.lower():
            entities.append("robot")
        if "sensor" in text.lower():
            entities.append("sensor")
        if "conveyor" in text.lower():
            entities.append("conveyor")
            
        # Parse Constraints  
        constraints = []
        if "above" in text.lower():
            constraints.append("above")
        if "below" in text.lower():
            constraints.append("below")
        if "left" in text.lower():
            constraints.append("left")
        if "right" in text.lower():
            constraints.append("right")
        if "near" in text.lower():
            constraints.append("near")
            
        print(f"🎯 Gefundene Entitäten: {entities}")
        print(f"🔗 Gefundene Constraints: {constraints}")
        
        return {
            "entities": entities,
            "constraints": constraints,
            "targets": {entity: {"x": random.uniform(0, 10), "y": random.uniform(0, 8)} 
                       for entity in entities}
        }

class SimpleSpatialAgent:
    """Einfacher 2D Spatial Agent"""
    
    def __init__(self, width=10, height=8):
        self.width = width
        self.height = height
        self.entities = {}
        self.gasm = SimpleGASMBridge()
        
    def initialize_entities(self, entity_names):
        """Initialisiert Entitäten mit Zufallspositionen"""
        for name in entity_names:
            self.entities[name] = {
                "x": random.uniform(1, self.width-1),
                "y": random.uniform(1, self.height-1),
                "size": 0.5
            }
        print(f"🎲 Initialisiert {len(self.entities)} Entitäten")
        
    def evaluate_constraints(self, constraints):
        """Bewertet wie gut die Constraints erfüllt sind"""
        if len(self.entities) < 2:
            return 1.0
            
        entities = list(self.entities.values())
        total_score = 0.0
        
        for constraint in constraints:
            if constraint == "above":
                # Box sollte über Robot sein
                if "box" in self.entities and "robot" in self.entities:
                    box_y = self.entities["box"]["y"]
                    robot_y = self.entities["robot"]["y"]
                    score = max(0, box_y - robot_y)  # Positiv wenn box über robot
                    total_score += score
                    
            elif constraint == "near":
                # Entitäten sollten nah beieinander sein
                if len(entities) >= 2:
                    dist = np.linalg.norm([
                        entities[0]["x"] - entities[1]["x"],
                        entities[0]["y"] - entities[1]["y"]
                    ])
                    score = max(0, 3.0 - dist)  # Gut wenn Abstand < 3
                    total_score += score
                    
        return total_score
        
    def optimize_step(self, constraints):
        """Ein Optimierungsschritt"""
        best_score = self.evaluate_constraints(constraints)
        best_positions = {name: dict(pos) for name, pos in self.entities.items()}
        
        # Versuche kleine Änderungen
        for name in self.entities:
            for _ in range(5):  # 5 Versuche pro Entität
                # Sichere aktuelle Position
                old_x, old_y = self.entities[name]["x"], self.entities[name]["y"]
                
                # Kleine zufällige Änderung
                self.entities[name]["x"] += random.uniform(-0.5, 0.5)
                self.entities[name]["y"] += random.uniform(-0.5, 0.5)
                
                # Bounds checking
                self.entities[name]["x"] = max(0.5, min(self.width-0.5, self.entities[name]["x"]))
                self.entities[name]["y"] = max(0.5, min(self.height-0.5, self.entities[name]["y"]))
                
                # Bewerte neue Position
                score = self.evaluate_constraints(constraints)
                if score > best_score:
                    best_score = score
                    best_positions = {name: dict(pos) for name, pos in self.entities.items()}
                else:
                    # Zurücksetzen wenn schlechter
                    self.entities[name]["x"] = old_x
                    self.entities[name]["y"] = old_y
                    
        # Verwende beste Positionen
        self.entities = best_positions
        return best_score
        
    def visualize(self, iteration, score):
        """Einfache Visualisierung"""
        plt.clf()
        colors = {"box": "red", "robot": "blue", "sensor": "green", "conveyor": "brown"}
        
        for name, pos in self.entities.items():
            color = colors.get(name, "black")
            plt.scatter(pos["x"], pos["y"], s=200, c=color, label=name, alpha=0.7)
            plt.text(pos["x"]+0.1, pos["y"]+0.1, name, fontsize=8)
            
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.grid(True, alpha=0.3)
        plt.title(f"Iteration {iteration} | Score: {score:.3f}")
        plt.legend()
        plt.pause(0.1)
        
    def run_optimization(self, text, max_iterations=20, visualize=True):
        """Führt die Optimierung aus"""
        print("🚀 GASM Spatial Agent - Einfache Demo")
        print("=" * 50)
        
        # Text verarbeiten
        result = self.gasm.process(text)
        entities = result["entities"]
        constraints = result["constraints"]
        
        if not entities:
            print("❌ Keine Entitäten gefunden!")
            return
            
        # Entitäten initialisieren
        self.initialize_entities(entities)
        
        if visualize:
            plt.figure(figsize=(8, 6))
            plt.ion()
            
        print(f"\n🔄 Starte Optimierung für {max_iterations} Iterationen...")
        
        for i in range(max_iterations):
            score = self.optimize_step(constraints)
            
            if visualize:
                self.visualize(i, score)
                
            print(f"Iteration {i+1:2d}: Score = {score:.3f}")
            
            # Konvergenz check
            if i > 5 and score > 3.0:  # Guter Score erreicht
                print("✅ Konvergiert!")
                break
                
        if visualize:
            plt.ioff()
            print("\n📊 Finale Positionen:")
            for name, pos in self.entities.items():
                print(f"  {name}: ({pos['x']:.2f}, {pos['y']:.2f})")
            plt.show()
            
        return self.entities

def main():
    """Hauptfunktion"""
    import sys
    
    agent = SimpleSpatialAgent()
    
    # Standard-Beispiel oder aus Kommandozeile
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Place box above robot"
        
    print(f"🎯 Aufgabe: '{text}'")
    
    # Optimierung ausführen (ohne Visualisierung im Batch-Modus)
    result = agent.run_optimization(text, max_iterations=20, visualize=False)
    
    print("\n🎉 Demo erfolgreich beendet!")

if __name__ == "__main__":
    main()