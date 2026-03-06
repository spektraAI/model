import numpy as np
import hashlib

class ConceptNode:
    def __init__(self, seed_structure):
        # Convertimos el string "carro" en un número entero para el generador
        seed_int = int(hashlib.sha256(seed_structure.encode()).hexdigest(), 16) % (2**32)
        
        # Fijamos la semilla localmente para este nodo
        rs = np.random.RandomState(seed_int)
        
        # Las 1,000 neuronas nacen de forma única para "carro"
        self.weights = rs.randn(1000, 1000) 
        self.maturity = 0