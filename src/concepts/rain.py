from typing import Self
import src.concepts.fall as c
from src.neuron.neuron import Neuron

neuron = Neuron()

class Rain:
    def __init__(self, state: int):
        self.state = state
        self.x = state
        self.sign = "rain"
        
    def rain(self, a) -> Self:
        return self
          
    def fall(self):
        return c.Fall(1)
    
    def down(self):
        return c.Down(1)
    
    def not_(self, a: int) -> int:
        return neuron.not_(a)   
    
    
    
