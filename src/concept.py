from src.concepts.rain import Rain
from src.neuron.neuron import Neuron

neuron = Neuron()
    
              
rain = Rain(1)

#rain not fall up
r1 = [rain.x, rain.not_(rain.fall().x), rain.fall().up().x]

print(r1)
print(all(r1))


