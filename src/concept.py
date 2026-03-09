import src.concepts as c
from src.neuron.neuron import Neuron

neuron = Neuron()
    
              
rain = c.Rain(1)

#rain not fall up
r1 = [rain.x, rain.not_(rain.fall().x), rain.fall().down().x]

print(r1)
print(all(r1))


you = c.You(1)

#you like coffe
r2 = [you.x, you.like().x, you.like().coffe().x]

print(r2)
print(all(r2))