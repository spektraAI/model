from matrix import ConceptMatrix
from utils.hashing import coordinates_from_index

MATRIX_SHAPE= 1_000_000_000

def posiciones_en_abecedario(palabra):
    abecedario = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
              'n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    palabra = palabra.lower()
    posiciones = []
    
    for letra in palabra:
        if letra in abecedario:
            posicion = abecedario.index(letra)
            posiciones.append(posicion + 1)
        else:
            posiciones.append(None)
    
    return posiciones


concepto = "cerebro"
definicion = ["animal", "de", "dos", "patas"]

concepto_raw_index = posiciones_en_abecedario(concepto)
print(f"concepto_raw_index: {concepto_raw_index}")


coordinates = coordinates_from_index(concepto_raw_index)
print(f"coordinates: {coordinates}")






cm = ConceptMatrix(shape=(MATRIX_SHAPE, MATRIX_SHAPE, MATRIX_SHAPE))

cm.set(coordinates, 2323232323.5)

print(cm.get(coordinates))