from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"
GRID = 28 * 7
RETINA = (GRID, GRID)

RED1= {}
frase1 = "a car is a road vehicle that is powered by an engine and is able to carry a small number of people."
frase1_chunks = []

def preprocesar_texto(frase):
    palabras = frase.split()
    
    for i in range(0, len(palabras) + 1):
        frs = " ".join(palabras[:i])
        frase1_chunks.append(frs)
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=2, wrap=True, size=(RETINA), fuente_size=12)
        
def entrenar_frase1(frase):
    preprocesar_texto(frase)
    
    for i, l in enumerate(frase1_chunks):
        
        if i == 0:
            continue
        
        RED1[i] = BAN()
        print(f"Added BAN: {l}")
                
        if i == 1:
            RED1[i].train_from_(filename=f"{i}.png", label=l)
            print(f"Trained BAN: {l}")
            continue
        

        red = list(RED1.values())[:-1]
        
        RED1[i].train_from_upstream_(filename=f"{i}.png", label=l, upstream=red)

        

    #ban1.summary()
    #ban1.memory_usage()
    #ban3.save("models/ban_v1.pkl")

def detectar_frase():
    red = list(RED1.values())
    ultima_ban = red[-1]
    
    result = ultima_ban.classify_chained_("1.png",  upstream=red)
    
    print(result)



entrenar_frase1(frase1)
detectar_frase()














def reconstruir_frase(clasificacion):
    prefix, frases_scores = clasificacion
    # add temperature
    # obtener frases válidas
    sentences = [s for s in frases_scores.keys() if s.strip()]

    split_sentences = [s.split() for s in sentences]
    max_len = max(len(s) for s in split_sentences)

    result = []

    for i in range(max_len):
        words = [s[i] for s in split_sentences if len(s) > i]
        word = Counter(words).most_common(1)[0][0]
        result.append(word)

    return result, " ".join(result)