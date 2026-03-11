from pathlib import Path
from src.neuron.memory import BAN
from src.utils import word_to_image
from collections import Counter

ruta_actual = Path.cwd()

INPUT_PATH = ruta_actual / "input"
OUTPUT_PATH = ruta_actual / "output"
GRID = 28 * 7
RETINA = (GRID, GRID)

REDES = {} 

documento = [
    "a cat with four wheels",
    "a car is used for transportation",
    "a car has four wheels",
]

def preprocesar_texto(frase):
    palabras = frase.split()
    resultado_tuplas = [] # Aquí guardaremos las tuplas (img, label)
    
    for i in range(1, len(palabras) + 1):
        # 1. Construimos el fragmento de la frase
        frs = " ".join(palabras[:i])
        
        # 2. Creamos la tupla y la añadimos a la lista
        nombre_archivo = f"{i}.png"
        resultado_tuplas.append((nombre_archivo, frs))
        
        word_to_image(path=INPUT_PATH, filename=str(i), frase=frs, padding=2, wrap=True, size=(RETINA), fuente_size=12)

    return resultado_tuplas

def entrenar_frase(red: dict, frase: str):
    phrase_chunks = preprocesar_texto(frase)

    for i in range(1, len(phrase_chunks) + 1):
        red[i] = BAN()
        print(f"\n--- Entrenando BAN{i} con toda la secuencia ---")

        for img, label in phrase_chunks:
            nodos_contexto = [red[j] for j in range(1, i) if j in red]

            if nodos_contexto:
                red[i].train_from_upstream_(img, label, upstream=nodos_contexto)
            else:
                red[i].train_from_(img, label)


def entrenar_documento():
    for p_idx, parrafo in enumerate(documento, 1):
        print(f"\n{'═'*55}")
        print(f"  PÁRRAFO {p_idx}: '{parrafo}'")
        print(f"{'═'*55}")
        REDES[p_idx] = {}
        entrenar_frase(REDES[p_idx], parrafo)



def clasificar_documento(imagen: str, verbose: bool = True):
    print(f"\n{'═'*55}")
    print(f"  CONSULTA: {imagen}")
    print(f"{'═'*55}")

    for p_idx, red in REDES.items():
        print(f"\n  PÁRRAFO {p_idx}")
        print(f"  {'─'*40}")

        n_bans = len(red)

        for i in range(1, n_bans + 1):
            upstream = [red[j] for j in range(1, i) if j in red]

            if upstream:
                label, scores, _ = red[i].classify_chained_(
                    imagen, upstream=upstream, verbose=False
                )
            else:
                label, scores = red[i].classify_(imagen, verbose=False)

            # ── ordenar scores de mayor a menor ─────────────────
            ranking = sorted(scores.items(), key=lambda x: -x[1])

            winner       = ranking[0]
            second       = ranking[1] if len(ranking) > 1 else None
            third       = ranking[2] if len(ranking) > 1 else None
            
            bar = "█" * int(abs(winner[1]) * 20)

            linea = f"  BAN{i}  {winner[1]:+.4f}  {bar:<20}  \"{winner[0]}\""

            if second:
                linea += f"   2°: \"{second[0]}\" {second[1]:+.4f}"
                
            if third:
                linea += f"   3°: \"{third[0]}\" {third[1]:+.4f}"
                
            print(linea)

    print(f"\n{'═'*55}")
    
def memoria_documento():
    print(f"\n{'═'*55}")
    print(f"  MEMORIA TOTAL DEL DOCUMENTO")
    print(f"{'═'*55}")

    total_doc_mb = 0.0

    for p_idx, red in REDES.items():
        print(f"\n  PÁRRAFO {p_idx}")
        print(f"  {'─'*50}")
        print(f"  {'BAN':<8} {'W_fwd':>10} {'A_rows':>10} {'B_rows':>10} {'TOTAL':>10}")
        print(f"  {'─'*50}")

        total_parrafo_mb = 0.0

        for i, ban in red.items():
            report = ban.memory_usage()

            # extraer valores numéricos del reporte
            w     = float(report["W_fwd"].replace(" MB", ""))
            a     = float(report["_A_rows"].replace(" MB", ""))
            b     = float(report["_B_rows"].replace(" MB", ""))
            total = float(report["TOTAL"].replace(" MB", ""))

            total_parrafo_mb += total

            print(f"  BAN{i:<5} {w:>9.3f}MB {a:>9.3f}MB {b:>9.3f}MB {total:>9.3f}MB")

        print(f"  {'─'*50}")
        print(f"  {'SUBTOTAL':<8} {'':>10} {'':>10} {'':>10} {total_parrafo_mb:>9.3f}MB")

        total_doc_mb += total_parrafo_mb

    print(f"\n{'═'*55}")
    print(f"  PÁRRAFOS   : {len(REDES)}")
    print(f"  BANS TOTAL : {sum(len(r) for r in REDES.values())}")
    print(f"  {'─'*40}")
    print(f"  TOTAL DOC  : {total_doc_mb:.3f} MB")
    print(f"{'═'*55}\n")

    return total_doc_mb

entrenar_documento()
print(REDES)
clasificar_documento("2.png")




#memoria_documento()

    
