from blake3 import blake3


def int_to_3d(number: int) -> list[int]:
    """
    Convierte un número entero en coordenadas 3D [X, Y, Z].

    - Divide los dígitos en 3 partes conservando el orden original.
    - Si no es divisible entre 3, el último elemento (Z) absorbe los dígitos extra.

    Args:
        number: Número entero de cualquier longitud.

    Returns:
        Lista de 3 enteros [X, Y, Z].
    """
    digits = str(abs(number))  # Trabajar con valor absoluto
    n = len(digits)
    chunk_size = n // 3

    x = int(digits[0 : chunk_size])
    y = int(digits[chunk_size : chunk_size * 2])
    z = int(digits[chunk_size * 2 :])  # Absorbe el resto si n % 3 != 0

    return [x, y, z]

def coordinates_from_index(arr: list[int]) -> tuple[int, int, int]:
    # 1. Convertimos el array a string para el hash
    s = ','.join(map(str, arr))
    
    # 2. Generamos el hash de 16 bytes
    h_bytes = blake3(s.encode()).digest(length=10)
    hash_entero = int.from_bytes(h_bytes, 'big')
    
    coordinates = int_to_3d(hash_entero)
    
    return tuple(coordinates)

