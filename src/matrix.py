import numpy as np
import sparse
import plotly.graph_objects as go
from typing import Any, List, Tuple, Union
from src.node import ConceptNode
from src.utils.hashing import coordinates_from_index, posiciones_en_abecedario

# Sentinel for empty cells
_EMPTY = object()

CoordList = List[Tuple[int, ...]]
Value = Union[float, int, str, CoordList, Any]


class ConceptMatrix:
    """
    Sparse N-dimensional matrix backed by a dict store.
    Each cell can hold any value: scalar, string, or an ordered
    list of coordinate tuples [(x1,y1,z1), (x2,y2,z2), ...].
    Empty cells return the _EMPTY sentinel — check with is_empty().
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self._matrix_storage: dict[Tuple[int, ...], Value] = {}
        self._node_storage = {}
        self.friction_threshold = 0.8

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, index: Tuple[int, ...]):
        if len(index) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)}-D index, got {len(index)}-D.")
        for axis, (i, dim) in enumerate(zip(index, self.shape)):
            if not (0 <= i < dim):
                raise IndexError(f"Index {i} out of bounds for axis {axis} (size {dim}).")

    @staticmethod
    def _is_coord_list(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(c, (list, tuple)) for c in value)
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def set(self, index: Tuple[int, ...], concept: str, value: Value):
        """
        Store any value at index.
          cm.set((x,y,z), 3.14)                    # scalar
          cm.set((x,y,z), [(4,3,3), (8,9,0)])      # ordered coord list
          cm.set((x,y,z), "label")                  # string
          cm.set((x,y,z), None)                     # removes the cell
        """
        self._validate(index)
        
        #impl validate concept 
        
        if value is None:
            self._matrix_storage.pop(index, None)
        elif self._is_coord_list(value):
            self._matrix_storage[index] = [tuple(c) for c in value]  # preserve order
            self.add_node(index, concept)
        else:
            self._matrix_storage[index] = value
            self.add_node(index, concept)
        
            
    def add_node(self, index: Tuple[int,...], concept: str):
        self._node_storage[index] = ConceptNode(self, index, concept)
            
    def get(self, index: Tuple[int, ...]) -> Value:
        """Return stored value, or _EMPTY sentinel if the cell is empty."""
        self._validate(index)
        return self._matrix_storage.get(index, _EMPTY)

    def is_empty(self, index: Tuple[int, ...]) -> bool:
        """True if the cell has no stored value."""
        return self.get(index) is _EMPTY

    def delete(self, index: Tuple[int, ...]):
        """Remove a cell (reset to empty)."""
        self._validate(index)
        self._matrix_storage.pop(index, None)

    # ------------------------------------------------------------------
    # Sparse export (scalars only)
    # ------------------------------------------------------------------

    def to_coo(self) -> sparse.COO:
        """
        Export numeric cells to sparse.COO.
        Cells containing coord lists or strings are skipped.
        """
        scalar_items = {
            k: v for k, v in self._matrix_storage.items()
            if isinstance(v, (int, float))
        }
        if not scalar_items:
            return sparse.COO(
                coords=np.zeros((len(self.shape), 0), dtype=int),
                data=np.array([], dtype=float),
                shape=self.shape,
            )
        coords = np.array(list(scalar_items.keys()), dtype=int).T
        data = np.array(list(scalar_items.values()), dtype=float)
        return sparse.COO(coords=coords, data=data, shape=self.shape)

    def get_coo_from_symbol(self, concept: str) -> tuple[int, int, int]:
        concepto_raw_index = posiciones_en_abecedario(concept)
        concept_index = coordinates_from_index(concepto_raw_index)
        
        return concept_index

    def add_concept(self, concept: str, definition: list[str]) -> tuple[int, int, int]:
        concepto_raw_index = posiciones_en_abecedario(concept)
        concept_index = coordinates_from_index(concepto_raw_index)

        
        concept_definition = []

        for c in definition:
            c_array = posiciones_en_abecedario(c)
            coo = coordinates_from_index(c_array)
            concept_definition.append(coo)        
        
        self.set(concept_index, concept, concept_definition)
        
        return concept_index


    def train(self, text: str, learning_rate: float = 0.05):
        """
        Entrenamiento por asociación temporal (Aprendizaje Hebbiano).
        """
        words = text.lower().split()
        # 1. Convertir palabras en nodos (recuperar de la Matrix)
        sequence = []
        for word in words:
            coords = self.get_coo_from_symbol(word)
            if coords not in self._node_storage:
                self.add_node(coords, concept=word)
            sequence.append(self._node_storage[coords])

        # 2. Refuerzo de Ventana (Contexto)
        # Por cada nodo, reforzamos la conexión con los vecinos inmediatos
        for i, node in enumerate(sequence):
            # Miramos 2 palabras adelante y 2 atrás (Ventana de contexto)
            start = max(0, i - 2)
            end = min(len(sequence), i + 3)
            
            for j in range(start, end):
                if i == j: continue
                
                neighbor = sequence[j]
                # Calculamos la distancia (a más cerca, más fuerza)
                dist_factor = 1.0 / abs(i - j)
                strength = learning_rate * dist_factor
                
                # El nodo crea o fortalece el puntero hacia su vecino
                node.add_pointer(neighbor.index, strength=strength)
                
        print(f"Entrenamiento completado: {len(words)} tokens procesados.")


    def send_signal(self, source_coords, target_coords, signal_vector):
        target_node = self._node_storage.get(target_coords)
        if not target_node:
            return None

        # 1. Asegurar dimensiones 1000x1
        signal_vector = np.array(signal_vector).reshape(1000, 1)
        identity_vector = target_node.get_identity_vector().reshape(1000, 1)

        # 2. Cálculo de Fricción (Coseno de Similitud invertido)
        norm_s = np.linalg.norm(signal_vector)
        norm_i = np.linalg.norm(identity_vector)
        
        if norm_s == 0 or norm_i == 0:
            friction = 1.0
        else:
            # Similitud de 1.0 (alineados) a 0.0 (ortogonales)
            dot_product = np.dot(signal_vector.flatten(), identity_vector.flatten())
            cos_sim = dot_product / (norm_s * norm_i)
            friction = 1.0 - max(0, cos_sim)

        # 3. Aplicar Madurez y Umbral
        effective_friction = friction * target_node.maturity

        if effective_friction > self.friction_threshold:
            # La señal es bloqueada por el Kernel
            return None 

        # 4. Atenuación y Activación
        attenuated_signal = signal_vector * (1.0 - effective_friction)
        return target_node.activate(attenuated_signal)


    def propagate(self, start_coords: Tuple[int, ...], initial_signal: np.ndarray, max_hops: int = 5):
        """
        Propaga una señal aplicando decaimiento, fricción y resonancia ontológica.
        """
        queue = [(start_coords, initial_signal, 1.0)]  # (coords, vector, energía_actual)
        results = []
        visited = set()

        print(f"--- Iniciando Propagación desde {start_coords} ---")

        while queue:
            current_coords, current_signal, energy = queue.pop(0)
            
            # El umbral de extinción ahora es más dinámico gracias a la resonancia
            if energy < 0.05 or max_hops <= 0: 
                continue
                
            node = self._node_storage.get(current_coords)
            if not node or current_coords in visited:
                continue
            
            visited.add(current_coords)
            
            # 1. El nodo procesa la señal
            output_signal = node.activate(current_signal)
            results.append((node.name, energy))
            
            # 2. Explorar punteros (relaciones)
            top_ptrs = node.get_top_pointers(limit=5) # Subimos a 5 para ver más resonancias
            
            for target_coords, strength in top_ptrs:
                target_node = self._node_storage.get(target_coords)
                if not target_node:
                    continue

                # --- NUEVA LÓGICA: RESONANCIA ONTOLÓGICA ---
                # Comparamos definiciones para modular la energía
                res_factor = self.calculate_resonance(node, target_node)
                
                # La energía ahora es producto de: 
                # Hábito (strength) * Lógica (res_factor) * Decaimiento (0.9)
                next_energy = energy * strength * res_factor * 0.9
                
                # Intentamos enviar la señal (aplica fricción vectorial)
                next_signal = self.send_signal(current_coords, target_coords, output_signal)
                
                if next_signal is not None:
                    queue.append((target_coords, next_signal, next_energy))
            
            max_hops -= 1
            
        return results

    def calculate_resonance(self, node_a, node_b):
        """
        Función auxiliar para medir la afinidad entre definiciones inmutables.
        """
        set_a = set(node_a.matrix.get_definitions_by_index(node_a.index))
        set_b = set(node_b.matrix.get_definitions_by_index(node_b.index))
        
        matches = set_a.intersection(set_b)
        
        if len(matches) > 0:
            # Bono: 1.0 + 15% por cada coincidencia semántica
            return min(1.0 + (len(matches) * 0.15), 2.0)
        
        # Penalización: Si no hay nada en común, el flujo se resiste
        return 0.7


    def get_definitions_by_index(self, index: Tuple[int, ...]):
        """
        Recupera la lista de conceptos definitorios asociados 
        a una coordenada específica en la Matrix.
        """
        # 1. Intentamos obtener el nodo en esa posición
        concept_definitions = self.get(index)
        
        if concept_definitions:
            # 2. Retornamos la definición inmutable guardada en el nodo
            # Usamos list() para asegurar que sea una copia trabajable
            return list(concept_definitions)
        
        # 3. Si no hay nada en esa coordenada, retornamos una lista vacía
        # o podrías lanzar una excepción según prefieras.
        print(f"Aviso: La coordenada {index} está vacía.")
        return []


    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def nnz(self) -> int:
        """Number of stored cells."""
        return len(self._matrix_storage)

    @property
    def density(self) -> float:
        """Fraction of cells that are occupied."""
        total = 1
        for d in self.shape:
            total *= d
        return self.nnz / total

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __getitem__(self, index: Tuple[int, ...]) -> Value:
        return self.get(index)

    def __setitem__(self, index: Tuple[int, ...], value: Value):
        self.set(index, value)

    def __repr__(self) -> str:
        return (
            f"ConceptMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"density={self.density:.2e})"
        )

