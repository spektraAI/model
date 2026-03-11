from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp
from PIL import Image, ImageFilter, ImageOps

# ══════════════════════════════════════════════════════════════════════════════
# Constantes globales
# ══════════════════════════════════════════════════════════════════════════════

GRID      = 28 * 7        # lado de la retina cuadrada  →  196 px
LABEL_DIM = 16 * 22       # dimensión del vector de etiqueta  →  352

INPUT_DIR  = Path.cwd() / "input"
OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de bajo nivel
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(image_input) -> Tuple[sp.csr_matrix, Tuple[int, int]]:
    """
    Imagen  →  vector binario {0, 1} de longitud GRID²  (csr_matrix).

    Parámetros
    ----------
    image_input : str | Path | np.ndarray | PIL.Image

    Retorno
    -------
    (vector_sparse, tamaño_original)
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        arr = (image_input if image_input.ndim == 2
               else np.mean(image_input, axis=2).astype(np.uint8))
        img = Image.fromarray(arr).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError(f"Formato no soportado: {type(image_input)}")

    img  = ImageOps.pad(img, (GRID, GRID), color=255)
    arr  = np.array(img, dtype=float)
    size = img.size

    vec = np.where(arr < arr.mean(), 1.0, 0.0).flatten()
    return sp.csr_matrix(vec), size


def _encode_label(idx: int, dim: int = LABEL_DIM) -> np.ndarray:
    """Vector bipolar {-1, +1} reproducible para la etiqueta idx."""
    rng = np.random.default_rng(seed=idx * 31337)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def _vec_to_image(vec, size: Tuple[int, int] = (200, 200)) -> Image.Image:
    """Vector binario/bipolar  →  PIL.Image redimensionada."""
    if sp.issparse(vec):
        vec = vec.toarray().flatten()
    grid      = vec.reshape(GRID, GRID)
    arr       = ((1 - grid) * 255).astype(np.uint8)
    img_small = Image.fromarray(arr, mode="L")
    img_small = img_small.filter(ImageFilter.GaussianBlur(radius=0.6))
    img_out   = img_small.resize(size, Image.LANCZOS)
    arr_out   = np.array(img_out)
    arr_out   = np.where(arr_out < 128, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_out, mode="L")


# ══════════════════════════════════════════════════════════════════════════════
# BAN  —  Red Asociativa Bidireccional base
# ══════════════════════════════════════════════════════════════════════════════

class BAN:
    """
    Bidirectional Associative Network.

    Aprende la correspondencia  imagen ↔ etiqueta  mediante pseudoinversa:

        W_fwd  =  pinv(A)  @  B

    Atributos públicos
    ------------------
    labels      : list[str]              — etiquetas en orden de registro
    label_vecs  : dict[str, np.ndarray]  — vectores bipolares por etiqueta
    W_fwd       : np.ndarray (GRID², LABEL_DIM)
    """

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self):
        self.labels      : list[str]             = []
        self.label_vecs  : dict[str, np.ndarray] = {}
        self._A_rows     : list[sp.csr_matrix]   = []
        self._B_rows     : list[np.ndarray]      = []
        self.A_mat       : sp.csr_matrix | None  = None
        self.B_mat       : np.ndarray | None     = None
        self.W_fwd       : np.ndarray | None     = None
        self._fitted     : bool                  = False
        self._seen_hashes: set                   = set()

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train_from_(self, filename: str, label: str,
                    save_output: bool = True) -> "BAN":
        """
        Registra un par (imagen, label) y reajusta W_fwd.

        Parámetros
        ----------
        filename    : nombre del archivo dentro de ./input/
        label       : etiqueta semántica  (ej. "gato")
        save_output : guarda imagen procesada en ./output/

        Retorno
        -------
        self  →  permite encadenamiento
        """
        label = label.strip().lower()
        ruta  = INPUT_DIR / filename

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró '{ruta}'.\n"
                f"Archivos en /input: {[f.name for f in INPUT_DIR.iterdir()]}"
            )

        vec_A, original_size = _preprocess(ruta)

        fingerprint = vec_A.toarray().tobytes()
        if fingerprint in self._seen_hashes:
            print(f"  ⚠️  '{label}' ya registrado (imagen duplicada), se omite.")
            return self
        self._seen_hashes.add(fingerprint)

        if label not in self.label_vecs:
            idx = len(self.labels)
            self.labels.append(label)
            self.label_vecs[label] = _encode_label(idx)

        vec_B = self.label_vecs[label]
        self._A_rows.append(vec_A)
        self._B_rows.append(vec_B)

        self._fit()

        if save_output:
            img_out = _vec_to_image(vec_A, original_size)
            img_out.save(OUTPUT_DIR / f"{label}_ban_entrenada.png")

        print(f"  ✓ '{label}'  ←  {filename}  |  "
              f"patrones={len(self.labels)}  muestras={len(self._A_rows)}")
        return self

    def _fit(self):
        """Recalcula W_fwd con pseudoinversa sobre todos los pares acumulados."""
        A_dense    = sp.vstack(self._A_rows).toarray().astype(np.float32)
        B_dense    = np.stack(self._B_rows).astype(np.float32)
        self.W_fwd = np.linalg.pinv(A_dense) @ B_dense
        self._fitted = True

    # ── Inferencia ────────────────────────────────────────────────────────────

    def _forward(self, A) -> np.ndarray:
        """Proyecta vector imagen → vector etiqueta bipolar."""
        if sp.issparse(A):
            A = A.toarray().flatten()
        raw = A.astype(np.float32) @ self.W_fwd
        return np.where(raw >= 0, 1.0, -1.0)

    def classify_(self, image_input,
                  verbose: bool = True) -> tuple[str, dict]:
        """
        Clasifica una imagen por similitud coseno en el espacio de etiquetas.

        Parámetros
        ----------
        image_input : str | Path | PIL.Image | np.ndarray

        Retorno
        -------
        (label_ganadora, scores_dict)
        """
        if not self._fitted:
            raise RuntimeError("BAN sin entrenar. Llama train_from_() primero.")

        if isinstance(image_input, str):
            image_input = INPUT_DIR / image_input

        vec, _  = _preprocess(image_input)
        B_hat   = self._forward(vec)

        scores = {
            lbl: float(np.dot(B_hat, lv)) /
                 (np.linalg.norm(B_hat) * np.linalg.norm(lv) + 1e-9)
            for lbl, lv in self.label_vecs.items()
        }
        winner = max(scores, key=scores.get)

        if verbose:
            print(f"\n🏆 Label   : {winner}")
            print("📊 Scores  :")
            for lbl, score in sorted(scores.items(), key=lambda x: -x[1]):
                marker = " ← ganador" if lbl == winner else ""
                print(f"   {lbl:<16} {score:+.5f}{marker}")

        return winner, scores

    # ── Utilidades ────────────────────────────────────────────────────────────

    def summary(self):
        print("\n── BAN Summary ──────────────────────────────────────")
        print(f"   Patrones  : {self.labels}")
        print(f"   Muestras  : {len(self._A_rows)}")
        if self._fitted:
            print(f"   W_fwd     : {self.W_fwd.shape}")
        else:
            print("   Estado    : sin entrenar")
        print("─────────────────────────────────────────────────────\n")

    def memory_usage(self) -> dict:
        def mb(arr):       return arr.nbytes / 1024**2 if arr is not None else 0
        def mb_sp(m):      return m.data.nbytes / 1024**2 if m is not None else 0

        A_mb  = sum(mb_sp(v) for v in self._A_rows)
        B_mb  = sum(v.nbytes for v in self._B_rows) / 1024**2
        total = A_mb + B_mb + mb(self.W_fwd)

        report = {
            "_A_rows": f"{A_mb:.2f} MB",
            "_B_rows": f"{B_mb:.2f} MB",
            "W_fwd"  : f"{mb(self.W_fwd):.2f} MB",
            "TOTAL"  : f"{total:.2f} MB",
        }
        print("\n── BAN Memory Usage ─────────────────────────────────")
        for k, v in report.items():
            if k == "TOTAL": print("─" * 40)
            print(f"   {k:<16} {v:>10}")
        print("─────────────────────────────────────────────────────\n")
        return report

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  💾 BAN guardada → {path}  "
              f"({path.stat().st_size / 1024**2:.2f} MB)")

    @staticmethod
    def load(path: str | Path) -> "BAN":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró '{path}'.")
        with open(path, "rb") as f:
            instance = pickle.load(f)
        print(f"  ✅ BAN cargada  ← {path}  |  patrones={instance.labels}")
        return instance


# ══════════════════════════════════════════════════════════════════════════════
# PlasticBAN  —  BAN con Plasticidad Neuronal Localizada
# ══════════════════════════════════════════════════════════════════════════════

class PlasticBAN(BAN):
    """
    Extiende BAN con plasticidad sináptica bio-inspirada.

    Mecanismos implementados
    ────────────────────────
    LTP  (Long-Term Potentiation)
        Sinapsis activas que reducen el error se fortalecen.

    LTD  (Long-Term Depression)
        Sinapsis inactivas en la zona plástica se debilitan.

    Fatiga neuronal
        Neuronas sobreusadas se inhiben temporalmente
        y se recuperan de forma gradual.

    Consolidación sináptica
        Las sinapsis más antiguas (más veces modificadas)
        resisten el cambio — simulando memoria a largo plazo.

    Localidad espacial
        Solo se modifica la vecindad del patrón activo,
        no la matriz completa.

    Reconsolidación
        Cada llamada a classify_() actualiza W_fwd levemente,
        igual que el hipocampo reescribe el recuerdo al recordarlo.

    Parámetros
    ──────────
    ltp_rate   : tasa de potenciación                  (default 0.01)
    ltd_rate   : tasa de depresión                     (default 0.005)
    decay      : olvido pasivo en cada clasificación   (default 0.0005)
    fatigue_k  : cuánto se fatiga una neurona al disparar (default 0.05)
    fatigue_recovery : recuperación de fatiga por paso (default 0.995)
    radius     : fracción del espacio que es plástica  (default 0.15)
    consolidation_k  : peso del envejecimiento en la resistencia al cambio
                       (default 0.05)

    Ejemplo
    ───────
    ban = PlasticBAN(ltp_rate=0.01, ltd_rate=0.005, radius=0.2)
    ban.train_from_("gato.png",  "gato")
    ban.train_from_("perro.png", "perro")

    for img in ["gato2.png", "gato3.png", "gato4.png"]:
        ban.classify_(img)          # W_fwd evoluciona en cada llamada

    ban.plasticity_report()
    ban.save("models/plastic_ban.pkl")
    """

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self,
                 ltp_rate          : float = 0.01,
                 ltd_rate          : float = 0.005,
                 decay             : float = 0.0005,
                 fatigue_k         : float = 0.05,
                 fatigue_recovery  : float = 0.995,
                 radius            : float = 0.15,
                 consolidation_k   : float = 0.05):
        super().__init__()

        # Hiperparámetros de plasticidad
        self.ltp_rate         = ltp_rate
        self.ltd_rate         = ltd_rate
        self.decay            = decay
        self.fatigue_k        = fatigue_k
        self.fatigue_recovery = fatigue_recovery
        self.radius           = radius
        self.consolidation_k  = consolidation_k

        # Estado neuronal  (se inicializa tras el primer _fit)
        self._fatigue    : np.ndarray | None = None   # (P,)   acumulación de uso
        self._synapse_age: np.ndarray | None = None   # (P, D) contador de cambios

        # Historial de eventos plásticos
        self._history: list[dict] = []

    # ── _fit: inicializa estado neuronal la primera vez ──────────────────────

    def _fit(self):
        super()._fit()
        P, D = self.W_fwd.shape
        if self._fatigue is None:
            self._fatigue     = np.zeros(P, dtype=np.float32)
            self._synapse_age = np.ones((P, D), dtype=np.float32)
            print(f"  🧠 Estado neuronal inicializado  "
                  f"[P={P}, D={D}]")

    # ── Máscara de zona plástica local ───────────────────────────────────────

    def _local_mask(self, A: np.ndarray) -> np.ndarray:
        """
        Retorna máscara booleana (P,) con True en las neuronas
        dentro del radio de activación del patrón A.

        Se modela el centro de activación como el centroide
        de los píxeles encendidos, y el radio como fracción
        del eje 1-D normalizado [0, 1].
        """
        active    = A.flatten().astype(bool)
        n         = len(active)
        positions = np.linspace(0.0, 1.0, n, dtype=np.float32)
        center    = float(positions[active].mean()) if active.any() else 0.5
        return np.abs(positions - center) <= self.radius

    # ── Classify con reconsolidación ─────────────────────────────────────────

    def classify_(self, image_input,
                  verbose        : bool = True,
                  reconsolidate  : bool = True) -> tuple[str, dict]:
        """
        Clasifica la imagen y, opcionalmente, aplica plasticidad post-evento.

        Parámetros adicionales vs BAN.classify_()
        ------------------------------------------
        reconsolidate : si True, W_fwd se actualiza tras cada clasificación
                        (reconsolidación hipocampal)
        """
        winner, scores = super().classify_(image_input, verbose)

        if reconsolidate and self._fitted:
            self._apply_plasticity(image_input, winner)

        return winner, scores

    # ── Motor de plasticidad ─────────────────────────────────────────────────

    def _apply_plasticity(self, image_input, winner: str):
        """
        Actualiza W_fwd aplicando:
            1. Olvido pasivo (decay)
            2. LTP  en zona activa  con error positivo
            3. LTD  en zona inactiva dentro del radio
            4. Inhibición por fatiga
            5. Puerta de consolidación (sinapsis viejas resisten)
        """
        if isinstance(image_input, str):
            image_input = INPUT_DIR / image_input

        vec, _ = _preprocess(image_input)
        A      = vec.toarray().flatten().astype(np.float32)   # (P,)

        # Proyección actual y objetivo
        B_hat    = self._forward(vec)                          # (D,) ∈ {-1,+1}
        B_target = self.label_vecs[winner].astype(np.float32)  # (D,)
        error    = B_target - B_hat                            # (D,) ∈ {-2,0,+2}

        # ── Zona plástica local ──────────────────────────────────
        mask     = self._local_mask(A)                         # (P,) bool
        A_masked = (A * mask).astype(np.float32)               # activas en zona

        # ── 1. Olvido pasivo ─────────────────────────────────────
        decay_term = self.W_fwd * self.decay

        # ── 2. LTP — activas + error → refuerzo ──────────────────
        ltp_delta  = np.outer(A_masked, error) * self.ltp_rate

        # ── 3. LTD — inactivas en zona → debilitamiento ──────────
        inactive   = ((1.0 - A) * mask).astype(np.float32)
        ltd_delta  = np.outer(inactive, np.abs(error)) * self.ltd_rate

        # ── 4. Fatiga ────────────────────────────────────────────
        #   Neuronas que dispararon mucho se inhiben
        fatigue_effect            = np.outer(
            self._fatigue * mask, np.ones(self.W_fwd.shape[1], dtype=np.float32)
        ) * self.ltp_rate * 0.5

        self._fatigue            += A_masked * self.fatigue_k
        self._fatigue            *= self.fatigue_recovery     # recuperación lenta

        # ── 5. Puerta de consolidación ────────────────────────────
        #   synapse_age crece con cada cambio → plasticity_gate baja
        plasticity_gate = 1.0 / (1.0 + self._synapse_age * self.consolidation_k)

        # ── Δ W total ─────────────────────────────────────────────
        dW = (ltp_delta - ltd_delta - fatigue_effect - decay_term) * plasticity_gate

        self.W_fwd += dW

        # ── Envejecer sinapsis modificadas ────────────────────────
        changed            = np.abs(dW) > 1e-7
        self._synapse_age += changed.astype(np.float32)

        # ── Registrar evento ──────────────────────────────────────
        self._history.append({
            "winner"         : winner,
            "active_neurons" : int(mask.sum()),
            "ltp_mean"       : float(np.abs(ltp_delta).mean()),
            "ltd_mean"       : float(np.abs(ltd_delta).mean()),
            "fatigue_mean"   : float(self._fatigue.mean()),
            "dW_norm"        : float(np.linalg.norm(dW)),
        })

    # ── Reentrenamiento forzado con plasticidad reforzada ────────────────────

    def reinforce_(self, filename: str, label: str,
                   repetitions: int = 5) -> "PlasticBAN":
        """
        Refuerza un patrón aplicando plasticidad repetida,
        simulando el efecto de repasar/practicar.

        Parámetros
        ----------
        filename    : imagen en ./input/
        label       : etiqueta esperada
        repetitions : cuántas veces se refuerza el patrón
        """
        label = label.strip().lower()
        print(f"\n  🔁 Reforzando '{label}' × {repetitions} …")
        for i in range(repetitions):
            self._apply_plasticity(INPUT_DIR / filename, label)
        print(f"  ✓ Refuerzo completado  |  "
              f"dW_norm_last={self._history[-1]['dW_norm']:.5f}")
        return self

    # ── Snapshot del estado sináptico ────────────────────────────────────────

    def synaptic_snapshot(self) -> dict:
        """
        Retorna un resumen del estado actual de W_fwd y del tejido neuronal.
        Útil para comparar W_fwd antes y después de sesiones de aprendizaje.
        """
        if not self._fitted:
            raise RuntimeError("PlasticBAN sin entrenar.")
        return {
            "W_mean"          : float(self.W_fwd.mean()),
            "W_std"           : float(self.W_fwd.std()),
            "W_max"           : float(self.W_fwd.max()),
            "W_min"           : float(self.W_fwd.min()),
            "W_norm"          : float(np.linalg.norm(self.W_fwd)),
            "fatigue_mean"    : float(self._fatigue.mean()),
            "fatigue_max"     : float(self._fatigue.max()),
            "synapse_age_mean": float(self._synapse_age.mean()),
            "synapse_age_max" : float(self._synapse_age.max()),
            "events"          : len(self._history),
        }

    # ── Reporte de plasticidad ────────────────────────────────────────────────

    def plasticity_report(self):
        """Imprime el estado completo del tejido plástico."""
        print("\n══ PlasticBAN — Reporte de Plasticidad ══════════════")

        # Estado sináptico
        if self._fitted:
            snap = self.synaptic_snapshot()
            print(f"\n  ── Matriz W_fwd  {self.W_fwd.shape} ──")
            print(f"     Media         : {snap['W_mean']:+.5f}")
            print(f"     Desv. estándar: {snap['W_std']:.5f}")
            print(f"     Rango         : [{snap['W_min']:+.4f}, {snap['W_max']:+.4f}]")
            print(f"     Norma Frobenius: {snap['W_norm']:.4f}")

        # Estado neuronal
        if self._fatigue is not None:
            young    = int((self._synapse_age <= 2).sum())
            mature   = int(((self._synapse_age > 2) & (self._synapse_age <= 10)).sum())
            old      = int((self._synapse_age > 10).sum())
            total_syn = self._synapse_age.size
            print(f"\n  ── Tejido neuronal ──")
            print(f"     Sinapsis jóvenes   (edad ≤ 2)  : {young:>8}  "
                  f"({100*young/total_syn:.1f}%)")
            print(f"     Sinapsis maduras   (2 < edad ≤ 10): {mature:>8}  "
                  f"({100*mature/total_syn:.1f}%)")
            print(f"     Sinapsis consolidadas (edad > 10): {old:>8}  "
                  f"({100*old/total_syn:.1f}%)")
            print(f"     Fatiga media actual : {self._fatigue.mean():.5f}")
            print(f"     Fatiga máxima       : {self._fatigue.max():.5f}")

        # Historial de eventos
        n = len(self._history)
        print(f"\n  ── Historial plástico ({n} eventos) ──")
        if n == 0:
            print("     Sin eventos aún.")
        else:
            ltp_vals = [e["ltp_mean"] for e in self._history]
            ltd_vals = [e["ltd_mean"] for e in self._history]
            dw_vals  = [e["dW_norm"] for e in self._history]
            winners  = {}
            for e in self._history:
                winners[e["winner"]] = winners.get(e["winner"], 0) + 1

            print(f"     LTP medio  : {np.mean(ltp_vals):.6f}")
            print(f"     LTD medio  : {np.mean(ltd_vals):.6f}")
            print(f"     ‖ΔW‖ medio : {np.mean(dw_vals):.6f}")
            print(f"     ‖ΔW‖ último: {dw_vals[-1]:.6f}")
            print(f"     Activaciones por etiqueta:")
            for lbl, cnt in sorted(winners.items(), key=lambda x: -x[1]):
                print(f"       {lbl:<16} {cnt:>5} veces")

        print("══════════════════════════════════════════════════════\n")

    # ── Resumen heredado enriquecido ──────────────────────────────────────────

    def summary(self):
        super().summary()
        print("── PlasticBAN extras ────────────────────────────────")
        print(f"   LTP rate       : {self.ltp_rate}")
        print(f"   LTD rate       : {self.ltd_rate}")
        print(f"   Decay          : {self.decay}")
        print(f"   Fatigue k      : {self.fatigue_k}")
        print(f"   Radius         : {self.radius}")
        print(f"   Consolidation k: {self.consolidation_k}")
        print(f"   Eventos plást. : {len(self._history)}")
        print("─────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# Demo 
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ban = PlasticBAN(
        ltp_rate        = 0.01,
        ltd_rate        = 0.005,
        decay           = 0.0005,
        fatigue_k       = 0.05,
        fatigue_recovery= 0.995,
        radius          = 0.20,
        consolidation_k = 0.05,
    )

    # ── Entrenamiento ────────────────────────────────────────────
    # (requiere imágenes en ./input/)
    # ban.train_from_("gato.png",  "gato")
    # ban.train_from_("perro.png", "perro")
    # ban.train_from_("auto.png",  "auto")

    # ── Clasificación con plasticidad activa ─────────────────────
    # ban.classify_("gato2.png")
    # ban.classify_("gato3.png")

    # ── Reforzar un patrón específico ────────────────────────────
    # ban.reinforce_("gato.png", "gato", repetitions=10)

    # ── Reportes ─────────────────────────────────────────────────
    # ban.summary()
    # ban.plasticity_report()
    # print(ban.synaptic_snapshot())

