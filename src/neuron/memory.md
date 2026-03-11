Paso 1 — Preprocesar la imagen cruda
pythonvec_img, _ = _preprocess(ruta)
```
```
carro.png  →  vector sparse (1, 38416)
               píxeles binarizados {0, 1}

Paso 2 — Propagar por la cadena
pythonrepresentation = vec_img
for ban in chain:
    representation = sp.csr_matrix(
        ban._forward(representation).reshape(1, -1)
    )
```

Cada iteración transforma la representación actual:
```
vec_img  (1, 38416)
    │
    ▼  BAN1._forward()
    │  → A.toarray().flatten() → (38416,)
    │  → (38416,) @ W_fwd(38416, 352) → (352,)
    │  → sign() → {-1,+1}  (352,)
    │  → reshape(1,-1) → (1, 352)
    │  → csr_matrix
    │
    ▼  BAN2._forward()
    │  → (1,352) → flatten → (352,)
    │  → (352,) @ W_fwd(352, 352) → (352,)
    │  → sign() → {-1,+1} (352,)
    │  → reshape(1,-1) → (1, 352)
    │  → csr_matrix
    │
    ▼
vec_A  (1, 352)  ← representación final

Paso 3 — Registrar el label
pythonif label not in self.label_vecs:
    idx = len(self.labels)
    self.labels.append(label)
    self.label_vecs[label] = _encode_label(idx)

vec_B = self.label_vecs[label]
```
```
"a car is"  →  si no existe, genera vector bipolar (352,) con seed fijo
               si ya existe, reutiliza el mismo vector
               → vec_B (352,)

Paso 4 — Acumular el par
pythonself._A_rows.append(vec_A)   # (1, 352) sparse
self._B_rows.append(vec_B)   # (352,) bipolar
```

Cada llamada agrega una fila nueva a la memoria:
```
_A_rows                    _B_rows
┌──────────────┐           ┌──────────────┐
│ firma img1   │           │ vec "a car"  │
│ firma img2   │           │ vec "a car"  │
│ firma img3   │           │ vec "a car is│
└──────────────┘           └──────────────┘

Paso 5 — Reajustar W_fwd
pythonself._fit()
pythonA_dense = sp.vstack(_A_rows).toarray()   # (N, 352)
B_dense = np.stack(_B_rows)              # (N, 352)

W_fwd = pinv(A_dense) @ B_dense          # (352, 352)
```

Resuelve en mínimos cuadrados:
```
"dado que la cadena produce ESTA firma
 para esta imagen, aprender a producir
 ESTE vector de label"
```

---

### Qué queda almacenado al final
```
W_fwd   (352, 352)   ← única matriz que sobrevive
label_vecs           ← diccionario label → vector
labels               ← lista de labels conocidos
_A_rows              ← firmas acumuladas (para reentrenar)
_B_rows              ← labels acumulados (para reentrenar)
```

---

### Resumen en una línea por paso
```
1. imagen   →  píxeles crudos (38416)
2. píxeles  →  firma de la cadena (352)   ← lo que cambia vs train_from_
3. label    →  vector bipolar fijo (352)
4. acumular par (firma, label)
5. W_fwd = pinv(firmas) @ labels          ← aprende la asociación
La diferencia clave con train_from_ es el paso 2 — en lugar de usar los píxeles directamente como entrada, primero los transforma en la firma que la cadena produciría, y eso es lo que BAN aprende a asociar con el label.