## Move Pieces (modelo 02) – plan de refactor

### Entorno y escena
- Mantener la configuración actual de los dos Lerobot SO101 (GL a la izquierda orientado hacia el compañero y GR a la derecha invertido 180° en Z) sin modificaciones adicionales.
- Añadir una cama de impresión física (box) centrada en `y = -0.30 m` con dimensiones aproximadas `0.70 × 0.45 × 0.02 m`, apoyada sobre el suelo y fácilmente identificable en la visualización.
- Instanciar todas las piezas reales del SO101 usando sus meshes originales (`*.stl` en `exercises/assets/SO101/assets/`). Nada de colisiones simplificadas: se usa la malla completa para colisión y rendering.
- Para cada pieza, definir una pose inicial sobre la cama de impresión (NI) y una pose objetivo en racks o zonas a los lados de la escena (NT). Guardar estas poses para todo el batch y reposicionar las piezas en cada `reset`.
- Añadir utilidades para apilar estados de piezas (posición, rotación, velocidades) y poder mostrarlos en debug (esferas NI/NT) si se usa el viewer o la grabación.

### Observaciones
- Seguir el formato típico del SO101: joint positions/velocities de ambos brazos, pose (pos + rot vec) y velocidades del `gripper_tip` de cada robot.
- Mantener la restricción de no usar información exclusiva de la simulación dentro del espacio de observaciones (nada de flags mágicos, fuerzas internas, etc.); sólo datos que puedan obtenerse en el robot real a partir de encoders/estimadores/sensores.
- Para avanzar mientras se integra el sistema de visión, se incluye temporalmente la pose “perfecta” de cada pieza y de su target. Esta parte se marcará explícitamente para sustituirla por la salida del detector real en cuanto esté disponible.

### Diseño de recompensas
- Recompensas principales:
  - `piecewise_distance`: error de posición pieza a pieza (`‖p_i - p_i^T‖`) con peso negativo `-w_pos`.
  - `piecewise_rot_distance`: error de orientación calculado como la norma del rotation vector entre la orientación actual y la objetivo. Se calcula mediante la delta de cuaterniones (`q_T * conj(q_actual)`) y se convierte a rotvec, cuidando el signo. Recompensa `-w_rot * piecewise_rot_distance`.
- Recompensas secundarias por fases:
  1. **Fase 1 (aproximación del brazo):** activa mientras `min(dist(GL, NI), dist(GR, NI)) > d_pick`. Recompensa `-w_phase1 * min(dist_GL, dist_GR)` y, si ambos están muy lejos (>0.15 m), se refuerza que el más cercano se acerque.
  2. **Fase 2 (transporte):** se activa cuando un brazo está lo bastante cerca de la pieza (`≤ d_pick`) pero la pieza aún está lejos del target (`‖NI - NT‖ > d_place`). Recompensa `-w_phase2 * ‖pose_piece - pose_target‖`.
  3. **Fase 3 (alineación fina):** si `‖pose_piece - pose_target‖ ≤ d_place`, combinamos `-w_phase3_pos * dist` y `-w_phase3_rot * rot_error` para asegurar que no se desplace del target mientras ajusta la orientación.
- Priorización de piezas: se procesa en orden fijo (o mediante `self.active_piece_idx`) para evitar ambigüedades. Al completar una pieza (DIST y ROT bajo umbral) se da un bonus y se pasa a la siguiente.
- Términos adicionales opcionales: penalizaciones suaves por fuerzas y velocidades extremas reutilizando la lógica existente.

### Recompensas/fases (pendiente de implementar)
- Recompensas principales `piecewise_distance` y `piecewise_rot_distance` que miden el error pose a pose entre cada pieza y su target.
- Recompensas secundarias condicionadas por fases:
  1. **Fase 1:** acercar GL/GR a la pieza más cercana (minimizar `min(dist(GL, NI), dist(GR, NI))`).
  2. **Fase 2:** transportar la pieza hacia su `NT` mientras se mantiene agarre (minimizar `dist(NI, NT)`).
  3. **Fase 3:** ajustar orientación y posición fina (combinar `dist` y `rot_dist`).
- Las fases se activarán por umbrales como en el esquema entregado (por ejemplo, pasar a Fase 2 cuando el brazo está a <0.02 m de NI, etc.). Implementaremos estos cálculos tras cerrar el rediseño del entorno.

Este documento sirve como guía para los cambios estructurales del entorno y como recordatorio de la siguiente tarea: implementar las recompensas por fases descritas arriba.
