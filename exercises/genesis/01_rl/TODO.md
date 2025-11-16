## TODO – mejoras pendientes

- **Episodios infinitos (`env.py`)**
   - `max_steps` nunca se usa, y `step` ignora el indicador `done` a pesar de que la docstring lo promete.
   - Acción: implementar condiciones de finalización (éxito por distancia al target y límite de pasos), devolver `done` desde `step` y resetear los entornos cuando corresponda. Propagar `done` al buffer para que el cálculo del retorno sea correcto.

- **Estado `previous_actions` nunca se actualiza (`env.py`)**
   - Se asigna sólo si la acción nueva es numéricamente idéntica, lo cual rara vez ocurre; la variable queda congelada desde el `reset`.
   - Acción: asignar siempre la nueva acción o invertir la lógica (`if not allclose`) para que la detección de acciones repetidas funcione.

- **Penalización de velocidad angular incorrecta (`env.py`)**
   - El término `gripper_angular_velocity` usa `get_ang()`, es decir, la orientación, no la velocidad angular.
   - Acción: usar la API de velocidad angular (`get_angvel()` o equivalente) para que la penalización corresponda a la descripción.

- **Selección de checkpoint frágil (`log.py`)**
   - `get_latest_model` hace `max(os.listdir(...))` sin filtrar directorios ni ordenar por fecha; cualquier archivo “grande” rompe la carga.
   - Acción: filtrar sólo carpetas de run y ordenarlas por timestamp (nombre o `mtime`) antes de elegir la última.

- **Pipeline de evaluación ausente (`train.py` + CLI)**
   - No hay un comando/script que cargue el último checkpoint y ejecute episodios de evaluación; tampoco existe una flag estilo `--load-latest-model`.
   - Acción: añadir una función de `eval` que use `log.get_latest_model`, exponerla mediante CLI y permitir que `train.py` acepte una opción (`--load-latest-model`) para continuar entrenando desde el último run.
