## TODO – mejoras pendientes

- **Pipeline de evaluación ausente (`train.py` + CLI)**
   - No hay un comando/script que cargue el último checkpoint y ejecute episodios de evaluación; tampoco existe una flag estilo `--load-latest-model`.
   - Acción: añadir una función de `eval` que use `log.get_latest_model`, exponerla mediante CLI y permitir que `train.py` acepte una opción (`--load-latest-model`) para continuar entrenando desde el último run.
