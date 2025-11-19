TODO
====

- [x] Instanciar dos brazos idénticos en el entorno y colocarlos en `x = +0.1` y `x = -0.1`, comprobando que ambos funcionan en batch.
- [ ] Introducir una única pieza (un bloque componente del robot) que aparezca en la zona negativa del eje Y y definir la acción de moverla a `+Y`.
- [ ] Implementar el flujo básico: el brazo del lado negativo agarra la pieza, la desplaza y la libera en la posición objetivo en `+Y`.
- [ ] Extender el reset para soportar varias piezas secuenciales, moviéndolas de `-Y` a `+Y` una a una.
- [ ] Modelar la impresora 3D bajo la posición inicial de las piezas para que parezcan salir de ella.
- [ ] Añadir una mesa física donde se apoyen la impresora y las posiciones de destino, ajustando colisiones y alturas para el flujo “impresora → mesa”.
