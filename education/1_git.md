# Git

Si programamos en equipo vamos a necesitar estar familiarizados con Git.

Podemos usar este repositorio para ir practicando.


## Recursos

- [Video corto](https://www.youtube.com/watch?v=hwP7WQkmECE&t=9s)
- [Contextualización desde un punto de vista más práctico de andar por casa](https://www.youtube.com/watch?v=mJ-qvsxPHpY)
- [Video](https://www.youtube.com/watch?v=HMoZ5cYzU4I)
- [Video](https://www.youtube.com/watch?v=-iWaarLI7zI)


## Cheat sheet

- Clonar un repositorio:

```sh
git clone url-del-repositorio
```

- Ver el estado de los ficheros:

```sh
git status
```

- Crear nueva rama (para modificar sin tocar el modelo main):

```sh
git checkout -b nombre_rama (sin '-b' para moverse entre ramas o volver al main)
```

- Para ver el grafo de commits:

```sh
git log --graph --all
```

- Para hacer un commit:

```sh
git add ruta-del-fichero      # Añadir fichero al commit (puede ser . para todos)
git commit -m "implement some feature"
```

- Para subir los cambios al repositorio remoto:

```sh
git push
```

- Para bajar los cambios del repositorio remoto:

```sh
git pull
```


