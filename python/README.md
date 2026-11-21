# Gogamen Python SDK (`stdlib`-only)

SDK Python ligero y sin dependencias externas que se comunica con el binario Rust `gogamen` mediante un protocolo residente **JSON-Lines** sobre `stdin`/`stdout` (`gogamen serve-protocol`).

## Arquitectura de frontera estable

- **Estado residente en Rust:** `GridWorldEnv` y sus sesiones viven dentro del proceso `gogamen serve-protocol`.
- **Sin duplicación de dominio:** Python no reimplementa la física ni el mapa; envía acciones discretas y recibe únicamente `delta`, observación ligera, recompensa, estado terminal y eventos.
- **Sin dependencias externas:** el paquete `python/gogamen` usa exclusivamente la biblioteca estándar de Python (`>= 3.10`) y el core Rust permanece `std`-only offline sin `pyo3`.

## Uso rápido sin instalación (`PYTHONPATH`)

1. Compila el binario CLI en modo offline:

```bash
cargo build -p gogamen-cli --offline
```

2. Ejecuta el ejemplo de Grid:

```bash
PYTHONPATH=python python3 -m gogamen.examples.run_grid
```

3. Ejecuta la suite de tests:

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

Si el binario `gogamen` está en una ruta personalizada, puedes indicarlo con la variable de entorno `GOGAMEN_BIN=/ruta/a/gogamen`.

## Build opcional y distribución

El directorio `python/` incluye un `pyproject.toml` estándar PEP 621 con `dependencies = []`:

- **Instalación editable local (cuando hay `pip`/`setuptools` disponible):**
  ```bash
  pip install --no-build-isolation --no-deps -e ./python
  ```
- **Construcción de wheel / sdist:**
  ```bash
  python3 -m build ./python
  ```
- **Distribución:** al empaquetar o desplegar, incluye el binario compilado `gogamen` en `PATH` o configura `GOGAMEN_BIN`.
