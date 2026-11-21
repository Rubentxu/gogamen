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

## Experimentos `CognitiveSpec` desde Python (GOG-194 / DEM-041)

`gogamen.experiment_spec` construye un experimento con builders (`system`, `node`, `cognitive`, `agent`, `world`, `ExperimentSpec.build`, ...), emite el **JSON canónico** y calcula el **hash** con las mismas reglas que Rust (`gogamen-spec::json`) y el DSL Kotlin: claves ordenadas por bytes UTF-8, `f64` en notación `Display` de Rust (`1.0`, sin exponente), nodos de expresión con `op` primero, y FNV-1a 64 sobre el JSON canónico en hex de 16 dígitos. Python no valida el dominio ni define un IR nuevo: pasa los `f64` como `float` y deja la validación al parser Rust.

```python
from gogamen import ExperimentSpec
spec = ExperimentSpec.from_json(open("spec/golden/cognitive_dsl/positive/h0.json").read())
spec.hash()                        # '46d838f79d6de6df'
spec.validate_native(timeout=60)   # parser Rust vía CLI; falla si el hash nativo difiere
```

- **Validación nativa:** usa la frontera existente `gogamen run-experiment --demo grid-transfer --dsl-file <spec>`. No hay un subcomando solo-validar, así que el CLI también ejecuta su demo fija (unos segundos), no derivada de la spec. Los rechazos llegan como `GogamenSpecError` con `path` (archivo enviado), `context`/`field` del parser y `pointer` (JSON pointer) cuando el campo es de primer nivel (p. ej. `/seeds`); los errores de canonicalización en Python llevan su JSON pointer exacto.
- **Barrido:** `gogamen.sweep.run_sweep(timeout=...)` ejecuta el ejemplo Rust de GOG-185/DEM-028 `sweep_cartpole` (`cargo build -p gogamen-arena --example sweep_cartpole`; ruta vía `GOGAMEN_SWEEP_BIN` o `CARGO_TARGET_DIR`), devuelve su CSV tal cual (`csv_text`, `rows`) y verifica el `csv_hash` que declara Rust. **Limitación:** el CLI `gogamen` no tiene subcomando de barrido y los ejes del ejemplo son fijos; todavía no se parametriza el barrido desde la spec.
- **Timeouts:** toda llamada a subproceso tiene timeout explícito y lanza `GogamenTimeoutError`.
- Los números describen únicamente esas ejecuciones; no se afirma ningún fenómeno emergente.

Test (desde la raíz del repo; el invocador `python3 -m unittest python.tests.test_experiment_spec` solo funciona con `PYTHONPATH=python`):

```bash
cargo build -p gogamen-cli -p gogamen-arena --example sweep_cartpole   # opcional: sin binarios, los tests nativos se saltan
PYTHONPATH=python python3 -m unittest python.tests.test_experiment_spec -v
```

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
