# Gogamen Kotlin DSL (`gogamen-dsl`)

Frontend declarativo mínimo en Kotlin/JVM para construir especificaciones `SystemSpec` (`gogamen.system_spec.v0`) en JSON canónico, completamente desacoplado del runtime Rust (sin JNI, sin callbacks en runtime y el núcleo no ejecuta scripts; el scripting es opcional, ver al final).

## Entorno validado

- **JDK:** Temurin `21.0.8+9.0.LTS` (configurado en `.tool-versions` para `asdf`)
- **Gradle:** `8.14.5`
- **Kotlin:** `2.4.10`

## Ejecución de pruebas

Desde `kotlin/`:

```bash
gradle test --offline
```

Las pruebas verifican:
1. Generación de JSON canónico equivalente a `canonical_two_node_fixture` y al `SystemSpec` de `canonical_grid_experiment_fixture` de `gogamen-spec`, además del uso de `@GogamenDsl` (`@DslMarker`).
2. Oráculo contra el validador Rust (`cargo run -q -p gogamen-cli --offline -- run-spec --json-file ...`) comprobando ejecución válida y propagación de errores de nombres de nodos/puertos y capacidades/operaciones inexistentes con su ruta de origen.

## Scripting opcional (spike GOG-83)

Fuente opcional `src/scripting` (`dev.gogamen.dsl.script`): scripts de confianza `*.gogamen.kts` que evalúan a `SystemSpec`. No forma parte del jar principal. Los scripts son código JVM arbitrario; los datos no confiables (JSON) nunca deben pasar por el evaluador. Ver `docs/scripting-threat-model.md` y `docs/adr-0001-scripting-vs-library.md`.

```bash
gradle test          # incluye scriptingTest (tests de scripting, diagnósticos y lint)
gradle scriptingTest
```
