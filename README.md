## Gogamen Workspace — Smoke local y CI

Requisitos: toolchain Rust fijada en [`rust-toolchain.toml`](rust-toolchain.toml) (`1.99.0` con `rustfmt` y `clippy`).

Comando único para verificar formato, lints y tests offline del workspace:

```sh
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace --offline
```

Para ejecutar la CLI mínima de verificación:

```sh
cargo run -p gogamen-cli --offline -- --version
```

### Primer recibo de experimento (Kart puro vs azar vs techo dirigido)

Sin red, sin JVM y sin JS externo. Un comando ejecuta el experimento y escribe `receipt.json` (especificación y hash, hipótesis prerregistradas y su hash, semillas, versión, libro de dotación, resultados por episodio, veredictos) y `report.html` (autónomo: libro de dotación, veredictos, curvas media ± desviación por semilla, episodios fallidos). Otro comando lo verifica re-ejecutando y comparando hashes; falla con la ruta de la primera diferencia si el recibo fue manipulado o no reproduce:

```sh
cargo run -p gogamen-cli --release --offline -- run-experiment --out-dir out/kart
cargo run -p gogamen-cli --release --offline -- verify-receipt --receipt-file out/kart/receipt.json
```

Código: `crates/gogamen-cli/src/receipt.rs` y `report.rs`; prueba decisiva: `cargo test -p gogamen-cli --offline receipt_roundtrip`. Un veredicto sin hash prerregistrado nunca es `SUPPORTED` (`--no-prereg` lo demuestra).

Tiempo medido (máquina de desarrollo de 9 hilos, Linux, caché de crates ya presente, `--offline`): build en frío de `gogamen-cli --release` con directorio target vacío = 21 s; `run-experiment` = 1,7 s; `verify-receipt` = 1,6 s. Total medido hasta el primer recibo: unos 23 s. NO medido: `git clone`, descarga de crates con red, instalación del toolchain `1.99.0` ni máquinas distintas; no extrapolar. Los 64 bits FNV-1a detectan ediciones y no-reproducibilidad, no son firmas criptográficas.

Demo de transferencia offline (GOG-178): `cargo run -p gogamen-cli --release --offline -- run-experiment --demo grid-transfer --out-dir out/demo` toma la spec cognitiva emitida por el DSL Kotlin (`spec/golden/cognitive_dsl/positive/h0.json`, o `--dsl-file`), la parsea y hashea en Rust, entrena un modelo tabular solo en el mapa Grid A, lo usa con el planificador FMC en el mapa B no visto (semillas holdout) y lo compara con baselines (aleatorio, prior sin entrenar, greedy tabular); el techo privilegiado se informa aparte. Emite `receipt.json` (spec hash, semillas, versiones/checksums del modelo, resultados, trazas por decisión, límites, veredicto) y `report.html`; `verify-receipt --receipt-file out/demo/receipt.json` lo re-ejecuta. Sin UI, LLM, JVM ni red en runtime. El veredicto puede ser negativo y se informa tal cual (en la corrida de referencia: TRANSFER_NOT_SUPPORTED, 0/6 frente a 0/6 del baseline aleatorio). Límites: un solo par de mapas y 6 semillas; el mundo/semillas de la demo aún no se derivan de la spec; variante Kart no incluida.

Release MVP pre-1.0 (local, sin publicación automática): ver [`docs/RELEASE.md`](docs/RELEASE.md) y `scripts/release_check.sh`.

---

# Gogamen — Semilla de Backlog.md v3

> **Gogamen** (del euskera *gogamen*: mente, intelecto y facultad cognitiva en acción) es un framework de computación cognitiva, adaptativa y funcional de propósito general para inteligencias situadas.  
> Concebido en la propuesta inicial bajo el nombre provisional de *Noesis*.
>
> * **Repositorio**: [Rubentxu/gogamen](https://github.com/Rubentxu/gogamen)
> * **Licencia**: [MIT License](LICENSE)
> * **Diseño original y exploración del DSL**: [`ChatGPT_Diseñar-lenguaje-DSL_20261120-1105.md`](ChatGPT_Diseñar-lenguaje-DSL_20261120-1105.md)

---

Paquete de inicialización para un proyecto nuevo, no repositorio ejecutable del motor. Fecha de propuesta: 2026-11-20. **117 tareas** con 351 AC, DoD explícito, dependencias técnicas, 6 documentos nativos a importar y 13 decisiones `proposed`. Nada se declara implementado ni Done. Sustituye la semilla v2: conserva sus 92 alias de diseño, refina 6 tarjetas y añade 25. **Solo para un tablero virgen o una importación parcial de v3**. Si se importó v2, usar la conciliación descrita abajo (para v1 se requiere comparar la semilla v1 con v3; el delta incluido solo cubre v2): no ejecutar la semilla sobre trabajo existente sin revisar. Los alias NOE-xxx son claves del paquete, NO IDs Backlog reales. Tras importar, Backlog.md pasa a ser la única autoridad; eliminar o guardar este ZIP fuera del repositorio de código, no mantenerlo sincronizado.

## Inicio recomendado

1. Elegir explícitamente `noesis-home` privado como raíz Backlog (o XDG privado si no se crea Home Repo); conectar `noesis` como repo de código. Nombres sugeridos, no creados ni reservados. Si ya existe binding software-knowledge, localizarlo con `locate-home` y usar `initialize-home` sólo si está autorizado. No crear dos tableros. No usar un repositorio público para el Home.
2. Comprobar CLI: `backlog --version`, `backlog instructions overview`, `backlog instructions task-creation`. Si no existe, instalar **solo con aprobación** desde [Backlog.md](https://github.com/MrLesk/Backlog.md). Inicializar en raíz privada elegida con `backlog init` después de leer ayuda y responder a configuración. Se recomienda prefijo `NOE` o el que el usuario elija, estado inicial de configuración y Git privado. No asumir prefijo ni estados. No copiar archivos directamente al directorio gestionado.
3. Revisar configuración antes de mutar: `backlog config list`, `backlog config get defaultStatus`, `backlog config get priorities`, `backlog config get autoCommit`, `backlog config get remoteOperations`, `backlog config get bypassGitHooks`, `backlog config get onStatusChange`, `backlog config get checkActiveBranches`. Confirmar que high/medium/low son reconocidos por CLI, que efectos Git/callbacks están autorizados; sugerir autoCommit=false, remoteOperations=false, bypassGitHooks=false y ningún callback, pero **no cambiar sin permiso**. Revisar `backlog task list --json`, `backlog search NOE --json`, `backlog doc list --plain`, `backlog decision list --json`; el importador es para tablero nuevo o importación parcial. No ejecutar en paralelo.
4. Extraer ZIP **fuera del proyecto de código**. Inspeccionar contenido, ejecutar sin mutaciones:

```sh
python3 import_backlog.py
python3 test_seed.py
```

5. Con raíz inicializada y autorización de importación:

```sh
python3 import_backlog.py --home /ruta/noesis-home --apply --confirm-config
```

`--section tasks|docs|decisions` permite resolver fases por separado. El importador usa Python estándar y CLI, no instala, no inicializa, no hace commits ni pushes por cuenta propia. Las opciones nativas de Backlog pueden tener efectos: revisión previa obligatoria. Se basa en CLI upstream vigente consultada; **no se ejecutó contra una instalación real** (binario no disponible). Ante un error para y conserva lo importado. No repetir a ciegas: comprobar lista y `view`, comparar el contenido con seed, reparar únicamente por CLI. Las tareas ya existentes con marcador se conservan (preflight de título, dependencias, descripción, AC y presencia del DoD; conserva checks y notas). Documentos y decisiones existentes se saltan, incluyendo un doc vacío si se interrumpió entre create/update: recuperar su ID real y completar manualmente. No aceptar decisiones automáticamente. Metadata y cuerpos solo vía CLI. `--confirm-config` es declaración del operador, no una verificación automatizada de permisos/configuración.

Si la CLI cambia: construir cada tarea con `backlog task create "[NOE-xxx] título" -d "descripción" --ac "criterio" --dod "DoD" --dep <IDs observados>`; usar `task list/view --json` para mapear alias; no asignar IDs propios. Para documento `backlog doc create "[NOESIS-DOC-xx] título" -p ruta -t specification` y con ID real `backlog doc update doc-N --content "$(cat seed/docs/archivo.md)" --tags "..."`. Para decisión `backlog decision create "[NOESIS-Dxx] título" -s proposed --context ... --decision ... --alternatives ... --consequences ...`. Consultar ayuda instalada. Completar documentos en `seed/documents.json`, incluyendo sus tags y rutas; glosario consolidado inicial, refinar por bounded context si crecimiento justifica.

6. Al finalizar: `backlog task list --json`, `backlog task list --ready --json`, `backlog decision list --json`, `backlog doc list --plain`; corroborar todos los IDs y enlaces. `backlog doctor` solo diagnóstico opcional. No marcar Done, no iniciar workers sin readiness, no archivar nada. Si se solicita commit, usar las políticas del Home Repo y registrar referencias reales, no incluir el paquete de importación en producto.

## Mapa de capacidades (referencia de importación, no roadmap paralelo)

| Etiqueta | Objetivo | Entrega |
|---|---|---|
| M0 | Contratos, spec y runtime general | Sistema reactivo, primer mundo reproducible y Studio Grid |
| M1 | Núcleo científico y motivación | Grid, FMC anytime, péndulo y timeline/replay |
| M2 | Cognición, memoria, aprendizaje y automatización | Language World y aplicación general offline, Jev opt-in |
| M3 | Mundos interactivos y perfiles temporales | Kart, Life, Arena y control multirritmo |
| M4 | Entornos externos | ROM libre, Mario legal, MuJoCo y robot simulado |
| M5 | SDK y autoría compilada | API Rust, conformance, Kotlin DSL, plan portable; Python optativo |
| M6 | MVP release | Distribución local y demostradores; no depende de ROM propietaria, hardware ni Jev live |
| FUT | Investigación y especialización | POMCP, empowerment, HIL, Linux RT, WASM y embebido por evidencia |

Etiquetas no son gates: una tarea puede iniciarse cuando sus dependencias técnicas estén resueltas. Repriorizar desde Backlog CLI; MVP empieza por NOE-001 (workspace), seguido por los contratos generales y adaptativos según dependencias. NOE-063–070 introducen el sustrato general y compilación; NOE-071–078 las capacidades adaptativas compartidas; NOE-080–084 los contratos temporales; NOE-085–089 la autoría y conformance. NOE-093–117 incorporan el SDK de proyecciones, shell, controles, trazas, inspección, Arena, vistas por dominio y distribución visual. Las notas de tamaño son hipótesis, revisar readiness antes de delegar. Validación de benchmarks desde M1; optimizar sólo medición posterior. No exigir fases enteras para integrar una capacidad.

## Delegación y /goal inicial

Usar las skills existentes, no reimplementar su scheduler. Supervisor Elephant: seleccionar siguiente incremento ready, validar anclas, políticas, scope, dependencia real y oráculos. Un escritor por recurso y una tarea por delegación; worker contexto fresco, brief mínimo con IDs reales, SHA observado, criterios, límites de archivos y plan de pruebas. Una tarea no se hace Done por elapsed time ni por commit. Las categorías son estimaciones iniciales, recalcular al observar código: standard ≤30 min, endurance ≤50 min, oversized >60 dividir por valor o resolver incertidumbre. En `delegated-task-loop` nuevos runs `relayPolicy=default`, `maxAttempts=2`, checkpoint 50 min, reconciliación 60, evaluar a 45. Tras 2 intentos estériles o 20–30 min sin avance activar flow-recovery, antes si señales; no abrir nuevo escritor sin parada confirmada y baton validado. Límite acumulado 120 min desde claim, no resetear. Si falta host fresh/cancelación, NOT_RUN o límite cooperativo; nada de procesos externos de agentes. Worker devuelve READY_FOR_REVIEW, PARTIAL o PENDING; supervisor verifica oráculo decisivo proporcional sin repetir toda suite y cierra por CLI con evidencia por AC/DoD. Goldfish si riesgo material y soporte real; no otra fuente de seguimiento. Guardar conocimiento duradero en Home privado, no transcripts. No se exige WIP administrativo; foco 1 por responsable, sugerencia 3 globales.

Prompt para iniciar (después de importar y autorizar alcance):

```
/goal Entrega los primeros slices ready de M0 y M1 de Noesis con el backlog privado como única autoridad. Usa backlog-agile-workflow y delegated-task-loop: elige slices ready, un worker fresco por tarea, relevo por presupuesto y evidencia proporcional. Ejecuta laboratorios headless reproducibles y contrasta FMC con las referencias; no avances afirmaciones ni marques Done sin AC/DoD observados. Mantén simulación aislada de actuación, core rico reutilizable y offline, sin dependencias obligatorias UI/IA. Spec y compilación tempranas, Kotlin solo autoría. Construye Studio Grid pronto mediante Projection SDK, sin acoplar el motor al visor; separa trace/checkpoint de frames visuales. No trasladar algoritmos core a extensiones. Detente al completar los incrementos, al agotar presupuesto autorizado o ante bloqueo real; registra el estado y siguiente tarea ready en Backlog.md.
```

## Procedencia y límites de la preparación

Skills leídas desde GitHub `Rubentxu/agent-skills` main: [backlog-agile-workflow](https://github.com/Rubentxu/agent-skills/blob/main/skills/backlog-agile-workflow/SKILL.md) blob `8688e78568539c777d36b01343baabd179f03e40`; [delegated-task-loop](https://github.com/Rubentxu/agent-skills/blob/main/skills/delegated-task-loop/SKILL.md) blob `53530e6be2749f5b6f0dbf15eea1f88ed4e592d0`; `references/endurance-relay.md` blob `e4edf9cc087ec5d0c5e546ff4d1e0f6dca015232`; `references/lenses/board-create.md` blob `831d7dd6d0365db09e6d52d597baceb9c59701b3`; `references/cli-reference.md` blob `fb3acb17a9c0ff17308824fa8d1794962ca9efac`; `references/knowledge-contract.md` blob `83e3d0bd505f9af937e1e268c6b47095700a41c6`. Documentación upstream [CLI-INSTRUCTIONS](https://github.com/MrLesk/Backlog.md/blob/main/CLI-INSTRUCTIONS.md). No se han ejecutado algoritmos ni laboratorios, ni creado repos remotos. Validación local de estructura, dependencias y construcción de argumentos solamente; mock no certifica CLI real. Blog original de Sergio no accesible en consultas anteriores (no revalidado en v3); fuentes científicas en `seed/docs/science.md`. No suponer aceptación humana de las decisiones propuestas. El ZIP no contiene ROMs, credenciales ni binarios.

### Revisión v2
Se conservan las referencias científicas v1. Se consultaron además [MuJoCo modelo/datos](https://mujoco.readthedocs.io/en/stable/overview.html), [Kotlin builders](https://kotlinlang.org/docs/type-safe-builders.html), [ROS 2 tiempo real](https://design.ros2.org/articles/realtime_background.html). La validación del paquete es estructural y con CLI simulada; ningún runtime, benchmark ni sistema físico se ha ejecutado.

## Si ya se importó v2: conciliación sin sobrescribir

`seed/v2-reconciliation.json` contiene las seis tarjetas con sus valores antes/después, 25 claves nuevas y los documentos/decisiones nuevos o modificados. Es material de importación efímero, no el tablero. El importador **no migra automáticamente** v2 y falla ante divergencias.

1. En el Home real leer `backlog task list --json` y `backlog task view <ID> --json`; mapear alias por marcador exacto `[NOE-xxx]`, revisar duplicados y archivados. Separar las tarjetas con actividad, AC marcados o planes de las intactas. No asumir que un alias es el ID.
2. Para nuevas tareas recorrer `seed/tasks.json` en orden topológico; crear con la CLI usando `argv_for` como referencia, resolver cada dependencia al ID existente o recién creado. Si una dependencia apunta a una tarjeta que requiere refinamiento, registrar sus cambios en el mismo ciclo; no duplicar tarjeta.
3. Para las seis modificadas, comparar `before`, `after` y el estado observado (merge de tres vías humano/agente). Si ya hay evidencia en una tarea Done, mantener historial y capturar sólo el trabajo residual como nuevo slice si procede. No reabrir ni desmarcar evidencia automáticamente. Aplicar sólo cambios autorizados via `backlog task edit` según ayuda local; remover AC por índice observado, agregar AC por texto, conservar notas y checks no afectados. Para dependencias que deban eliminarse, consultar ayuda vigente y reconciliar con CLI/MCP; no editar Markdown gestionado.
4. Actualizar los cuatro documentos modificados tras cotejar el contenido: `backlog doc view <ID>`, `backlog doc update <ID> --content ...`; conservar tags y conocimiento nuevo. Crear el nuevo documento y las tres decisiones como proposed; no aceptar en nombre del usuario.
5. Releer y verificar dependencias, AC y estados. Si el tablero evolucionó más allá de la semilla, conservar el trabajo y adaptar sólo el delta útil; no ejecutar el importador completo después de una conciliación que intencionadamente difiera de la semilla. No copiar la semilla al Home como segundo tracker.

## Fuentes específicas de v3

Se volvieron a leer las dos skills en main y Endurance Relay: sus blob SHA coinciden con las referencias anteriores. Se consultó CLI-INSTRUCTIONS upstream y documentación Bevy, bevy_egui, Rerun y MuJoCo Studio (enlaces en NOESIS-DOC-06). No se compiló el runtime, ni se ejecutaron tests gráficos, ni se importó contra Backlog real.
