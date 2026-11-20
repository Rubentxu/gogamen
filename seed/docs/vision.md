# Noesis — visión y alcance

Framework de computación funcional, reactiva, cognitiva y adaptativa de propósito general, especializado por dominios; inteligencia situada, no un wrapper de LLM ni una afirmación de AGI. Un agente percibe, estima, imagina, elige, actúa y contrasta. Cuerpo virtual o físico; lenguaje como sentido y actuador de comunicación. Núcleo local Rust, proveedor System One opcional. El algoritmo de planificación debe poder funcionar sin Internet.

## MVP entregable (capacidades M0–M3, SDK Rust, frontend de autoría Kotlin y release)
- Ejecutar Grid, CartPole, Kart y Life desde CLI con semillas, trazas, baselines y replay; mantener física y render separados. Studio compartido opcional visualiza Grid, CartPole, Kart, Life, Language World y Automation; proyecciones comunes, inspector, timeline y replay. La CLI y el core no necesitan dependencias gráficas. Arena, contrafactual, editor, Rerun y web no bloquean MVP.
- Language World: observación humana falible, grounded semánticamente, fusión, planificación bajo observación parcial, acción Ask/Inspect, proveedor replay sin coste. Adaptador Jev real opt-in, no condición de release.
- Implementación FMC con variante publicada fijada; proxy CEF explícitamente etiquetado. Estudio de rendimiento comparativo, no promesa de vencer MCTS. Resultados negativos válidos.
- Core general: grafo tipado reactivo, reloj, scheduler, workflow, efectos, guardia. Laboratorio de servicio local funciona sin IA y añade clasificación semántica por composición. Core de inteligencia incluye FMC, entropía, percepción, cognición, memoria, aprendizaje, motivación y comunicación, con implementación offline de referencia; no trasladarlos a extensiones.
- SDK Rust y conformance externo; SystemSpec temprana, compilador Rust, artefacto ejecutable sin JVM y frontend Kotlin tipado. No exigir lenguaje general nuevo, JNI, SDK Python ni Wasm para release.
- Control multirritmo simulado, aislamiento de cognición, presupuestos y métricas temporales. Hard realtime no garantizado. M4 Mario/MuJoCo en paralelo cuando dependencias disponibles; no bloquean MVP, Jev real tampoco. No calendarizar hitos ni convertirlos en dependencias artificiales.

## North star
Un tercero puede crear un entorno, enchufar una inteligencia y observar cómo se adapta sin escribir el algoritmo de cada comportamiento. Un mismo planner se reutiliza en simulaciones continuas/discretas y con conocimiento parcial con adaptaciones explícitas. El humano puede advertir, enseñar o proponer objetivos sin invadir la autoridad de seguridad.

Indicadores: éxito y supervivencia bajo presupuestos iguales, violaciones de invariantes, coste de decisión, latencia, calibración, capacidad de recuperación de error del modelo, capacidad de un tercero de integrar un mundo. Objetivos de superioridad se establecen tras baseline, no antes. Noesis no promete inteligencia general, conciencia, emociones sentidas ni control autónomo seguro de cualquier máquina.

Productos: Engine, SDK, Labs, Studio, Connect, Language son superficies, inicialmente un workspace. Repos sugeridos `noesis` (código) y `noesis-home` (privado, Backlog único); nombres no verificados como disponibles ni registrados. Licencia del core propuesta Apache-2.0 o MIT tras revisión de dependencias; no copiar implementaciones con licencia incompatible, documentar origen de algoritmos.
