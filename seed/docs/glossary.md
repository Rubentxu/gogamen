# Lenguaje ubicuo inicial (propuesto)

| ID | Definición | Contexto/alias |
|---|---|---|
| world.state | Estado completo latente del entorno en un tick | ground truth; no lectura de planner por defecto |
| world.observation | Señal parcial con fuente, tiempo y unidades | sensor measurement / semantic signal |
| world.belief | Distribución sobre estados o proposiciones plausibles | belief state |
| world.snapshot | Estado suficiente para repetir transición | checkpoint simulador, no estado de robot físico |
| world.model | Dinámica generativa de mundos hipotéticos | simulator / digital twin |
| planning.walker | Trayectoria virtual de la población FMC | no organismo real |
| planning.belief-particle | Muestra del estado latente actual | no walker futuro |
| planning.budget | Límite de transiciones y/o fecha monotónica | no recompensa |
| planning.future-freedom | Proxy explícito de libertad de futuros viables | no empowerment exacto |
| semantic.grounding | Asociación de expresión a entidad/relación/acción posible | situada en mundo |
| semantic.signal | Interpretación tipada de observación humana | no permiso |
| semantic.likelihood | Probabilidad de observar señal dado estado | distinto de confianza del modelo |
| action.proposal | Acción candidata que no produce efecto | salida planner |
| action.authorized | Acción validada por política, acotada y caducable | no garantía absoluta |
| action.outcome | Consecuencia observada al aplicar una acción | feedback |
| life.drive | Magnitud de necesidad que altera utilidad | homeostasis, no emoción sentida |
| life.genome | Parámetros heredables y mutables dentro de límites | no configuración de permisos |
| experiment.replay | Reproducción de observaciones, seed, versión y parámetros | no simulación nueva de hecho real |
| system.spec | Definición declarativa portable y versionada | no ejecuta efectos |
| system.plan | Plan inmutable de operaciones resueltas | no contiene punteros serializados |
| system.instance | Estado mutable de una instancia de plan | memoria, genoma y parámetros |
| system.capability | Contrato de operación, requisitos y efectos | declaración no certificación |
| runtime.deadline | Límite expresado en reloj monotónico | distinto del presupuesto de transiciones |
| runtime.fallback | Control alternativo validado por dominio | no equivalencia universal a stop |
| learning.strategy | Política candidata/promovida con evidencia | no altera permisos |
| cognition.memory | Evidencia episódica, semántica o procedimental | no proveedor LLM |
| automation.workflow | Transición reactiva de estados y efectos propuestos | no motor distribuido obligatorio |
| studio.projection | Transformación de snapshot a escena o vista derivada | no autoridad de mundo |
| studio.frame | Muestra visual descartable con secuencia y procedencia | no checkpoint |
| studio.control-command | Solicitud tipada de cambio con ID y revisión | no mutación directa |
| studio.ack | Confirmación o rechazo con instante y revisión | no promesa de seguridad física |
| experiment.trace | Registro versionado de entradas, acciones y metadatos | completitud explícita |
| experiment.branch | Simulación alternativa desde checkpoint | no historia real |
| experiment.arena | Comparación pareada de estrategias bajo condiciones equivalentes | no ranking por demo |
