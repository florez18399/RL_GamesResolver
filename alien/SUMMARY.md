# Resumen del experimento — DQN en Atari Alien

## Arquitecturas evaluadas

Se compararon 4 variantes de DQN sobre el mismo pipeline de preprocesamiento
(grayscale, 84×84, frame-skip 4, frame-stack 4) y el mismo presupuesto de
entrenamiento (100 000 pasos por variante).

| Variante           | Red           | Doble Q | Origen                          |
|--------------------|---------------|---------|---------------------------------|
| Vanilla DQN        | `VanillaDQN`  | No      | Mnih et al. 2015 (Nature)       |
| Double DQN         | `VanillaDQN`  | Sí      | Van Hasselt et al. 2015         |
| Dueling DQN        | `DuelingDQN`  | No      | Wang et al. 2016                |
| Double + Dueling   | `DuelingDQN`  | Sí      | Combinación estándar pre-Rainbow|

Diseño: eje 1 = clase de red (vanilla vs dueling), eje 2 = flag `double_dqn`.
Esto aísla el efecto de cada mejora y permite medir su combinación.

Parámetros compartidos (`BASE_KW`): `lr=1e-4`, `batch=32`, `gamma=0.99`,
`buffer=50k`, `target_update=1k`, `eps 1.0→0.05` en 50k pasos,
`reward_clip=1.0`, `train_every=4`, `grad_clip=10`.

## Resultados observados

- La combinación **Double + Dueling** fue la mejor en evaluación, seguida por
  Dueling solo. Double DQN aportó estabilidad frente a Vanilla pero poca
  ganancia de score absoluto con 100k pasos.
- Dueling tiene más parámetros (~2.22M vs ~1.69M) pero converge comparable en
  tiempo por el mismo cuello de botella convolucional.
- Con 100k pasos todas las variantes están lejos del régimen "resuelto": el
  paper original usa ~50M decisiones (500× más). Los resultados aquí son
  tendencias, no números finales.

## Comportamiento del agente

Observación clave desde los GIFs: **el agente aprendió a recolectar huevos
pero no a disparar ni a huir estratégicamente de los aliens.**

Causas identificadas:

1. **Recompensa densa vs cadena larga**: los huevos dan reward inmediato al
   pasar por encima; matar un alien requiere la secuencia "moverse → apuntar →
   disparar → impacto", mucho más difícil de descubrir por exploración.
2. **`reward_clip=1.0`** aplana la diferencia entre huevo (~10 pts) y alien
   (~200–300 pts): para el agente valen lo mismo, así que prefiere la opción
   segura y frecuente.
3. **`terminal_on_life_loss=True`** castiga el riesgo: acercarse a un alien
   puede terminar el episodio, desincentivando el combate.
4. **Exploración epsilon-greedy ineficiente**: 9 de 18 acciones involucran
   FIRE, pero la combinación "moverse + apuntar + disparar" tiene
   probabilidad ~1/18³ bajo política aleatoria.
5. **Presupuesto insuficiente**: 100k pasos es muy poco para que el crédito
   se propague a conductas multi-paso como el combate.

## Conclusiones

- El orden Vanilla < Double < Dueling < Double+Dueling es consistente con la
  literatura, incluso con presupuesto reducido.
- Las mejoras arquitectónicas ayudan, pero el comportamiento emergente está
  dominado por la **función de recompensa** y el **presupuesto de entrenamiento**,
  no por la red.
- Para que el agente aprenda a disparar/huir, lo más rentable sería: quitar
  `reward_clip`, desactivar `terminal_on_life_loss`, y entrenar ≥1M pasos.
  Extensiones útiles: Prioritized Experience Replay, Noisy Nets, o ir directo
  a Rainbow DQN.

## Archivos generados

- Pesos: `vanilla_dqn_weights.pt`, `double_dqn_weights.pt`,
  `dueling_dqn_weights.pt`, `double_dueling_dqn_weights.pt`
- Historiales: `*_run.pkl` por variante
- GIFs: `*_play.gif` por variante (con `loop=0` para reproducción infinita)
