# Laboratorio: DQN en Atari Alien

Laboratorio de aprendizaje por refuerzo profundo sobre el entorno
`ALE/Alien-v5` de Gymnasium/ALE-py, comparando cuatro variantes de DQN:
Vanilla, Double, Dueling y Double+Dueling.

Todo el código vive en `alien_train.ipynb`.

---

## 1. Ambiente: Alien (Atari 2600)

El jugador controla a un astronauta dentro de una nave invadida por aliens.
Debe recorrer pasillos destruyendo huevos que los aliens depositan por el
mapa, mientras evita o dispara a los aliens que lo persiguen. Cada vida
termina cuando un alien toca al jugador.

### 1.1 Espacio de acciones

`ALE/Alien-v5` expone el **conjunto completo de 18 acciones** del joystick
Atari 2600 (`full_action_space=True`). Cada acción combina una dirección de
8 posiciones + botón FIRE:

| Id | Acción         | Id | Acción             |
|----|----------------|----|---------------------|
| 0  | NOOP           | 9  | UPFIRE              |
| 1  | FIRE           | 10 | RIGHTFIRE           |
| 2  | UP             | 11 | LEFTFIRE            |
| 3  | RIGHT          | 12 | DOWNFIRE            |
| 4  | LEFT           | 13 | UPRIGHTFIRE         |
| 5  | DOWN           | 14 | UPLEFTFIRE          |
| 6  | UPRIGHT        | 15 | DOWNRIGHTFIRE       |
| 7  | UPLEFT         | 16 | DOWNLEFTFIRE        |
| 8  | DOWNRIGHT      | 17 | DOWNLEFT            |

9 de las 18 acciones involucran FIRE. La política aleatoria
que usa epsilon-greedy dispara muy poco en la dirección correcta porque la
combinación "moverse + apuntar + disparar" es rara por azar.

### 1.2 Espacio de observaciones

Frame crudo RGB de Atari: `(210, 160, 3)` uint8. Para DQN se aplica el
pipeline estándar:

1. **NoopReset** (hasta 30 no-ops al inicio) para variar el estado inicial.
2. **Frame-skip 4** con max-pool de los 2 últimos frames (evita parpadeo de
   sprites en Atari).
3. **Grayscale + resize a 84×84**.
4. **FrameStack 4**: la observación final es `(4, 84, 84)` uint8 — el agente
   "ve" los últimos 4 frames apilados, lo que codifica velocidad y dirección
   de movimiento sin memoria recurrente.
5. **TransformObservation** a `float32` normalizado a `[0, 1]` justo antes
   de entrar a la red.

### 1.3 Recompensa

Recompensa directa del juego: puntos por recolectar huevos y por matar
aliens. Durante entrenamiento se aplica **reward clipping a `[-1, +1]`**
(`reward_clip=1.0`) para estabilizar el entrenamiento pero aplana la diferencia entre eventos de bajo y alto valor (un huevo ≈ un alien muerto para el agente).

Señal adicional: `terminal_on_life_loss=True` — cada pérdida de vida se reporta como fin de episodio para el agente (el entorno real continúa hasta perder las 3 vidas).

---

## 2. Flujo lógico del entrenamiento

```
┌─────────────────────────────────────────────────────────────┐
│  loop de entrenamiento (total_steps = 100 000)              │
│                                                             │
│  1. env.reset()  →  obs (4,84,84) float32                   │
│  2. a = agent.act(obs)   # epsilon-greedy sobre Q(s,·)      │
│  3. obs', r, term, trunc, info = env.step(a)                │
│  4. replay_buffer.push(obs, a, r_clipped, obs', done)       │
│  5. cada train_every=4 pasos, y con buffer ≥ min_buffer:    │
│        batch = replay_buffer.sample(32)                     │
│        loss = (r + γ·Q_target(s', a*) - Q(s,a))²            │
│        loss.backward(); optimizer.step()                    │
│  6. cada target_update_freq=1000 pasos:                     │
│        Q_target.load_state_dict(Q.state_dict())             │
│  7. cada eval_every=10 000 pasos:                           │
│        evaluar con 3 episodios sin exploración              │
│        guardar history (train_returns, eval_returns)        │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Detalles del ambiente

- **18 acciones** → exploración más costosa que en juegos simples (Pong
  tiene 6). El agente necesita más pasos para cubrir el espacio.
- **Frame-skip 4** → una decisión del agente se repite 4 frames. El horizonte
  efectivo es ×4: `γ=0.99` sobre pasos del agente equivale a `γ≈0.96` sobre
  frames del juego.
- **Cadena de crédito larga para disparar**: mover → apuntar → disparar →
  impacto son ≥3 decisiones del agente. El reward clipping + life-loss
  termination hacen que el agente prefiera la conducta densa y segura
  (recoger huevos) sobre la cadena larga y riesgosa (combatir aliens).

### 2.2 Hiperparámetros compartidos (`BASE_KW`)

| Parámetro              | Valor    |
|------------------------|----------|
| `learning_rate`        | 1e-4     |
| `batch_size`           | 32       |
| `gamma`                | 0.99     |
| `grad_clip`            | 10.0     |
| `buffer_capacity`      | 50 000   |
| `min_buffer_size`      | 5 000    |
| `target_update_freq`   | 1 000    |
| `eps_start → eps_end`  | 1.0 → 0.05 en 50 000 pasos |
| `reward_clip`          | 1.0      |
| `train_every`          | 4        |

---

## 3. Redes neuronales

### 3.1 VanillaDQN 
```
Input: (B, 4, 84, 84)  float32 ∈ [0,1]
  ↓
Conv2d(4 → 32, kernel=8, stride=4) + ReLU       → (B, 32, 20, 20)
Conv2d(32 → 64, kernel=4, stride=2) + ReLU      → (B, 64,  9,  9)
Conv2d(64 → 64, kernel=3, stride=1) + ReLU      → (B, 64,  7,  7)
Flatten                                         → (B, 3136)
Linear(3136 → 512) + ReLU
Linear(512 → 18)                                → Q(s, a) ∀ a
```

**Parámetros:** ~1.69 M.

### 3.2 DuelingDQN

Mismo tronco convolucional, pero separa la cabeza en **dos streams**:

```
   ... Flatten → (B, 3136)
            │
   ┌────────┴────────┐
   │                 │
Value stream     Advantage stream
Linear(3136→512) Linear(3136→512)
ReLU             ReLU
Linear(512→1)    Linear(512→18)
   │                 │
   V(s)            A(s,a)
       ↓          ↓
   Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
```

La descomposición `Q = V + (A - mean A)` permite aprender el **valor del
estado** independientemente de la acción, lo cual es útil en estados donde
la acción importa poco (pasillos vacíos) y discriminar cuando sí importa
(alien cerca). La resta de la media es la normalización de identificabilidad
propuesta por Wang et al.

**Parámetros:** ~2.22 M.

### 3.3 Doble Q-learning (Van Hasselt 2015)

No es una arquitectura distinta, es un cambio en el **target**:

```
# DQN estándar
y = r + γ · max_a Q_target(s', a)

# Double DQN
a* = argmax_a Q_online(s', a)
y  = r + γ · Q_target(s', a*)
```

Reduce el sesgo de sobreestimación de `max` cuando las Q tienen ruido.
Se activa con `config.double_dqn=True` y funciona con cualquier red
(vanilla o dueling).

---

## 4. Resultados

Cada variante se entrenó con los mismos 100 000 pasos del entorno. Se
evaluó cada 10 000 pasos con 3 episodios en modo greedy (`eps=0`).

### 4.1 Orden observado

De peor a mejor en retorno de evaluación final:

```
Vanilla DQN  <  Double DQN  <  Dueling DQN  <  Double + Dueling
```

- **Vanilla** entrega el baseline. Con 100k pasos y 18 acciones apenas sale
  de la política casi-aleatoria.
- **Double** estabiliza pero no explota el score — con buffer pequeño los
  errores de sobreestimación no son aún el cuello de botella dominante.
- **Dueling** aporta la mejora individual más grande. La separación V/A
  ayuda a aprender el valor de "estar vivo en este pasillo" sin necesitar
  acertar la acción óptima en cada estado.
- **Double + Dueling** combina ambas ganancias y es la mejor en score y en
  estabilidad entre evaluaciones.

---

## 5. Reflexiones
Las 4 alternativas analizadas presentaron un agente que si aprendió las bases del juego, aleatoriamente el puntaje rondaba 30 puntos, mientras que con el agente entranado se lograron puntajes de mas de 200 puntos (el mejor de 720 puntos)

Estas arquitecturas son complejas y la literatura menciona que son necesarios cerca de 1 Millon de pasos de entrenamiento, en este laboratorio por cada una se hicieron 100mil por lo que hay una gran posibilidad de que pudieran mejorar aún más.

El entorno de Alien presentó una complejidad, en la cuadrícula, en los frames y en la cantidad de movimientos. Validando el movimiento del jugador se encuentra que aprendió a recoger los huevos y disparar, pero prefiere recoger los huevos ya que la recompensa es más instantánea. Otros movimientos como detectar a un alien -> apuntar -> disparar , son más complejos y en tan pocas etapas de entrenamiento es difícil que el agente encuentre esa secuencia las veces suficientes como para entender que ese movimiento le da más recompensa. De igual manera si se realizara un entrenamiento con más pasos y con un decaimiento menor del valor de epsilon es posible que el agente intentara y aprendiera nuevos movimientos.

Otras conclusiones se encuentran en SUMMARY.md

---

## 5. Siguientes pasos sugeridos

Para cerrar la brecha entre "recolectar huevos" y "jugar Alien":

1. **Quitar reward clipping** (`reward_clip=None`) — experimento de control
   barato para ver si el agente aprende a disparar cuando el alien vale más.
2. **Desactivar `terminal_on_life_loss`** — deja de castigar el riesgo.
3. **Entrenar ≥1M pasos** con `eps_decay_steps=200 000` — presupuesto mínimo
   realista para Alien.
4. **Prioritized Experience Replay** — prioriza transiciones con mayor
   TD-error (las pocas veces que mata un alien).
5. **Noisy Nets** — reemplaza epsilon-greedy por exploración paramétrica,
   mucho más eficiente en espacios de 18 acciones.
6. **Rainbow DQN** — combina las 6 extensiones estándar (Double, Dueling,
   PER, N-step, Distributional, Noisy). Es el punto de referencia moderno.
