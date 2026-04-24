"""
Visualizador del agente entrenado jugando Blackjack.
Carga la Q-table guardada por train_blackjack.py y muestra cada mano
como una pantalla de juego animada con matplotlib.

Uso:
    python watch_agent.py              # 20 manos automáticas
    python watch_agent.py --hands 10   # número personalizado
    python watch_agent.py --step        # avanza mano a mano con Enter
"""

import argparse
import pickle
import time
from collections import defaultdict

import gymnasium as gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# Cargar Q-table
# ─────────────────────────────────────────────

def load_q_table(path="./blackjack/q_table.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    q = defaultdict(lambda: np.zeros(2))
    q.update(data)
    return q


def get_action(q_values, obs):
    return int(np.argmax(q_values[obs]))


# ─────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────

CARD_W, CARD_H = 0.55, 0.85
SUITS = ["♠", "♥", "♦", "♣"]


def draw_card(ax, x, y, label, face_down=False):
    color = "#2c3e50" if face_down else "white"
    txt_color = "white" if face_down else "#c0392b" if label in ["♥", "♦"] else "#2c3e50"
    rect = mpatches.FancyBboxPatch(
        (x, y), CARD_W, CARD_H,
        boxstyle="round,pad=0.04",
        linewidth=1.5,
        edgecolor="#7f8c8d",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    if not face_down:
        ax.text(
            x + CARD_W / 2, y + CARD_H / 2, str(label),
            ha="center", va="center",
            fontsize=18, fontweight="bold", color=txt_color, zorder=4,
        )
    else:
        ax.text(
            x + CARD_W / 2, y + CARD_H / 2, "?",
            ha="center", va="center",
            fontsize=22, fontweight="bold", color="white", zorder=4,
        )


def render_state(ax, player_sum, dealer_card, usable_ace, action_label,
                 result_label, result_color, episode, wins, losses, draws):
    ax.clear()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#076324")

    # ── Fondo ──
    fig = ax.get_figure()
    fig.patch.set_facecolor("#076324")

    # ── Título ──
    ax.text(3, 4.75, "BLACKJACK  —  Agente Q-Learning",
            ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")

    # ── Marcador ──
    ax.text(0.15, 4.75,
            f"Mano {episode}    W:{wins}  L:{losses}  D:{draws}",
            ha="left", va="center", fontsize=9, color="#f0e68c")

    # ── Dealer ──
    ax.text(3, 3.85, "Dealer", ha="center", va="center",
            fontsize=11, color="white", fontstyle="italic")
    draw_card(ax, 1.8, 2.85, dealer_card)
    draw_card(ax, 2.5, 2.85, face_down=True, label="?")

    # ── Jugador ──
    ax.text(3, 2.45, "Jugador", ha="center", va="center",
            fontsize=11, color="white", fontstyle="italic")

    ace_label = "  (As usable)" if usable_ace else ""
    ax.text(3, 2.2, f"Valor mano: {player_sum}{ace_label}",
            ha="center", va="center", fontsize=10, color="#f0e68c")

    # Representar la mano del jugador con una carta grande central
    draw_card(ax, 2.72, 1.1, player_sum)

    # ── Acción ──
    if action_label:
        a_color = "#27ae60" if action_label == "STAND" else "#e74c3c"
        ax.text(3, 0.82, f"Acción: {action_label}",
                ha="center", va="center", fontsize=12,
                fontweight="bold", color=a_color)

    # ── Resultado ──
    if result_label:
        ax.text(3, 0.45, result_label,
                ha="center", va="center", fontsize=15,
                fontweight="bold", color=result_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa",
                          edgecolor=result_color, linewidth=2))

    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()


# ─────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=20, help="Número de manos a jugar")
    parser.add_argument("--step", action="store_true", help="Avanzar con Enter")
    parser.add_argument("--delay", type=float, default=1.8,
                        help="Segundos entre manos (modo automático)")
    args = parser.parse_args()

    q_values = load_q_table()
    env = gym.make("Blackjack-v1", sab=False)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#076324")
    plt.tight_layout(pad=0.3)

    wins = losses = draws = 0

    for episode in range(1, args.hands + 1):
        obs, _ = env.reset()
        player_sum, dealer_card, usable_ace = obs
        done = False
        last_action = None

        while not done:
            action = get_action(q_values, obs)
            last_action = "HIT" if action == 1 else "STAND"

            render_state(ax, player_sum, dealer_card, usable_ace,
                         last_action, None, "white",
                         episode, wins, losses, draws)
            plt.pause(0.6)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
            player_sum, dealer_card, usable_ace = obs

        # Resultado final
        if reward > 0:
            result, color = "VICTORIA ✓", "#2ecc71"
            wins += 1
        elif reward < 0:
            result, color = "DERROTA ✗", "#e74c3c"
            losses += 1
        else:
            result, color = "EMPATE", "#f39c12"
            draws += 1

        render_state(ax, player_sum, dealer_card, usable_ace,
                     last_action, result, color,
                     episode, wins, losses, draws)

        if args.step:
            input(f"  [{episode}/{args.hands}] {result}  — Presiona Enter para continuar...")
        else:
            plt.pause(args.delay)

    total = wins + losses + draws
    print(f"\n── Resumen ──────────────────────────")
    print(f"  Manos jugadas : {total}")
    print(f"  Victorias     : {wins}  ({wins/total:.1%})")
    print(f"  Derrotas      : {losses}  ({losses/total:.1%})")
    print(f"  Empates       : {draws}  ({draws/total:.1%})")

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
