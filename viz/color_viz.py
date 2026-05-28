"""Visualisation helpers for the color quantization case study."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_palette(palette: np.ndarray, title: str = "Optimised Palette", outfile=None):
    """Tira horizontal de K rectángulos de color con valores RGB anotados."""
    k = len(palette)
    fig, ax = plt.subplots(figsize=(k * 1.4, 1.8))
    for i, color in enumerate(palette):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255.0))
        ax.text(i + 0.5, -0.18,
                f"({color[0]},{color[1]},{color[2]})",
                ha="center", va="top", fontsize=7)
    ax.set_xlim(0, k)
    ax.set_ylim(-0.35, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig


def plot_quantization_comparison(original_pixels: np.ndarray,
                                  quantized_pixels: np.ndarray,
                                  palette: np.ndarray,
                                  outfile=None):
    """Scatter 3D original vs cuantizado en espacio RGB."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    subplot_kw={"projection": "3d"})
    rng    = np.random.default_rng(0)
    sample = min(2000, len(original_pixels))
    idx    = rng.choice(len(original_pixels), sample, replace=False)

    op = original_pixels[idx] / 255.0
    ax1.scatter(op[:, 0], op[:, 1], op[:, 2], c=op, s=2, alpha=0.5)
    ax1.set_title("Colores originales")
    ax1.set_xlabel("R"); ax1.set_ylabel("G"); ax1.set_zlabel("B")

    qp  = quantized_pixels[idx] / 255.0
    pal = palette / 255.0
    ax2.scatter(qp[:, 0], qp[:, 1], qp[:, 2], c=qp, s=2, alpha=0.5)
    ax2.scatter(pal[:, 0], pal[:, 1], pal[:, 2],
                c=pal, s=200, marker="*", edgecolors="black", zorder=5)
    ax2.set_title("Cuantizado — paleta PSO")
    ax2.set_xlabel("R"); ax2.set_ylabel("G"); ax2.set_zlabel("B")

    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig


def plot_color_convergence(result, k: int, outfile=None):
    """Convergencia del error de cuantización a lo largo de las iteraciones."""
    iters   = [r.iteration   for r in result.history]
    fitness = [r.best_fitness for r in result.history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, fitness, color="#c62828", lw=1.8, label="PSO best error")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error de cuantización (sum of sq. dist.)")
    ax.set_title(f"Cuantización de color — K={k} colores — Convergencia PSO")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outfile:
        fig.savefig(str(outfile), dpi=150)
    plt.close(fig)
    return fig
