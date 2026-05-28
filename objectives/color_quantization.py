"""
Caso de uso: optimización de paleta de colores para impresión 3D.

Encuentra los K colores que minimizan el error de cuantización total:
  f(C) = Σ_i  min_j  ||p_i − c_j||²

Cada partícula codifica K colores como vector plano de K×3 floats ∈ [0,255].

Soporta V4 vectorized: fn_vec(positions) evalúa todo el enjambre a la vez
mediante broadcasting (M, n, K) sin bucle Python sobre partículas.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


def _load_pixels(source) -> np.ndarray:
    if isinstance(source, np.ndarray):
        if source.ndim == 3:
            return source.reshape(-1, 3).astype(np.float64)
        return source.astype(np.float64)
    if isinstance(source, (str, Path)):
        from PIL import Image
        img = np.array(Image.open(source).convert("RGB"))
        return img.reshape(-1, 3).astype(np.float64)
    raise TypeError(f"Unsupported source type: {type(source)}")


def make_color_objective(source, k: int = 8, sample: int = 5000):
    """
    Construye la función objetivo PSO para cuantización de color.

    Parameters
    ----------
    source : "synthetic" | path | ndarray
        Imagen o fuente de píxeles.
    k : int
        Número de colores en la paleta (dimensión = k*3).
    sample : int
        Máximo de píxeles a usar (submuestrea si la imagen es grande).

    Returns
    -------
    fn        : callable  f(x: ndarray[k*3]) -> float
    bounds    : list of (0.0, 255.0)  length k*3
    pixels    : ndarray (M, 3)  píxeles muestreados
    """
    if source == "synthetic":
        rng = np.random.default_rng(42)
        centers = rng.integers(30, 225, size=(5, 3)).astype(np.float64)
        pixels = np.vstack([c + rng.normal(0, 15, (200, 3)) for c in centers])
        pixels = np.clip(pixels, 0, 255)
    else:
        pixels = _load_pixels(source)

    if len(pixels) > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pixels), sample, replace=False)
        pixels = pixels[idx]

    # ── función escalar (una partícula) ──────────────────────────────────
    def fn(x: np.ndarray) -> float:
        palette = x.reshape(k, 3)
        dists = np.sum((pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
        return float(np.sum(np.min(dists, axis=1)))

    # ── función vectorizada (todo el enjambre, para V4) ──────────────────
    def fn_vec(positions: np.ndarray) -> np.ndarray:
        """
        positions : (n_particles, k*3)
        returns   : (n_particles,)  array de fitness

        Broadcasting: (M, 1, 1, 3) − (1, n, k, 3) → (M, n, k)
        Con M=5000, n=50, k=8: ~16 MB en float64 — manejable.
        """
        n = positions.shape[0]
        palettes = positions.reshape(n, k, 3)              # (n, k, 3)
        # pixels (M,3) → (M,1,1,3),  palettes (n,k,3) → (1,n,k,3)
        dists = np.sum(
            (pixels[:, None, None, :] - palettes[None, :, :, :]) ** 2,
            axis=3,
        )                                                   # (M, n, k)
        return np.sum(np.min(dists, axis=2), axis=0).astype(float)  # (n,)

    fn.fn_vec = fn_vec

    bounds = [(0.0, 255.0)] * (k * 3)
    return fn, bounds, pixels


def decode_palette(x: np.ndarray, k: int) -> np.ndarray:
    """Vector plano → (k, 3) uint8 clampeado a [0,255]."""
    return np.clip(x.reshape(k, 3), 0, 255).astype(np.uint8)


def quantize_image(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Reemplaza cada píxel por el color de paleta más cercano."""
    flat = image.reshape(-1, 3).astype(np.float64)
    pal  = palette.astype(np.float64)
    dists = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    return palette[np.argmin(dists, axis=1)].reshape(image.shape)
