"""
run_color_case.py — Caso de uso: optimización de paleta de colores para impresión 3D.

Encuentra los K colores (paleta) que minimizan el error de cuantización RGB
sobre una imagen dada, usando PSO con cualquiera de las 5 estrategias V0–V4.

V4 aprovecha fn_vec con broadcasting (M,n,K) sin bucle sobre partículas.

Uso:
    python -m experiments.run_color_case
    python -m experiments.run_color_case --k 6 --strategy V4_vectorised
    python -m experiments.run_color_case --source mi_imagen.png --k 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import PSOConfig, run_pso
from objectives.color_quantization import make_color_objective, decode_palette, quantize_image
from parallel import get_evaluator
from viz.color_viz import plot_palette, plot_quantization_comparison, plot_color_convergence
import persistence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("color_case")


def kmeans_baseline(pixels: np.ndarray, k: int, fn) -> float:
    """Calcula el fitness de la solución K-Means como baseline de comparación."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
        return float(fn(km.cluster_centers_.flatten()))
    except ImportError:
        return None


def main():
    p = argparse.ArgumentParser(description="PSO color palette optimisation")
    p.add_argument("--source",       default="synthetic",
                   help="'synthetic' | path to PNG/JPG")
    p.add_argument("--k",            type=int,   default=8,
                   help="Number of palette colors")
    p.add_argument("--sample",       type=int,   default=5000,
                   help="Max pixels to sample from image")
    p.add_argument("--n-particles",  type=int,   default=50)
    p.add_argument("--n-iterations", type=int,   default=400)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--strategy",     default="V4_vectorised")
    p.add_argument("--outdir",       default="results/color_case")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Building color objective: source=%s  K=%d  sample=%d",
                args.source, args.k, args.sample)
    fn, bounds, pixels = make_color_objective(
        source=args.source, k=args.k, sample=args.sample
    )
    dim = args.k * 3
    logger.info("Dimensión del problema: %d  (%d colores × 3 canales RGB)", dim, args.k)

    config = PSOConfig(
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        seed=args.seed,
        w=0.7298, c1=1.49618, c2=1.49618,
        stagnation=80,
    )
    evaluator = get_evaluator(args.strategy)

    logger.info("Running PSO: %d particles, %d iters, strategy=%s",
                config.n_particles, config.n_iterations, args.strategy)
    result = run_pso(fn, bounds, config, evaluator=evaluator)

    # ── Decode solution ───────────────────────────────────────────────────
    palette  = decode_palette(result.best_position, args.k)
    pso_fit  = result.best_fitness
    km_fit   = kmeans_baseline(pixels, args.k, fn)

    logger.info("=== RESULTS ===")
    logger.info("PSO  error: %.2f  (%.4f per pixel)", pso_fit, pso_fit / len(pixels))
    if km_fit is not None:
        ratio = km_fit / pso_fit if pso_fit > 0 else float("inf")
        logger.info("KMeans err: %.2f  ratio PSO/KMeans=%.3f  (%s)",
                    km_fit, ratio,
                    "PSO wins" if pso_fit < km_fit else "KMeans wins")
    logger.info("Palette (%d colors):", args.k)
    for i, c in enumerate(palette):
        logger.info("  Color %2d: RGB(%3d, %3d, %3d)", i + 1, *c)

    # ── Persist ───────────────────────────────────────────────────────────
    extra = {
        "k": args.k,
        "source": args.source,
        "n_pixels_sampled": len(pixels),
        "pso_error": pso_fit,
        "pso_error_per_pixel": pso_fit / len(pixels),
        "kmeans_error": km_fit,
        "palette_rgb": palette.tolist(),
    }
    persistence.save_result(
        result, outdir / "pso_result",
        benchmark_name=f"color_k{args.k}",
        strategy_name=args.strategy,
        extra_meta=extra,
    )

    # ── Visualisations ────────────────────────────────────────────────────
    plot_color_convergence(result, k=args.k, outfile=outdir / "convergence.png")

    plot_palette(
        palette,
        title=f"Paleta óptima PSO — {args.k} colores  (error={pso_fit:.0f})",
        outfile=outdir / "palette.png",
    )

    quantized = quantize_image(pixels, palette)
    plot_quantization_comparison(
        pixels, quantized, palette,
        outfile=outdir / "rgb_scatter.png",
    )

    # ── Comparativa de estrategias ────────────────────────────────────────
    logger.info("Running strategy comparison (V0 vs V4)...")
    comparison = {}
    for strat in ("V0_sequential", "V4_vectorised"):
        import time
        ev = get_evaluator(strat)
        cfg2 = PSOConfig(n_particles=30, n_iterations=200, seed=42,
                         w=0.7298, c1=1.49618, c2=1.49618, stagnation=50)
        t0 = time.perf_counter()
        r2 = run_pso(fn, bounds, cfg2, evaluator=ev)
        elapsed = time.perf_counter() - t0
        comparison[strat] = {"error": r2.best_fitness, "time": elapsed}
        logger.info("  %s → error=%.1f  time=%.3fs", strat, r2.best_fitness, elapsed)

    v0e = comparison["V0_sequential"]["time"]
    v4e = comparison["V4_vectorised"]["time"]
    logger.info("V4 speedup sobre V0: %.2f×", v0e / v4e if v4e > 0 else 0)

    (outdir / "strategy_comparison.json").write_text(
        json.dumps(comparison, indent=2)
    )
    logger.info("Results saved to %s/", outdir)


if __name__ == "__main__":
    main()
