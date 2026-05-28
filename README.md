# PSO Benchmark Suite — Programación Paralela

Implementación de Particle Swarm Optimization (PSO) con **cinco estrategias de evaluación paralela** (V0–V4) y **dos casos de uso reales**:  
- Optimización de portfolio de inversiones (máximo ratio de Sharpe)  
- Cuantización óptima de paleta de colores para impresión 3D

Asignatura: **Programación Paralela** · CUNEF Universidad · Prof. Dr. José Luis Salmerón · Curso 2025-2026

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Comandos principales

```bash
# Suite de benchmarks completa (V0, V1, V4 × 4 funciones × 3 dims × 5 seeds)
python -m experiments.run_benchmarks --dims 2 10 30

# Grid search de hiperparámetros (405 combinaciones)
python -m experiments.run_grid_search

# Demo asyncio con latencia I/O simulada (V0 vs V1 vs V3)
python -m experiments.run_asyncio_demo

# Caso de uso 1: portfolio de inversiones
python -m experiments.run_portfolio_case --n-assets 10 --strategy V4_vectorised

# Caso de uso 2: cuantización de color para impresión 3D
python -m experiments.run_color_case --k 8 --strategy V4_vectorised
python -m experiments.run_color_case --source mi_imagen.png --k 6

# Animaciones GIF + profiling cProfile real
python -m experiments.run_extras

# Generar informe PDF
python -m experiments.generate_report --outdir results

# Tests (16 tests, 100% pass)
python -m pytest tests/ -v
```

---

## Estructura del proyecto

```
core/
├── engine.py            Bucle PSO principal (run_pso)
├── config.py            PSOConfig (w, c1, c2, n_particles, …)
├── state.py             IterationRecord, PSOResult
├── boundaries.py        clamp, reflect
└── topologies.py        global-best, ring

objectives/
├── functions.py         sphere, rosenbrock, rastrigin, ackley (+ fn_vec)
├── suite.py             BENCHMARKS, BenchmarkInstance
├── portfolio.py         Caso de uso 1: portfolio financiero
└── color_quantization.py  Caso de uso 2: paleta de colores (con fn_vec)

parallel/
├── sequential.py        V0 — baseline secuencial (SISD)
├── threading_eval.py    V1 — ThreadPoolExecutor / TLP
├── multiprocessing_eval.py  V2 — ProcessPoolExecutor + batching (MIMD)
├── async_eval.py        V3 — asyncio.gather (concurrencia cooperativa)
├── vectorised_eval.py   V4 — NumPy broadcasting / SIMD
└── factory.py           get_evaluator(name)

experiments/
├── run_benchmarks.py    Suite 180 ejecuciones
├── run_grid_search.py   405 combinaciones de hiperparámetros
├── run_asyncio_demo.py  Experimento latencia I/O asimétrica
├── run_portfolio_case.py  Caso de uso: portfolio
├── run_color_case.py    Caso de uso: cuantización de color
├── run_extras.py        GIF animations + cProfile real
├── analyse_results.py   Análisis y speedup plots
└── generate_report.py   Genera informe PDF completo

viz/                     Convergencia, contornos, speedup, portfolio, color
persistence/             JSON + CSV con metadatos de hardware
tests/                   pytest — reproducibilidad, monotonicidad, correctitud
results/                 Resultados pre-generados (benchmarks, casos de uso, PDF)
```

---

## Estrategias de paralelización (Taxonomía de Flynn)

| V  | Mecanismo | Flynn | Cuándo usar |
|----|-----------|-------|-------------|
| V0 | Loop secuencial Python | SISD | Baseline, funciones sub-µs |
| V1 | ThreadPoolExecutor | TLP | Evaluaciones I/O-bound |
| V2 | ProcessPoolExecutor + batching | MIMD | Evaluaciones CPU > 50ms |
| V3 | asyncio.gather | Cooperativo | Latencia I/O asimétrica |
| V4 | NumPy broadcasting | SIMD | Funciones analíticas vectorizables |

---

## Caso de uso 1: Portfolio de inversiones

PSO busca los pesos óptimos entre N activos que **maximizan el ratio de Sharpe** S = (µ_p − r_f) / σ_p.  
Cada partícula codifica N pesos normalizados a suma 1. Dimensión = N. Bounds = [0, 1].

```bash
python -m experiments.run_portfolio_case --n-assets 10 --n-particles 50 --n-iterations 500
```

## Caso de uso 2: Cuantización de color (impresión 3D)

PSO encuentra la paleta de K colores que **minimiza el error cuadrático de cuantización**:  
f(C) = Σᵢ minⱼ ||pᵢ − cⱼ||²  
Cada partícula codifica K colores RGB como vector de K×3 floats ∈ [0, 255]. Dimensión = K×3.  
Implementa `fn_vec` con broadcasting (M, n, K) para aprovechar V4 de forma nativa.

```bash
python -m experiments.run_color_case --k 8 --strategy V4_vectorised
python -m experiments.run_color_case --source imagen.png --k 6 --n-particles 50
```

---

## Informe

El informe PDF completo (`informe_PSO.pdf`) se encuentra en la raíz del proyecto y en `results/`.  
Cubre: metodología, Taxonomía de Flynn, Ley de Amdahl, Ley de Gustafson, Condiciones de Bernstein,  
resultados benchmark, grid search, asyncio, portfolio, cuantización de color, profiling cProfile real.

```bash
python -m experiments.generate_report  # regenera results/informe_PSO.pdf
```
