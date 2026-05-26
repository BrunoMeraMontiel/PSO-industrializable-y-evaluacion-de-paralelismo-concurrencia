# PSO Benchmark Suite

Particle Swarm Optimization con 5 estrategias (V0–V4) y caso de uso aplicado a optimización de portfolio de inversiones.

## Instalación (WSL + Ubuntu)

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

## Comandos

```bash
# PSO individual
python -m experiments.run_pso --benchmark sphere_d2 --strategy V0_sequential --seed 42

# Suite de benchmarks
python -m experiments.run_benchmarks --dims 2 10 30

# Grid search
python -m experiments.run_grid_search --config configs/grid_search.yaml

# Visualizaciones
python -m experiments.make_viz --benchmark sphere_d2 --animate

# Análisis de resultados
python -m experiments.analyse_results --results-dir results/benchmarks

# Caso de uso: portfolio de inversiones
python -m experiments.run_portfolio_case --n-assets 10 --n-particles 50

# Tests
python -m pytest tests/ -v
```

## Estructura

```
core/
├── protocols.py         Interfaces (ObjectiveFn, Evaluator)
├── config.py            PSOConfig
├── state.py             IterationRecord, PSOResult
├── boundaries.py        clamp, reflect
├── topologies.py        global-best, ring
└── engine.py            Bucle PSO (run_pso)

objectives/
├── functions.py         sphere, rosenbrock, rastrigin, ackley (+vec)
├── suite.py             BenchmarkInstance, BENCHMARKS
└── portfolio.py         Caso de uso: portfolio de inversiones

parallel/
├── sequential.py        V0 — baseline secuencial
├── threading_eval.py    V1 — ThreadPoolExecutor (GIL)
├── multiprocessing_eval.py  V2 — ProcessPoolExecutor + batching
├── async_eval.py        V3 — asyncio (concurrencia cooperativa)
├── vectorised_eval.py   V4 — NumPy vectorizado
└── factory.py           get_evaluator()

persistence/             JSON + CSV + NPZ
viz/                     Convergencia, contornos, 3D, GIF, speedup, portfolio
experiments/             Scripts CLI + caso de uso
tests/                   pytest
```

## Diagrama de dependencias

```
experiments/  ──→  core/
    │                ↑
    ├──→  objectives/
    ├──→  parallel/     ──→  core/protocols.py (Evaluator)
    ├──→  persistence/
    └──→  viz/
```

## Estrategias

| V  | Fichero | Descripción | Cuándo usar |
|----|---------|-------------|-------------|
| V0 | sequential.py | Loop secuencial | Baseline, debug |
| V1 | threading_eval.py | ThreadPoolExecutor | f(x) con I/O |
| V2 | multiprocessing_eval.py | ProcessPoolExecutor + batch | f(x) > 10ms CPU |
| V3 | async_eval.py | asyncio.gather | f(x) con latencia I/O |
| V4 | vectorised_eval.py | NumPy broadcasting | f(x) analítica |

## Caso de uso: Portfolio de inversiones

Dado un conjunto de N activos financieros con rendimientos esperados y matriz de covarianza, PSO busca los pesos óptimos [0,1] que maximizan el ratio de Sharpe (rendimiento ajustado por riesgo). Cada partícula codifica N pesos que se normalizan para sumar 1. La función objetivo es -Sharpe (negado porque PSO minimiza). Dimensión = N activos, bounds [0,1].
