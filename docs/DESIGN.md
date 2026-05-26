# Documento de Diseño — PSO Benchmark Suite

## 1. Arquitectura

```
pso_project/
├── core/           Motor PSO (7 ficheros)
├── objectives/     Benchmarks + caso de uso portfolio (3 ficheros)
├── parallel/       Estrategias V0–V4 (6 ficheros)
├── persistence/    JSON + CSV + NPZ (2 ficheros)
├── viz/            Visualización (7 ficheros)
├── experiments/    Scripts CLI (7 ficheros)
├── tests/          pytest (1 fichero)
├── configs/        YAML (2 ficheros)
└── docs/           Diseño + informe
```

## 2. Decisiones clave

- **Evaluator protocol**: interfaz `evaluator(fn, positions) -> ndarray`.
- **Boundary**: clamp por defecto (determinista, `np.clip`).
- **Persistencia**: JSON meta + CSV historia + NPZ trayectorias.
- **V1 threading**: GIL impide paralelismo real CPU-bound.
- **V2 multiprocessing**: batching reduce IPC; rentable si f(x) > 10ms.
- **V3 asyncio**: útil con latencia I/O; `make_latent_fn` simula esto.
- **V4 vectorizado**: mejor para funciones analíticas (SIMD implícito).

## 3. Caso de uso: Portfolio de inversiones

PSO busca los pesos óptimos de N activos que maximizan el ratio de Sharpe.
Cada partícula = N pesos [0,1], normalizados a sumar 1.
Función objetivo = -Sharpe (rendimiento_ajustado_riesgo).
Espacio continuo, multimodal, sin grafos: ideal para PSO.
