"""Factory for building evaluators by name."""
from parallel.sequential import sequential_evaluator
from parallel.threading_eval import ThreadEvaluator
from parallel.multiprocessing_eval import ProcessEvaluator
from parallel.async_eval import AsyncEvaluator
from parallel.vectorised_eval import VectorisedEvaluator

def get_evaluator(name, **kwargs):
    match name:
        case "V0_sequential": return sequential_evaluator
        case "V1_threading": return ThreadEvaluator(**kwargs)
        case "V2_multiprocessing": return ProcessEvaluator(**kwargs)
        case "V3_asyncio": return AsyncEvaluator(**kwargs)
        case "V4_vectorised": return VectorisedEvaluator(**kwargs)
        case _: raise ValueError(f"Unknown strategy: {name}")
