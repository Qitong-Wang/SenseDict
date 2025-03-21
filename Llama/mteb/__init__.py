from __future__ import annotations

from importlib.metadata import version

from mteb.benchmarks import (
    MTEB_MAIN_EN,
    MTEB_MAIN_RU,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
    CoIR,
)
from mteb.evaluation import *
from mteb.load_results import load_results
from mteb.models import get_model, get_model_meta
from mteb.overview import TASKS_REGISTRY, get_task, get_tasks

from .benchmarks import Benchmark

#__version__ = version("mteb")  # fetch version from install metadata
print("./mteb/__init__.py omit check version")

__all__ = [
    "MTEB_MAIN_EN",
    "MTEB_MAIN_RU",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "CoIR",
    "TASKS_REGISTRY",
    "get_tasks",
    "get_task",
    "get_model",
    "get_model_meta",
    "load_results",
    "Benchmark",
]
