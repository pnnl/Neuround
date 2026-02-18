import importlib


def __getattr__(name):
    if name in ("quadratic", "nonconvex", "rosenbrock"):
        return importlib.import_module(f"experiments.{name}")
    raise AttributeError(f"module 'experiments' has no attribute {name!r}")
