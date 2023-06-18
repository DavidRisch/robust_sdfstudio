from typing import Callable


class ConfigurationsSetter:
    """
    Reconfigures an existing pipeline. Used to evaluate different configuration options-
    """

    def __init__(self, name: str, set_func: Callable):
        self.name = name
        self.set_func = set_func
