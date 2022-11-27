from abc import ABC, abstractmethod
from typing import Any, Dict, Union, TypeVar, Generic, List

from omegaconf import DictConfig

from registry import Registry

V = TypeVar('V')


class MultiScaleParamStrategy(ABC, Generic[V]):
    """
    A strategy computing how a parameter value should evolve over multiple steps.
    """

    def __init__(self):
        self._step = 0

    def step(self) -> None:
        """
        Increments the step by one.
        Can be used to precalculate the value if the value is computed recursively.
        """
        self._step += 1

    @abstractmethod
    def compute(self) -> V:
        """
        Returns the value which the strategy calculated for its current step.

        :return: the value
        """
        pass


def strategy_build_func(params: Union[DictConfig, Dict[str, Any], V], registry: Registry) -> MultiScaleParamStrategy[V]:
    """
    Extension over the regular build function to handle constant values.

    The regular build function which always passes the params to the constructor of the class.
    In this case we allow passing any constant values except dictionaries to be passed as well.
    If a non-dictionary value is passed, we return a :py:class:`~.ConstantParamStrategy` which always
    computes the value specified in `params`.

    :param params: either a dictionary describing the strategy or a constant non-dictionary value
    :param registry: should always resolve to `MULTISCALE_STRATEGIES`. Accepted for compatibility
        with the registry.
    :return: the built MultiScaleParamStrategy
    """
    if isinstance(params, (dict, DictConfig)):
        params = dict(params)
        typ = params.pop('type')
        return registry[typ](**params)
    return ConstantParamStrategy(params)


MULTISCALE_STRATEGIES = Registry('multiscale-strategies', build_fun=strategy_build_func)


@MULTISCALE_STRATEGIES.register_class
class ConstantParamStrategy(MultiScaleParamStrategy):
    """
    Multiscale param strategy, which simply returns a constant value, which is specified during initialization.

    :param value: the constant value
    """

    def __init__(self, value: V):
        super().__init__()
        self._value = value

    def compute(self) -> V:
        return self._value


@MULTISCALE_STRATEGIES.register_class
class ListParamStrategy(MultiScaleParamStrategy):
    """
    Multiscale param strategy, which receives a list of values and simply returns
    the value at the index of the current step.

    :param values: the list of values which should have the length of number of steps,
        which are executed.
    """

    def __init__(self, values: List[V]):
        super().__init__()
        self._values = values

    def compute(self) -> V:
        return self._values[self._step]
