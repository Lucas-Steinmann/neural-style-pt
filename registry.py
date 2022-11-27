from typing import Generic, TypeVar, Callable, Dict, Any, Optional, Type, Mapping, Iterator, Union

from omegaconf import DictConfig

T = TypeVar('T')


class Registry(Generic[T], Mapping[str, Type[T]]):
    """ Class registry, which can be used to construct objects from dictionaries or dict-like configs. """

    def __init__(self, name: str, build_fun: Optional[Callable[[Union[DictConfig, Dict[str, Any]]], T]] = None):
        self._name = name
        self._build_fun = build_fun
        self._registry: Dict[str, Type[T]] = {}

    def register_class(self, cls: Type[T]):
        cls_name = cls.__name__
        if cls_name in self._registry:
            raise ValueError(f"Tried to register class with same name twice: {cls_name}")
        self._registry[cls_name] = cls
        return cls

    def build(self, params: Union[DictConfig, Dict[str, Any]]) -> T:
        if self._build_fun is not None:
            return self._build_fun(params, self)
        else:
            params = dict(params)
            typ = params.pop('type')
            return self._registry[typ](**params)

    def __getitem__(self, class_name: str) -> Type[T]:
        return self._registry[class_name]

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)
