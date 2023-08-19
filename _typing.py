
from typing import Union, Protocol, runtime_checkable

Numeric = Union[float, int]


@runtime_checkable
class Estimator(Protocol):

    def fit(self, *args, **kwargs):
        ...

    def predict(self, *args, **kwargs):
        ...
