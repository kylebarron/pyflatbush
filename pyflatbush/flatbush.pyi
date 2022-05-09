from array import array
from typing import Optional

import numpy as np
from numpy.typing import NDArray

class Flatbush:

    numItems: int
    nodeSize: int
    _levelBounds: array
    _pos: int
    minX: float
    minY: float
    maxX: float
    maxY: float

    data: bytearray
    # These don't _actually_ inherit from memoryview.
    # Not sure how best to type them
    _boxes: memoryview
    _indices: memoryview

    @classmethod
    def from_buffer(cls, data: bytearray) -> "Flatbush": ...
    def __init__(
        self, numItems: int, nodeSize: int = 16, data: Optional[bytearray] = None
    ) -> None: ...
    def add_vectorized(
        self,
        minX: NDArray[np.float64],
        minY: NDArray[np.float64],
        maxX: NDArray[np.float64],
        maxY: NDArray[np.float64],
    ) -> memoryview: ...
    def add(self, minX: float, minY: float, maxX: float, maxY: float) -> int: ...
    def finish(self) -> None: ...
    def search(self, minX: float, minY: float, maxX: float, maxY: float) -> array: ...
