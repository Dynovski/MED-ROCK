from typing import Tuple


class Point2D:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    @property
    def value(self) -> Tuple[float, float]:
        return self.x, self.y
