# src/algos/scout/schedules.py
from __future__ import annotations
import math


class LinearSchedule:
    def __init__(self, start: float, end: float, iters: int):
        self.start, self.end, self.iters = start, end, max(1, iters)

    def at(self, it: int) -> float:
        t = min(max(it, 0), self.iters)
        return self.start + (self.end - self.start) * (t / self.iters)


class CosineSchedule:
    def __init__(self, start: float, end: float, iters: int):
        self.start, self.end, self.iters = start, end, max(1, iters)

    def at(self, it: int) -> float:
        cos_t = 0.5 * (
            1.0 - math.cos(math.pi * min(max(it, 0), self.iters) / self.iters)
        )
        return self.start + (self.end - self.start) * cos_t
