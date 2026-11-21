"""Page-Hinkley sequential change detector for error streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    delta: float = 0.03
    threshold: float = 0.25

    def __post_init__(self):
        if self.delta < 0 or self.threshold <= 0:
            raise ValueError("delta must be non-negative and threshold must be positive.")
        self.reset()

    def update(self, value: float) -> bool:
        if not 0 <= value <= 1:
            raise ValueError("Page-Hinkley observations must be in [0, 1].")
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        self.statistic = self.cumulative - self.minimum
        self.alarm = self.statistic > self.threshold
        return self.alarm

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.statistic = 0.0
        self.alarm = False

