"""Validation-only stopping. A plateau is a stopping rule, not proof of overfit."""
from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    patience: int = 0
    min_delta: float = 0.0
    best: float = -math.inf
    stale_epochs: int = 0

    def __post_init__(self):
        if self.patience < 0 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('patience and min_delta must be nonnegative')

    def update(self, score):
        if not math.isfinite(score):
            raise ValueError('Validation score must be finite')
        if score > self.best + self.min_delta:
            self.best, self.stale_epochs = score, 0
        else:
            self.stale_epochs += 1
        return self.patience > 0 and self.stale_epochs >= self.patience
