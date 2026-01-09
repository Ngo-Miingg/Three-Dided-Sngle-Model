import numpy as np

class EMASmoother:
    """Simple exponential moving average smoother for 1D or vector values."""

    def __init__(self, alpha: float = 0.75):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    def reset(self):
        self._state = None

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._state is None:
            self._state = x.copy()
        else:
            self._state = self.alpha * self._state + (1.0 - self.alpha) * x
        return self._state.copy()
