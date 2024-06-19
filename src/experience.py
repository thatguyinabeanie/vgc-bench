from dataclasses import dataclass

from torch import Tensor


@dataclass
class Experience:
    state: Tensor
    action: int
    next_state: Tensor
    reward: float
    done: bool
