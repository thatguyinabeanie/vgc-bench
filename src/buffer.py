from typing import Any

from stable_baselines3.common.buffers import RolloutBuffer


class MultiAgentBuffer(RolloutBuffer):
    def __init__(self, *args: Any, **kwargs: Any):
        buffer_size_override = kwargs["buffer_size_override"]
        kwargs.pop("buffer_size_override")
        super().__init__(buffer_size_override, *args[1:], **kwargs)
