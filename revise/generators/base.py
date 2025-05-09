from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class GenerationParams:
    """
    Generic generation parameters. Can be converted to specific framework configs.
    """

    do_sample: bool = True
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    num_completions: int = 1
    seed: int | None = None
    skip_special_tokens: bool = True


class BaseGenerator(ABC):
    """
    Abstract base class for generation engines.
    """

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[List[str]]: ...
