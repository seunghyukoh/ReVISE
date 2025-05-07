from abc import ABC, abstractmethod
from typing import List

import regex
from tqdm import tqdm


class BaseFilter(ABC):
    TIMEOUT_ERROR = "[timeout]"
    ERROR = "[error]"
    NO_MATCH = "[no match]"
    FALLBACK = "[fallback]"

    @abstractmethod
    def run(self, text: str) -> str: ...


class RegexFilter(BaseFilter):
    def __init__(
        self,
        pattern: str,
        select_index: int = 0,
        timeout: float = 5.0,
    ):
        self.pattern = pattern
        self.select_index = select_index
        self.timeout = timeout

    def run_single(self, text: str) -> str:
        try:
            match = regex.findall(self.pattern, text, timeout=self.timeout)
            if match:
                match = match[self.select_index]
                if isinstance(match, tuple):
                    match = [m for m in match if m]
                    if match:
                        match = match[0]
                    else:
                        match = self.FALLBACK
                return match.strip()
            else:
                return self.NO_MATCH
        except TimeoutError:
            return self.TIMEOUT_ERROR
        except Exception as _:
            return self.ERROR

    def run(self, texts: List[str], disable_tqdm: bool = False) -> List[str]:
        results = []

        for text in tqdm(texts, disable=disable_tqdm):
            results.append(self.run_single(text))

        return results
