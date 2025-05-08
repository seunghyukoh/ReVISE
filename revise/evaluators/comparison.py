from abc import ABC, abstractmethod
from typing import List, Literal

from revise.filters import BaseFilter, RegexFilter
import evaluate


class BaseComparisonEvaluator(ABC):
    answer_filter: BaseFilter
    prediction_filter: BaseFilter

    @abstractmethod
    def run(self, answers: List[str], predictions: List[str]) -> float:
        """Run the evaluator on the given answers and predictions.

        Args:
            answers (List[str]): gt_answers including the reference answer
            predictions (List[str]): predictions including the extracted answer

        Returns:
            float: score
        """
        ...


class GSM8KEvaluator(BaseComparisonEvaluator):
    STRICT_MATCH = "#### (\\-?[0-9\\.\\,]+)"
    FLEXIBLE_EXTRACT = "(-?[$0-9.,]{2,})|(-?[0-9]+)"

    def __init__(
        self,
        mode: Literal["strict", "flexible"] = "strict",
    ):
        assert mode in ["strict", "flexible"], "Invalid mode"
        self.mode = mode
        is_strict = self.mode == "strict"

        self.answer_filter = RegexFilter(pattern=self.STRICT_MATCH)
        self.prediction_filter = RegexFilter(
            pattern=self.STRICT_MATCH if is_strict else self.FLEXIBLE_EXTRACT,
            select_index=0 if is_strict else -1,
        )

        self.metric = evaluate.load("exact_match")
        self.regexes_to_ignore = [
            r",",
            r"\$",
            r"(?s).*#### ",
            r"\.$",
        ]
        self.ignore_case = True
        self.ignore_punctuation = False

    def run(self, answers: List[str], predictions: List[str]) -> float:
        references = self.answer_filter.run(answers)
        predictions = self.prediction_filter.run(predictions)

        results = self.metric.compute(
            references=references,
            predictions=predictions,
            regexes_to_ignore=self.regexes_to_ignore,
            ignore_case=self.ignore_case,
            ignore_punctuation=self.ignore_punctuation,
        )

        return results["exact_match"]
