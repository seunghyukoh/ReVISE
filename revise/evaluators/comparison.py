import re
import string
from abc import ABC, abstractmethod
from typing import List, Literal

import numpy as np

from revise.filters import BaseFilter, RegexFilter


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


def exact_match_compute(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list), "score_list": score_list.tolist()}


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

        self.regexes_to_ignore = [
            r",",
            r"\$",
            r"(?s).*#### ",
            r"\.$",
        ]
        self.ignore_case = True
        self.ignore_punctuation = False

    def run(
        self, answers: List[str], predictions: List[str], return_results: bool = False
    ) -> float:
        references = self.answer_filter.run(answers)
        predictions = self.prediction_filter.run(predictions)

        results = exact_match_compute(
            references=references,
            predictions=predictions,
            regexes_to_ignore=self.regexes_to_ignore,
            ignore_case=self.ignore_case,
            ignore_punctuation=self.ignore_punctuation,
        )

        if return_results:
            return results

        return results["exact_match"]
