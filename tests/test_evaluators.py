import pytest

from revise.evaluators.comparison_evaluator import GSM8KEvaluator
from revise.evaluators.math_evaluator import MATHEvaluator


@pytest.mark.unit
class TestGSM8KEvaluator:
    gt_answers = [
        """Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72""",
        """Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10""",
        """In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5""",
    ]

    strict_correct_predictions = [
        """Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72""",
        """Weng earns 12/60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10""",
        """In the beginning, Betty has only 100 / 2 = $50.
Betty's grandparents gave her 15 * 2 = $30.
This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
#### 5""",
    ]

    flexible_correct_predictions = [
        """Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
The answer is 72.""",
        """Weng earns 12/60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $10.
The answer is 10.""",
        """In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $30.
This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
The answer is 5.""",
    ]

    def test_run_strict_mode_correct_predictions(self):
        evaluator = GSM8KEvaluator(mode="strict")
        exact_match = evaluator.run(self.gt_answers, self.strict_correct_predictions)
        assert exact_match == 1.0

        exact_match = evaluator.run(self.gt_answers, self.flexible_correct_predictions)
        assert exact_match == 0.0

    def test_run_flexible_mode_correct_predictions(self):
        evaluator = GSM8KEvaluator(mode="flexible")
        exact_match = evaluator.run(self.gt_answers, self.strict_correct_predictions)
        assert exact_match == 1.0

        exact_match = evaluator.run(self.gt_answers, self.flexible_correct_predictions)
        assert exact_match == 1.0

    def test_run_return_results(self):
        evaluator = GSM8KEvaluator(mode="strict")
        results = evaluator.run(
            self.gt_answers, self.strict_correct_predictions, return_results=True
        )
        assert results["exact_match"] == 1.0
        assert all(results["score_list"])


@pytest.mark.unit
class TestMATHEvaluator:
    gt_answers = [
        """For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$.
For example, $ax+3$ and $x-5$ must be equal when $x=2$.
This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$.
Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$.
Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."""
    ]

    correct_predictions = ["So $a+b=-3+3=\\boxed{0}$."]
    wrong_predictions = [
        "So $a+b=-3+3=\\boxed{1}$.",  # Wrong answer
        "So $a+b=-3+3=boxed{0}$.",  # Wrong format
        "",  # Blank
    ]

    def test_run_correct_predictions(self):
        evaluator = MATHEvaluator()
        exact_match = evaluator.run(self.gt_answers, self.correct_predictions)
        assert exact_match == 1.0

    def test_run_wrong_predictions(self):
        evaluator = MATHEvaluator()
        exact_match = evaluator.run(
            self.gt_answers * len(self.wrong_predictions), self.wrong_predictions
        )
        assert exact_match == 0.0

    def test_run_return_results(self):
        evaluator = MATHEvaluator()
        results = evaluator.run(
            self.gt_answers, self.correct_predictions, return_results=True
        )
        assert results["exact_match"] == 1.0
        assert all(results["score_list"])

    def test_run_return_results_wrong_predictions(self):
        evaluator = MATHEvaluator()
        results = evaluator.run(
            self.gt_answers * len(self.wrong_predictions),
            self.wrong_predictions,
            return_results=True,
        )
        assert results["exact_match"] == 0.0
        assert not all(results["score_list"])
