import pytest

from revise.evaluators.comparison import GSM8KEvaluator


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
