import pytest
from unittest.mock import patch


@pytest.fixture
def mock_vllm():
    """Fixture that patches vLLM dependencies for testing."""
    with patch("ray.init"), patch("ray.get"), patch("ray.shutdown"), patch(
        "torch.cuda.device_count", return_value=2
    ):
        yield
