from unittest.mock import patch

import pytest

from revise.generators.base import GenerationParams
from revise.generators.vllm_generator import VllmGenerationParams, VllmGenerator


@pytest.mark.unit
class TestGenerationParamsWithPytest:
    """Test cases for GenerationParams using pytest."""

    def test_default_params(self):
        """Test default generation parameters."""
        params = GenerationParams()
        assert params.do_sample is True
        assert params.max_new_tokens == 16
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k is None
        assert params.num_completions == 1

    def test_custom_params(self):
        """Test custom generation parameters."""
        params = GenerationParams(
            do_sample=False,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_completions=3,
        )
        assert params.do_sample is False
        assert params.max_new_tokens == 32
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.num_completions == 3


@pytest.mark.unit
class TestVllmGenerationParamsWithPytest:
    """Test cases for VllmGenerationParams using pytest."""

    def test_to_vllm_sampling_params(self):
        """Test conversion to vLLM SamplingParams."""
        params = VllmGenerationParams(
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_completions=3,
        )

        sampling_params = params.to_vllm_sampling_params()

        assert sampling_params.max_tokens == 32
        assert sampling_params.temperature == 0.7
        assert sampling_params.top_p == 0.9
        assert sampling_params.top_k == 50
        assert sampling_params.n == 3

    def test_to_vllm_sampling_params_none_top_k(self):
        """Test conversion with None top_k value."""
        params = VllmGenerationParams(top_k=None)

        sampling_params = params.to_vllm_sampling_params()

        assert sampling_params.top_k == -1


@pytest.mark.unit
class TestVllmGeneratorWithPytest:
    """Test cases for VllmGenerator using pytest and fixtures."""

    def test_init(self, mock_vllm):
        """Test initialization of VllmGenerator."""
        generator = VllmGenerator(model="test-model")

        assert generator.model == "test-model"
        assert generator.num_gpus == 2

    @patch("torch.cuda.device_count", return_value=0)
    def test_init_no_gpus(self, _):
        """Test initialization with no GPUs available."""
        with pytest.raises(RuntimeError):
            VllmGenerator(model="test-model")

    @patch("ray.get")
    @patch("revise.generators.vllm_generator._generate_on_gpu")
    def test_generate(self, mock_generate_on_gpu, mock_ray_get, mock_vllm):
        """Test generation with VllmGenerator."""
        mock_generate_on_gpu.remote.return_value = "future"
        mock_ray_get.return_value = [
            [(0, ["generated text 1"]), (1, ["generated text 2"])]
        ]

        generator = VllmGenerator(model="test-model")
        results = generator.generate(["prompt 1", "prompt 2"])

        assert results == [["generated text 1"], ["generated text 2"]]
        assert mock_generate_on_gpu.remote.call_count == 2  # Device count

    def test_chunk_prompts(self, mock_vllm):
        """Test _chunk_prompts method."""
        generator = VllmGenerator(model="test-model")
        prompts = ["prompt 1", "prompt 2", "prompt 3"]

        prompt_chunks, index_chunks = generator._chunk_prompts(prompts)

        assert prompt_chunks == [["prompt 1", "prompt 2"], ["prompt 3"]]
        assert index_chunks == [[0, 1], [2]]
