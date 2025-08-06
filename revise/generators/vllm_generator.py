from dataclasses import dataclass
from typing import List, Tuple

import ray
import torch
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .base import BaseGenerator, GenerationParams


@dataclass
class VllmGenerationParams(GenerationParams):
    """
    Generation parameters for vLLM.
    """

    def to_vllm_sampling_params(self) -> SamplingParams:
        """
        Convert to vLLM SamplingParams.
        """
        return SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k is not None else -1,
            n=self.num_completions,
            seed=self.seed,
            skip_special_tokens=self.skip_special_tokens,
        )


@ray.remote(num_gpus=1)
def _generate_on_gpu_chat(
    messages: List[List[ChatCompletionMessageParam]],
    indices: List[int],
    sampling_params: VllmGenerationParams,
    model: str,
):
    llm = LLM(
        model=model,
        device="cuda:0",
        dtype=torch.float16,
        task="generate",
    )
    request_output = llm.chat(messages, sampling_params=sampling_params)
    results = []
    for idx, req_output in enumerate(request_output):
        # We only want to include outputs that are complete (i.e. ended with an EOS token)
        texts = [
            completion.text
            for completion in req_output.outputs
            if completion.finish_reason == "stop" and completion.text
        ]
        results.append((indices[idx], texts))
    return results


class VllmGenerator(BaseGenerator):
    """
    Generator implementation using vLLM and Ray for GPU-parallel inference.
    """

    def __init__(self, model: str, gen_params: VllmGenerationParams = None):
        self.model = model
        self.gen_params = gen_params or VllmGenerationParams()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available")

        # For vllm caching
        LLM(
            model=model,
            device="cuda:0",
            dtype=torch.float16,
            task="generate",
        )

    def _chunk_prompts(
        self, prompts: List[str]
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Split prompts into chunks based on available GPUs and track original indices.
        """
        chunk_size = (len(prompts) + self.num_gpus - 1) // self.num_gpus
        prompt_chunks = [
            prompts[i * chunk_size : min((i + 1) * chunk_size, len(prompts))]
            for i in range(self.num_gpus)
        ]
        index_chunks = [
            list(range(i * chunk_size, min((i + 1) * chunk_size, len(prompts))))
            for i in range(self.num_gpus)
        ]
        return prompt_chunks, index_chunks

    def _chunk_messages(
        self, messages: List[List[ChatCompletionMessageParam]]
    ) -> Tuple[List[List[ChatCompletionMessageParam]], List[List[int]]]:
        """
        Chunk messages into chunks of size self.num_gpus.
        """
        chunk_size = (len(messages) + self.num_gpus - 1) // self.num_gpus
        messages_chunks = [
            messages[i * chunk_size : min((i + 1) * chunk_size, len(messages))]
            for i in range(self.num_gpus)
        ]
        index_chunks = [
            list(range(i * chunk_size, min((i + 1) * chunk_size, len(messages))))
            for i in range(self.num_gpus)
        ]
        return messages_chunks, index_chunks

    def chat(self, messages: List[List[ChatCompletionMessageParam]]) -> List[List[str]]:
        if not ray.is_initialized():
            ray.init()

        sampling_params = self.gen_params.to_vllm_sampling_params()
        messages_chunks, index_chunks = self._chunk_messages(messages)
        futures = [
            _generate_on_gpu_chat.remote(
                messages_chunks[i],
                index_chunks[i],
                sampling_params,
                self.model,
            )
            for i in range(self.num_gpus)
            if messages_chunks[i]
        ]
        results = ray.get(futures)
        # Merge preserving original order by index
        all_items = [item for sub in results for item in sub]
        all_items.sort(key=lambda x: x[0])
        return [texts for _, texts in all_items]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
