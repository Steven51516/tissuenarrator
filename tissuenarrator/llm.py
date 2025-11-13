from vllm import LLM, SamplingParams

class VLLMWrapper:
    """Wrapper for inference using the vLLM backend."""

    def __init__(self, model_path: str, max_length: int = 32000):
        """
        Initialize the vLLM model.

        Args:
            model_path: Path to the model or model name.
            max_length: Maximum context length for the model.
        """
        self.llm = LLM(model=model_path, max_model_len=max_length)
        self.max_model_len = max_length

    def preprocess(self, prompts):
        """Ensure input is a list of strings."""
        return [prompts] if isinstance(prompts, str) else prompts

    def _make_sampling_params(
        self,
        max_new_tokens: int = 400,
        greedy: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 10,
        min_p: float = 0.0,
        seed: int | None = None,
    ) -> SamplingParams:
        """
        Create SamplingParams for text generation.

        Args:
            max_new_tokens: Number of tokens to generate.
            greedy: If True, use deterministic decoding.
            temperature, top_p, top_k, min_p: Sampling parameters for non-greedy mode.
            seed: Random seed for reproducibility during sampling.

        Returns:
            A SamplingParams object.
        """
        if greedy:
            return SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        return SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed if seed is not None else 42,
        )

    def inference(self, prompt: str, **kwargs) -> str:
        """Run inference on a single prompt."""
        prompts = self.preprocess(prompt)
        sampling_params = self._make_sampling_params(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs[0].outputs[0].text.strip()

    def inference_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Run inference on a batch of prompts."""
        prompts = self.preprocess(prompts)
        sampling_params = self._make_sampling_params(**kwargs)
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
