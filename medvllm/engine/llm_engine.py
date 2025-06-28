import atexit
from dataclasses import fields
from time import perf_counter
from typing import Any, Dict, List, Union

import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

from medvllm.config import Config
from medvllm.engine.model_runner import ModelRunner
from medvllm.engine.scheduler import Scheduler
from medvllm.engine.sequence import Sequence
from medvllm.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self, prompt: str | list[int], sampling_params: SamplingParams
    ) -> None:
        """Add a new request to the engine's queue.

        Args:
            prompt: The input prompt as a string or list of token IDs.
            sampling_params: The sampling parameters for generation.

        Returns:
            None: This method adds the sequence to the scheduler's queue.
        """
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt)
            if not isinstance(prompt_ids, list):
                prompt_ids = prompt_ids.tolist()
            seq = Sequence(prompt_ids, sampling_params)
        elif isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
            seq = Sequence(prompt, sampling_params)
        else:
            raise ValueError("Prompt must be either a string or a list of integers")

        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate text from prompts.

        Args:
            prompts: List of prompts (either strings or token IDs).
            sampling_params: Sampling parameters for generation.
            use_tqdm: Whether to show a progress bar.

        Returns:
            List of dictionaries containing 'text' and 'token_ids' for each generated sequence.
        """
        # Initialize progress bar if enabled
        if use_tqdm:
            pbar = tqdm(desc="Generating", unit=" seq")
        else:
            pbar = None

        # Handle single sampling params case
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Add all requests to the scheduler
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            # Ensure prompt is either a string or list of ints before passing to add_request
            if isinstance(prompt, str):
                self.add_request(prompt, sp)
            elif isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
                self.add_request(prompt, sp)
            else:
                raise ValueError(
                    f"Prompt at index {i} must be either a string or a list of integers"
                )

        # Initialize outputs as a list of dictionaries
        outputs: list[dict[str, Any]] = []
        seq_outputs: dict[int, list[int]] = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0

        # Process sequences until all are finished
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            # Update progress bar if enabled
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            # Process the output from this step
            if output:  # Check if output is not None and not empty
                for seq_id, token_ids in output:
                    if seq_id is not None and token_ids is not None:
                        seq_outputs[seq_id] = token_ids
                        if use_tqdm:
                            pbar.update(1)
        # Convert sequence outputs to the final result format
        final_outputs = []
        for seq_id in sorted(seq_outputs):
            token_ids = seq_outputs[seq_id]
            final_outputs.append(
                {
                    "text": self.tokenizer.decode(token_ids),
                    "token_ids": token_ids,
                }
            )

        # Clean up and return
        if use_tqdm:
            pbar.close()

        return final_outputs
