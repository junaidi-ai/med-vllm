import atexit
import logging
from dataclasses import fields
from time import perf_counter
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from medvllm.config import Config
from medvllm.engine.model_runner.base import ModelRunner
from medvllm.engine.scheduler import Scheduler
from medvllm.engine.sequence import Sequence
from medvllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, model: str, **kwargs: Any) -> None:
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # Initialize model runner
        self.model_runner = ModelRunner(
            config=config, rank=0, world_size=config.tensor_parallel_size
        )

        # Start additional processes for tensor parallelism if needed
        if config.tensor_parallel_size > 1:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(
                    target=ModelRunner,
                    args=(config, i, config.tensor_parallel_size, event),
                )
                process.start()
                self.ps.append(process)
                self.events.append(event)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self) -> None:
        """Clean up resources and terminate all processes."""
        if hasattr(self, "model_runner") and self.model_runner is not None:
            self.model_runner.cleanup()
            del self.model_runner

        # Terminate any remaining processes
        for p in self.ps:
            if p.is_alive():
                p.terminate()
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

    def step(self) -> tuple[list[tuple[Any, list[int]]], int]:
        """Execute one step of the generation process.

        Returns:
            A tuple containing:
                - List of (sequence_id, token_ids) for completed sequences
                - Number of tokens processed (positive for prefill, negative for decode)
        """
        # Get scheduled sequences and prefill flag
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return [], 0

        # Prepare inputs for the model
        input_sequences: List[List[int]] = []
        for seq in seqs:
            # Ensure both prompt and completion tokens are lists of integers
            prompt_tokens = (
                seq.prompt_token_ids if hasattr(seq, "prompt_token_ids") else []
            )
            completion_tokens = (
                seq.completion_token_ids if hasattr(seq, "completion_token_ids") else []
            )
            input_sequences.append(prompt_tokens + completion_tokens)

        # Convert to tensor and move to device
        input_tensor = torch.tensor(input_sequences, device=self.model_runner.device)

        # Run the model
        output_token_ids = self.model_runner.run(input_tensor, is_prefill=is_prefill)

        # Initialize with empty lists for each sequence
        processed_token_ids: List[List[int]] = []
        for _ in seqs:
            processed_token_ids.append([])

        def _convert_to_token_list(token_data: Any) -> list[Any]:
            """Convert various token data formats to a list of tokens."""
            if isinstance(token_data, (list, tuple)):
                return list(token_data)
            if hasattr(token_data, "tolist") and callable(token_data.tolist):
                result = token_data.tolist()
                return result if isinstance(result, list) else [result]
            return [token_data]

        def process_token_value(token_val: Any) -> int | None:
            """Convert a single token value to an integer if possible."""
            if token_val is None:
                return None
            try:
                if isinstance(token_val, (int, float)):
                    return int(token_val)
                token_str = str(token_val).strip()
                if token_str:  # Only process non-empty strings
                    return int(float(token_str))
            except (TypeError, ValueError):
                pass
            return None

        # Process output tokens if we have any
        if output_token_ids is not None and seqs and processed_token_ids:
            # Convert output_token_ids to a list of tokens
            try:
                token_list = _convert_to_token_list(output_token_ids)

                # Process each token in the sequence
                for i in range(min(len(token_list), len(processed_token_ids))):
                    item = token_list[i]
                    if item is None:
                        continue

                    # Initialize list for processed tokens
                    processed_tokens: List[int] = []

                    # Process the token(s)
                    if isinstance(item, (list, tuple)):
                        # Handle sequence of tokens
                        for token in item:
                            if token is not None:
                                token_val = process_token_value(token)
                                if token_val is not None:
                                    processed_tokens.append(token_val)
                    else:
                        # Handle single token
                        token_val = process_token_value(item)
                        if token_val is not None:
                            processed_tokens.append(token_val)

                    # Only update if we have valid tokens
                    if processed_tokens:
                        processed_token_ids[i] = processed_tokens

            except Exception as e:
                logger.warning(f"Error processing output tokens: {e}")
                # Continue with whatever tokens we've processed so far

        # Ensure we have the right number of token lists
        seq_count = len(seqs)
        if len(processed_token_ids) < seq_count:
            # Add empty lists for missing sequences
            processed_token_ids.extend(
                [[] for _ in range(seq_count - len(processed_token_ids))]
            )
        elif len(processed_token_ids) > seq_count:
            # Truncate if we have too many
            processed_token_ids = processed_token_ids[:seq_count]

        # Update sequences with new tokens
        for seq, new_tokens in zip(seqs, processed_token_ids):
            if new_tokens and hasattr(seq, "completion_token_ids"):
                seq.completion_token_ids.extend(new_tokens)

        # Post-process completed sequences if the scheduler has a postprocess method
        if hasattr(self.scheduler, "postprocess") and seqs:
            try:
                # Get token_ids if available
                token_ids = processed_token_ids[0] if processed_token_ids else []
                # Call postprocess with both sequences and token_ids
                self.scheduler.postprocess(seqs, token_ids=token_ids)
            except Exception as e:
                print(f"Warning: Failed to postprocess sequences: {e}")

        # Prepare outputs for completed sequences
        outputs: List[Tuple[Any, List[int]]] = []
        for seq in seqs:
            if not hasattr(seq, "is_finished") or not seq.is_finished:
                continue

            if not hasattr(seq, "seq_id") or not hasattr(seq, "completion_token_ids"):
                continue

            # Ensure completion_token_ids is a List[int]
            completion_ids = getattr(seq, "completion_token_ids", [])
            if not isinstance(completion_ids, list):
                continue

            # Filter out any non-integer values
            filtered_ids = [x for x in completion_ids if isinstance(x, int)]
            outputs.append((seq.seq_id, filtered_ids))

        # Calculate number of processed tokens
        num_tokens = 0
        for tokens in processed_token_ids:
            num_tokens += len(tokens)

        # For prefill, return positive count; for decode, return negative count
        return outputs, num_tokens if is_prefill else -num_tokens

    def is_finished(self) -> bool:
        """Check if all sequences have finished processing.

        Returns:
            bool: True if all sequences are finished, False otherwise.
        """
        try:
            return bool(self.scheduler.is_finished())
        except Exception as e:
            # Log the error and return True to prevent infinite loops
            print(f"Error checking if finished: {e}")
            return True

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
