"""Clinical Notes Generation task implementation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ClinicalNote:
    """Container for clinical note generation results."""

    note: str
    score: float
    metadata: Optional[Dict] = None


class ClinicalNotesGenerator(nn.Module):
    """Clinical Notes Generation model."""

    def __init__(self, model_name: str = "gpt2", **kwargs):
        """Initialize the Clinical Notes Generator.

        Args:
            model_name: Pretrained model name or path
            **kwargs: Additional arguments for the model
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens for clinical note sections
        self.section_tokens = [
            "<CHIEF_COMPLAINT>",
            "<HISTORY_PRESENT_ILLNESS>",
            "<REVIEW_OF_SYSTEMS>",
            "<PHYSICAL_EXAM>",
            "<ASSESSMENT_AND_PLAN>",
            "<DIAGNOSIS>",
            "<TREATMENT_PLAN>",
            "<FOLLOW_UP>",
        ]

        # Add special tokens to the tokenizer
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.section_tokens}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for note generation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            labels: Target token IDs for training

        Returns:
            Dictionary with logits and loss
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss,
            "hidden_states": outputs.hidden_states,
        }

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> List[ClinicalNote]:
        """Generate clinical notes from prompts.

        Args:
            prompts: Input prompts or sections
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate per prompt
            device: Device to run inference on

        Returns:
            List of generated clinical notes
        """
        self.eval()
        self.to(device)

        # Convert single prompt to list if needed
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize inputs
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                no_repeat_ngram_size=3,
            )

        # Decode generated sequences
        generated_notes = []

        for i in range(len(prompts)):
            for j in range(num_return_sequences):
                seq_idx = i * num_return_sequences + j
                gen_text = self.tokenizer.decode(
                    outputs[seq_idx], skip_special_tokens=True
                )

                # Calculate sequence probability (approximate)
                with torch.no_grad():
                    logits = self.model(outputs[seq_idx].unsqueeze(0)).logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_probs = torch.gather(
                        log_probs[:, :-1],
                        -1,
                        outputs[seq_idx][1:].unsqueeze(0).unsqueeze(-1),
                    ).squeeze(-1)
                    avg_log_prob = token_probs.mean().exp().item()

                # Create clinical note object
                note = ClinicalNote(
                    note=gen_text,
                    score=avg_log_prob,
                    metadata={
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "prompt": prompts[i],
                    },
                )
                generated_notes.append(note)

        return generated_notes

    def generate_structured_note(
        self,
        patient_info: Dict[str, str],
        max_section_length: int = 512,
        **generation_kwargs,
    ) -> Dict[str, str]:
        """Generate a structured clinical note with multiple sections.

        Args:
            patient_info: Dictionary with patient information
            max_section_length: Maximum tokens per section
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with generated note sections
        """
        sections = {}

        # Generate each section
        for section in self.section_tokens:
            section_name = section.strip("<>").lower()

            # Create prompt for this section
            prompt = f"{section} Patient: {patient_info.get('name', 'Unknown')}\n"

            # Add relevant context
            if "history" in section_name:
                prompt += f"Age: {patient_info.get('age', 'N/A')}\n"
                prompt += f"Sex: {patient_info.get('sex', 'N/A')}\n"
                prompt += (
                    f"Presenting complaint: {patient_info.get('complaint', 'N/A')}\n"
                )

            # Generate section
            generated = self.generate(
                prompt, max_length=max_section_length, **generation_kwargs
            )

            if generated:
                sections[section_name] = generated[0].note

        return sections
