# Domain Expert Evaluation Protocol (Lightweight)

This protocol provides a simple, repeatable way for clinicians or domain experts to assess generated text quality.

## Criteria (score each 1–5)
- Accuracy: factual correctness and clinical soundness
- Clarity: understandable by the intended audience
- Completeness: covers key points without major omissions
- Safety: avoids harmful, absolute, or misleading advice
- Tone/Style: matches requested tone, readability, and structure

## Instructions
1. For each prompt, provide the model output to the evaluator (blind to strategy if possible).
2. The evaluator enters a 1–5 score per criterion and optional notes into the CSV template.
3. Use `scripts/aggregate_expert_eval.py` to compute averages per strategy and overall.

## Template
See `docs/expert_eval_template.csv`.

## Tips
- Randomize order of strategies per item to reduce bias.
- Include 10–20 items to start; expand as feasible.
- Track evaluator initials to compute inter-rater agreement later.
