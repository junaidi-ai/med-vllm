import os
import time
import threading
from typing import Optional, Dict, Any

import gradio as gr

from huggingface_hub import HfApi, create_repo, hf_hub_url


DEFAULT_BASE_MODEL = "dmis-lab/biobert-base-cased-v1.1"
DEFAULT_DATASET = "conll2003"  # stronger baseline default
TARGET_REPO = os.getenv("MEDVLLM_TARGET_REPO", "Junaidi-AI/med-vllm")


def _train_ner_lora(
    base_model: str,
    dataset_name: str,
    output_dir: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    trust_dataset_scripts: bool = True,
    log_cb=None,
) -> Dict[str, Any]:
    """
    Minimal LoRA token-classification trainer.
    Uses conll2003 by default to be robust in Spaces. Extend to medical datasets later.
    """
    # Avoid importing any local dataset scripts even if present in working dir
    os.environ.setdefault("HF_DATASETS_DISABLE_LOCAL_IMPORTS", "1")
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
        Trainer,
    )
    from transformers.trainer_utils import set_seed
    from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
    from peft import LoraConfig, get_peft_model, TaskType

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    set_seed(42)

    ds_spec = (dataset_name or "").strip()
    log(f"Loading dataset: {ds_spec}")
    # Support optional config via 'name:config' (e.g., 'wikiann:en')
    try:
        # Medical aliases -> BigBio NER configs
        alias_map = {
            # BigBio script-based configs (preferred with datasets<3.0)
            "bc5cdr": ("bigbio/bc5cdr", "bigbio_ner"),
            "ncbi_disease": ("bigbio/ncbi_disease", "bigbio_ner"),
        }
        lower_spec = ds_spec.lower()
        if lower_spec in alias_map:
            repo_id, subset = alias_map[lower_spec]
            # 1) Try script loader first (requires datasets<3.0)
            try:
                log(f"Trying BigBio script loader: load_dataset('{repo_id}', '{subset}')")
                ds = load_dataset(repo_id, subset, trust_remote_code=trust_dataset_scripts)
            except Exception as e_script:
                log(f"Script loader failed: {e_script}")
                # 2) Fallback to Parquet discovery via HTTPS
                log("Falling back to Parquet discovery via refs/convert/parquet")
                api = HfApi()
                files = api.list_repo_files(
                    repo_id=repo_id, repo_type="dataset", revision="refs/convert/parquet"
                )

                def split_files(split: str):
                    shard_prefix = f"{subset}/{split}-"
                    dir_prefix = f"{subset}/{split}/"
                    out = []
                    for path in files:
                        if not path.endswith(".parquet"):
                            continue
                        if path.startswith(shard_prefix) or path.startswith(dir_prefix):
                            out.append(
                                hf_hub_url(
                                    repo_id=repo_id,
                                    filename=path,
                                    repo_type="dataset",
                                    revision="refs/convert/parquet",
                                )
                            )
                    return sorted(out)

                train_files = split_files("train")
                val_files = split_files("validation") or split_files("valid") or split_files("dev")
                test_files = split_files("test")
                if not train_files:
                    raise RuntimeError(
                        "No train parquet files found for BigBio subset; merge PR to pin datasets<3.0 or choose another dataset"
                    )
                data_files = {"train": train_files}
                if val_files:
                    data_files["validation"] = val_files
                if test_files:
                    data_files["test"] = test_files
                ds = load_dataset("parquet", data_files=data_files)
        elif ":" in ds_spec:
            ds_name, ds_config = [s.strip() for s in ds_spec.split(":", 1)]
            # Respect UI toggle for trusting dataset scripts
            trust = trust_dataset_scripts or ("/" in ds_name)
            ds = load_dataset(ds_name, ds_config, trust_remote_code=trust)
        else:
            trust = trust_dataset_scripts or ("/" in ds_spec)
            ds = load_dataset(ds_spec, trust_remote_code=trust)
    except Exception as e:
        # Fallback: if it looks like 'name:config' but was treated as a local path, try explicit two-arg call
        err_msg = str(e)
        log(f"Dataset load failed: {err_msg}")
        if ":" in ds_spec:
            try:
                ds_name, ds_config = [s.strip() for s in ds_spec.split(":", 1)]
                log(f"Retrying with split name/config: {ds_name}, {ds_config}")
                trust = trust_dataset_scripts or ("/" in ds_name)
                ds = load_dataset(ds_name, ds_config, trust_remote_code=trust)
            except Exception as e2:
                log(f"Retry failed: {e2}")
                raise
        else:
            raise

    if "train" not in ds:
        raise RuntimeError("Dataset must have a train split")

    # Detect token and label columns across common schemas
    features = ds["train"].features
    token_candidates = ["tokens", "words"]
    tag_candidates = ["ner_tags", "tags", "labels", "ner_tags_general"]
    token_col = next((c for c in token_candidates if c in features), None)
    tag_col = next((c for c in tag_candidates if c in features), None)
    if not token_col or not tag_col:
        raise RuntimeError(
            "Dataset must provide token and tag columns. Looked for tokens/words and ner_tags/tags/labels."
        )

    label_list = ds["train"].features[tag_col].feature.names
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    log(f"Loading tokenizer/model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(base, peft_config)

    # Tokenize with alignment
    def tokenize_align(batch):
        tokenized = tokenizer(
            batch[token_col], is_split_into_words=True, truncation=True, padding=False
        )
        # Build aligned labels per example
        new_input_ids = []
        new_labels = []
        for tokens, tags in zip(batch[token_col], batch[tag_col]):
            enc = tokenizer(tokens, is_split_into_words=True, truncation=True, padding=False)
            word_ids = enc.word_ids()
            lab = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    lab.append(-100)
                else:
                    tag_id = tags[wid]
                    # Only label first subword
                    if wid != prev_wid:
                        lab.append(tag_id)
                        prev_wid = wid
                    else:
                        lab.append(-100)
            new_input_ids.append(enc["input_ids"])  # unused but keeps shape; collator will pad
            new_labels.append(lab)
        enc = tokenizer(batch[token_col], is_split_into_words=True, truncation=True, padding=True)
        enc["labels"] = new_labels
        return enc

    log("Tokenizing dataset...")
    tokenized = ds.map(tokenize_align, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metrics_holder: Dict[str, float] = {}

    def compute_metrics(p):
        preds, labels = p
        preds = preds.argmax(-1)
        true_predictions = []
        true_labels = []
        for pred, lab in zip(preds, labels):
            curr_pred = []
            curr_lab = []
            for p_i, l_i in zip(pred, lab):
                if l_i != -100:
                    curr_pred.append(id2label[int(p_i)])
                    curr_lab.append(id2label[int(l_i)])
            true_predictions.append(curr_pred)
            true_labels.append(curr_lab)
        out = {
            "f1": f1_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
        metrics_holder.update(out)
        return out

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") or tokenized.get("dev") or tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    log("Starting training...")
    trainer.train()

    log("Saving adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Compose commit description with metrics
    desc_lines = [
        f"base_model: {base_model}",
        f"dataset: {dataset_name}",
        f"epochs: {num_train_epochs}",
        f"batch_size: {per_device_train_batch_size}",
        f"learning_rate: {learning_rate}",
        f"lora_r: {lora_r}",
        f"lora_alpha: {lora_alpha}",
        f"lora_dropout: {lora_dropout}",
        "",
        "metrics:",
        *(f"- {k}: {v:.4f}" for k, v in metrics_holder.items()),
    ]
    commit_description = "\n".join(desc_lines)

    # Push to the umbrella repo under checkpoints/
    api = HfApi()
    run_name = os.path.basename(output_dir.rstrip("/"))
    path_in_repo = f"checkpoints/ner-{run_name}"
    log(f"Pushing to {TARGET_REPO}:{path_in_repo}")
    commit = api.upload_folder(
        repo_id=TARGET_REPO,
        repo_type="model",
        folder_path=output_dir,
        path_in_repo=path_in_repo,
        commit_message=f"Add NER LoRA checkpoint ({run_name})",
        commit_description=commit_description,
        create_pr=True,
    )
    log(f"Pushed: {commit}")

    # Also publish to a dedicated med-vllm-* variant repo
    try:
        base_short = base_model.split("/")[-1].replace(" ", "-").lower()
        ds_short = dataset_name.split("/")[-1].replace(" ", "-").lower()
        variant_name = f"Junaidi-AI/med-vllm-ner-{ds_short}-{base_short}-lora-v1"
        log(f"Ensuring repo exists: {variant_name}")
        try:
            create_repo(repo_id=variant_name, repo_type="model", exist_ok=True, private=False)
        except Exception:
            pass
        commit2 = api.upload_folder(
            repo_id=variant_name,
            repo_type="model",
            folder_path=output_dir,
            path_in_repo=".",
            commit_message=f"Initial LoRA checkpoint from {base_model} on {dataset_name}",
            commit_description=commit_description,
            create_pr=False,
        )
        log(f"Variant published: {commit2}")
    except Exception as e:
        log(f"Warning: failed to publish variant repo: {e}")

    return {"commit": str(commit), "path_in_repo": path_in_repo, "metrics": metrics_holder}


class TrainerThread:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.logs = ""
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def _log(self, msg: str):
        self.logs += msg + "\n"

    def start(self, **kwargs):
        if self.thread and self.thread.is_alive():
            raise gr.Error("Training is already running")

        def target():
            try:
                self._log("Initializing training...")
                res = _train_ner_lora(log_cb=self._log, **kwargs)
                self.result = res
                self._log("Training complete")
            except Exception as e:
                self.error = str(e)
                self._log(f"ERROR: {e}")

        self.logs = ""
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def status(self):
        running = self.thread.is_alive() if self.thread else False
        return running, self.logs, self.result, self.error


TRAINER = TrainerThread()


def build_ui():
    with gr.Blocks(title="Med vLLM Train (LoRA NER)") as demo:
        gr.Markdown(
            f"""
            # Med vLLM Train (LoRA NER)
            This Space fine-tunes a token-classification model with LoRA.

            - Base model default: `{DEFAULT_BASE_MODEL}`
            - Dataset default: `{DEFAULT_DATASET}` (robust demo). Medical sets like `bc5cdr`/`ncbi_disease` may require custom preprocessing.
            - Checkpoints will be pushed to `{TARGET_REPO}` under `checkpoints/` as a PR.
            """
        )
        with gr.Row():
            base_model = gr.Textbox(value=DEFAULT_BASE_MODEL, label="Base model")
            dataset_name = gr.Dropdown(
                choices=[
                    "bc5cdr",
                    "ncbi_disease",
                    "wikiann:en",
                    "conll2003",
                ],
                value=DEFAULT_DATASET,
                allow_custom_value=True,
                label="Dataset (token classification)",
            )
        trust_scripts = gr.Checkbox(
            value=True, label="Trust dataset script (required for many HF datasets incl. conll2003)"
        )
        with gr.Row():
            epochs = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Epochs")
            batch = gr.Slider(minimum=4, maximum=16, step=2, value=8, label="Batch size")
            lr = gr.Textbox(value="2e-5", label="Learning rate")
        with gr.Row():
            lora_r = gr.Slider(minimum=4, maximum=32, step=2, value=8, label="LoRA r")
            lora_alpha = gr.Slider(minimum=8, maximum=64, step=8, value=16, label="LoRA alpha")
            lora_dropout = gr.Slider(
                minimum=0.0, maximum=0.5, step=0.05, value=0.1, label="LoRA dropout"
            )
        with gr.Row():
            run_name = gr.Textbox(value=f"run-{int(time.time())}", label="Run name (folder)")
        with gr.Row():
            start_btn = gr.Button("Start Training")
            status_btn = gr.Button("Refresh Status")
        logs = gr.Textbox(label="Logs", lines=18)
        result = gr.Textbox(label="Result / Commit info")

        def on_start(bm, ds, ep, bs, lr_s, r, alpha, drop, rn, trust):
            try:
                out_dir = os.path.join("outputs", rn)
                os.makedirs(out_dir, exist_ok=True)
                TRAINER.start(
                    base_model=bm,
                    dataset_name=ds,
                    output_dir=out_dir,
                    num_train_epochs=int(ep),
                    per_device_train_batch_size=int(bs),
                    learning_rate=float(lr_s),
                    lora_r=int(r),
                    lora_alpha=int(alpha),
                    lora_dropout=float(drop),
                    trust_dataset_scripts=bool(trust),
                )
                return "Started"
            except Exception as e:
                return f"ERROR starting: {e}"

        def on_status():
            running, l, res, err = TRAINER.status()
            info = "Running" if running else ("Error" if err else "Idle/Done")
            res_s = str(res) if res else ""
            return f"[{info}]\n" + l, res_s

        start_btn.click(
            on_start,
            inputs=[
                base_model,
                dataset_name,
                epochs,
                batch,
                lr,
                lora_r,
                lora_alpha,
                lora_dropout,
                run_name,
                trust_scripts,
            ],
            outputs=[logs],
        )
        status_btn.click(on_status, outputs=[logs, result])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
