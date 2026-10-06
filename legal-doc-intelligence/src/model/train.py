"""Fine-tune BART on the section-aligned IN-ABS dataset (chunk_text -> aligned_target).

Reads the output of ``scripts/build_dataset.py`` - either a local build
directory containing ``train.jsonl`` / ``validation.jsonl`` or a Hub dataset
repo - and trains ``config.BART_MODEL`` with ``Seq2SeqTrainer`` using the
recipe in ``Documents/Dataset_and_Training_Guide.md`` §3.3.

Validation ROUGE reported here is **chunk-level** (prediction vs pseudo-target)
and is only a proxy used for checkpoint selection. The number that goes in the
report is document-level - see ``src/eval/evaluate.py``.

Usage
-----
    # smoke test: a few steps on a slice, to check the plumbing and memory
    python -m src.model.train --data artifacts/training_data/full \
        --output_dir models/bart-smoke --max_train_samples 64 --max_eval_samples 16 --epochs 1

    # real run
    python -m src.model.train --data artifacts/training_data/full --output_dir models/legallens-bart-base

    # from the Hub instead of a local build
    python -m src.model.train --data <HF_USERNAME>/in-abs-section-aligned --output_dir ...

Then point inference at the result:
    LEGAL_BART_MODEL=models/legallens-bart-base python cli.py summarize artifacts/sample_judgment.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from config import BART_MODEL, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, NUM_BEAMS  # noqa: E402

logger = logging.getLogger("train")

INPUT_FIELD = "chunk_text"
TARGET_FIELD = "aligned_target"


def load_splits(data: str, max_train: int | None, max_eval: int | None):
    """Load train/validation from a local build directory or a Hub dataset repo."""
    from datasets import load_dataset

    path = Path(data)
    if path.is_dir():
        files = {split: str(path / f"{split}.jsonl") for split in ("train", "validation")
                 if (path / f"{split}.jsonl").exists()}
        if "train" not in files:
            raise SystemExit(f"No train.jsonl in {path} - run scripts/build_dataset.py first.")
        ds = load_dataset("json", data_files=files)
    else:
        ds = load_dataset(data)

    if "validation" not in ds:
        raise SystemExit(
            "No validation split found. Checkpoint selection needs one - build the "
            "dataset with all splits (the default) rather than --split train."
        )

    ds = ds.select_columns([INPUT_FIELD, TARGET_FIELD])
    if max_train:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    if max_eval:
        ds["validation"] = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))
    return ds


def tokenize_splits(ds, tokenizer, max_input: int, max_target: int):
    """Tokenize and report how many examples the length limits truncate."""

    def encode(batch):
        model_inputs = tokenizer(batch[INPUT_FIELD], max_length=max_input, truncation=True)
        labels = tokenizer(text_target=batch[TARGET_FIELD], max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        # Untruncated lengths, for the truncation report only.
        model_inputs["n_input"] = [len(x) for x in tokenizer(batch[INPUT_FIELD])["input_ids"]]
        model_inputs["n_target"] = [len(x) for x in tokenizer(text_target=batch[TARGET_FIELD])["input_ids"]]
        return model_inputs

    tokenized = ds.map(encode, batched=True, remove_columns=ds["train"].column_names)

    report = {}
    for split in tokenized:
        n_in, n_tg = tokenized[split]["n_input"], tokenized[split]["n_target"]
        report[split] = {
            "examples": len(n_in),
            "inputs_truncated": sum(n > max_input for n in n_in),
            "targets_truncated": sum(n > max_target for n in n_tg),
            "max_input_tokens": max(n_in),
            "max_target_tokens": max(n_tg),
        }
        logger.info("%s: %s", split, report[split])
        if report[split]["examples"] and report[split]["inputs_truncated"] / report[split]["examples"] > 0.05:
            logger.warning(
                "%s: more than 5%% of inputs exceed %d tokens - lower config.CHUNK_SIZE and "
                "rebuild rather than training on truncated text (guide §3.2).", split, max_input,
            )

    return tokenized.remove_columns(["n_input", "n_target"]), report


def make_compute_metrics(tokenizer):
    from rouge_score import rouge_scorer
    import numpy as np

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(decoded_preds, decoded_labels):
            # rougeLsum expects newline-separated sentences.
            scores = scorer.score("\n".join(ref.split(". ")), "\n".join(pred.split(". ")))
            totals["rouge1"] += scores["rouge1"].fmeasure
            totals["rouge2"] += scores["rouge2"].fmeasure
            totals["rougeL"] += scores["rougeLsum"].fmeasure
        n = max(len(decoded_preds), 1)
        result = {k: round(100 * v / n, 2) for k, v in totals.items()}
        result["gen_len"] = round(float(np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds])), 1)
        return result

    return compute_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="Build directory (train.jsonl/validation.jsonl) or Hub dataset id")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default=BART_MODEL)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-device batch. 4 fits bart-base at 1024 tokens on 8 GB with gradient checkpointing.")
    p.add_argument("--grad_accum", type=int, default=8, help="Effective batch = batch_size x grad_accum (default 32)")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH)
    p.add_argument("--max_target_length", type=int, default=MAX_TARGET_LENGTH)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2, help="Early-stopping patience, in evaluations")
    p.add_argument("--max_train_samples", type=int)
    p.add_argument("--max_eval_samples", type=int, default=500,
                   help="Cap the validation set: generating with beams for every chunk each epoch is slow.")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16
    if not use_cuda:
        logger.warning("No CUDA device - training on CPU will be very slow. Use Colab (guide §3.3).")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = load_splits(args.data, args.max_train_samples, args.max_eval_samples)
    tokenized, length_report = tokenize_splits(ds, tokenizer, args.max_input_length, args.max_target_length)

    output_dir = Path(args.output_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        bf16=bf16,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=NUM_BEAMS,
        logging_steps=25,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    train_result = trainer.train()
    final_metrics = trainer.evaluate()

    # Save the best checkpoint (load_best_model_at_end) at the top of output_dir,
    # which is what LEGAL_BART_MODEL should point at.
    model.config.use_cache = True
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "data": args.data,
        "args": vars(args),
        "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
        "length_report": length_report,
        "train_metrics": train_result.metrics,
        "eval_metrics_chunk_level": final_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "note": "Validation ROUGE is chunk-level against pseudo-targets; report document-level numbers from src/eval.",
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Saved model to %s (best checkpoint %s)", output_dir, trainer.state.best_model_checkpoint)
    logger.info("Chunk-level validation: %s", final_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
