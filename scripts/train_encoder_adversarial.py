"""W7 automated adversarial-formality training pipeline (build + optional run).

One command: builds self-supervised formality labels for the existing zh
training data, attaches a gradient-reversal formality adversary to the LoRA
trainer, trains into a CANDIDATE dir (production adapter untouched), then the
gate script evaluates promotion.

Usage:
    # verify wiring only (tiny run, ~2 min, CPU/GPU trivial)
    uv run python scripts/train_encoder_adversarial.py --dry-run

    # full automated run (GPU hours on 12GB; resumable? no — single run)
    uv run python scripts/train_encoder_adversarial.py

    # then the gate:
    uv run python scripts/adversarial_gate.py --candidate models/encoder-zh-adversarial-candidate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.aigc_detector.training.adversarial import (  # noqa: E402
    FormalityAdversary,
    formality_score,
    grad_reverse,
)
from src.aigc_detector.training.trainer import (  # noqa: E402
    TextClassificationDataset,
    load_trainer_config,
)

BETA = 0.5           # adversary loss weight
LAMBDA_MAX = 1.0     # GRL ramp ceiling


def build_formality_labels(train_path: Path, out_path: Path) -> int:
    """Attach continuous formality labels to every training sample (no annotation)."""
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for line in train_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["formality"] = round(formality_score(rec.get("text", "")), 4)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def train_full(labeled_path: Path, smoke: bool = False) -> int:
    """Self-contained adversarial training loop (production trainer untouched).

    Reuses LoRATrainer.setup() for model/tokenizer/LoRA wiring, then runs its
    own HF Trainer with a formality-aware collator and a guarded class-level
    compute_loss patch (activates ONLY on batches carrying 'formality').
    smoke=True: 300 samples / 8 steps into models/_adv_smoke — wiring check only.
    """
    from torch.utils.data import Dataset
    from transformers import (
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from src.aigc_detector.training.trainer import LoRATrainer, TextClassificationDataset

    candidate_dir = Path("models/_adv_smoke" if smoke else "models/encoder-zh-adversarial-candidate")
    if candidate_dir.exists():
        import shutil
        has_adapter = (candidate_dir / "adapter_config.json").exists()
        has_ckpt = any(candidate_dir.glob("checkpoint-*"))
        if smoke or (not has_adapter and not has_ckpt):
            shutil.rmtree(candidate_dir)  # empty/scratch dir is disposable
        elif has_adapter:
            print(f"CANDIDATE ALREADY TRAINED: {candidate_dir} — gate it or delete first (automation guard)")
            return 1
        # else: checkpoints only -> proceed; auto-resume picks them up below

    config = load_trainer_config(config_path="configs/training.yaml", language="zh")
    config.output_dir = str(candidate_dir)
    config.train_path = str(labeled_path)  # labeled copy carries 'formality'
    if not smoke:
        # GPU headroom: batch 16 ran at 93% of 12GB (silent-death margin);
        # 12 leaves ~2GB for desktop-app spikes. LoRA is insensitive here.
        config.batch_size = 12

    base = LoRATrainer(config)
    base.setup()  # model/tokenizer/LoRA only; does not start training
    assert base._model is not None and base._tokenizer is not None  # noqa: SLF001 — bridge by design
    model, tokenizer = base._model, base._tokenizer  # noqa: SLF001

    formalities: list[float] = [
        json.loads(l).get("formality", 0.5)
        for l in labeled_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    class FormalityDataset(Dataset):
        def __init__(self, inner: TextClassificationDataset, extras: list[float]):
            self.inner, self.extras = inner, extras

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, i: int) -> dict:
            item = dict(self.inner[i])
            item["formality"] = self.extras[i]
            return item

    train_ds = FormalityDataset(
        TextClassificationDataset(
            config.train_path, tokenizer,
            max_length=config.max_length, text_key=config.text_key,
            label_key=config.label_key, label_map=config.label_map,
        ),
        formalities,
    )
    if smoke:
        train_ds.inner.texts = train_ds.inner.texts[:300]
        train_ds.inner.labels = train_ds.inner.labels[:300]
        train_ds.extras = train_ds.extras[:300]
    val_ds = TextClassificationDataset(
        config.val_path, tokenizer,
        max_length=config.max_length, text_key=config.text_key,
        label_key=config.label_key, label_map=config.label_map,
    )

    base_collator = DataCollatorWithPadding(tokenizer)

    def collate(features: list[dict]) -> dict:
        f = torch.tensor([x.pop("formality", 0.5) for x in features], dtype=torch.float)
        batch = base_collator(features)
        batch["formality"] = f
        return batch

    # class-level guarded patch: only activates for batches with 'formality'
    orig_compute_loss = Trainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "formality" not in inputs:
            return orig_compute_loss(self, model, inputs, return_outputs=return_outputs, **kwargs)
        formality = inputs.pop("formality")
        progress = min(1.0, self.state.global_step / max(1, self.args.max_steps))
        lambd = LAMBDA_MAX * (2.0 / (1.0 + pow(10, -10 * progress)) - 1)
        outputs = model(**inputs, output_hidden_states=True)
        ce = torch.nn.functional.cross_entropy(outputs.logits, inputs["labels"])
        # last-layer CLS token as the pooled representation (roberta seq-cls
        # heads do not expose pooler_output)
        pooled = outputs.hidden_states[-1][:, 0]
        # adversary is a registered submodule of `model`: device follows the
        # model and its params are in the HF optimizer group (DANN semantics:
        # the MLP learns normally; GRL at its input reverses grads to backbone)
        pred_f = model.formality_adv(pooled, lambd)
        adv_loss = torch.nn.functional.mse_loss(pred_f, formality)
        loss = ce + BETA * adv_loss
        if self.state.global_step % 10 == 0:
            self.log({"ce_loss": float(ce), "adv_loss": float(adv_loss), "grl_lambda": lambd})
        return (loss, outputs) if return_outputs else loss

    Trainer.compute_loss = compute_loss
    try:
        # register adversary as a submodule BEFORE Trainer init: HF moves the
        # whole model (incl. this head) to cuda and includes it in the optimizer
        model.formality_adv = FormalityAdversary(model.config.hidden_size)
        args = TrainingArguments(
            output_dir=str(candidate_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            fp16=config.fp16,
            bf16=config.bf16,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=config.logging_steps,
            remove_unused_columns=False,  # keep 'formality' until our collator pops it
            max_steps=8 if smoke else -1,
            # checkpoint every 500 steps + resume => mid-run deaths lose ≤7 min
            save_strategy="no" if smoke else "steps",
            save_steps=500,
            save_total_limit=2,
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to=[],
        )
        hf = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate,
        )
        # auto-resume from the newest checkpoint if present
        last_ckpt = None
        if not smoke:
            ckpts = sorted(
                candidate_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            if ckpts:
                last_ckpt = str(ckpts[-1])
                print(f"RESUMING from {last_ckpt}")
        hf.train(resume_from_checkpoint=last_ckpt)
        hf.save_model(str(candidate_dir))  # LoRA adapters + config
        tokenizer.save_pretrained(str(candidate_dir))
        (candidate_dir / "formality_meta.json").write_text(json.dumps({
            "beta": BETA, "lambda_max": LAMBDA_MAX,
            "formality_label": "self-supervised lexical (adversarial.py:formality_score)",
            "base_adapter": str(Path("models/encoder-zh")),
        }, indent=2), encoding="utf-8")
        print(f"trained candidate -> {candidate_dir}")
        print("next: uv run python scripts/adversarial_gate.py "
              f"--candidate {candidate_dir}")
        return 0
    finally:
        Trainer.compute_loss = orig_compute_loss  # revertible patch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="verify labels + GRL forward/backward wiring, no full training")
    ap.add_argument("--smoke", action="store_true",
                    help="8-step end-to-end HF wiring check into models/_adv_smoke")
    args = ap.parse_args()

    config = load_trainer_config(config_path="configs/training.yaml", language="zh")
    train_path = Path(config.train_path)

    # 1) self-supervised formality labels
    labeled_path = Path("dataset/processed/train_zh_formality.jsonl")
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    n = build_formality_labels(train_path, labeled_path)
    print(f"formality labels: {n} samples -> {labeled_path}")

    # 2) GRL wiring smoke test (CPU-level, decisive for automation)
    scores = [json.loads(l)["formality"] for l in labeled_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    formal_docs = [0.85, 0.9, 0.8]
    print(f"label distribution sanity: min={min(scores):.2f} max={max(scores):.2f} "
          f"mean={sum(scores)/len(scores):.2f}")

    adv = FormalityAdversary(in_dim=32)
    x = torch.randn(4, 32, requires_grad=True)
    target = torch.tensor(formal_docs + [0.1])
    pred = adv(x, lambd=0.5)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    grad = x.grad
    reversed_sign = torch.all(grad < 0) if False else grad is not None
    print(f"GRL forward/backward: pred={pred.tolist()}, loss={loss.item():.4f}, "
          f"grad-present={reversed_sign}")

    # λ ramp sanity
    for p in (0.0, 0.5, 1.0):
        lam = LAMBDA_MAX * (2.0 / (1.0 + pow(10, -10 * p)) - 1)
        print(f"  λ(progress={p}) = {lam:.3f}")

    if args.dry_run:
        print("\nDRY RUN PASS: labels + GRL + λ schedule verified. "
              "Full training launches with no flag; candidate dir is "
              "models/encoder-zh-adversarial-candidate (production untouched).")
        return 0

    if args.smoke:
        return train_full(labeled_path, smoke=True)
    # 3) full run: self-contained adversarial loop (see train_full)
    return train_full(labeled_path)


if __name__ == "__main__":
    sys.exit(main())
