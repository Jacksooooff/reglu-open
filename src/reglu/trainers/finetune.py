from __future__ import annotations

from reglu.artifacts import append_metrics, ensure_run_layout, write_config_snapshot, write_summary
from reglu.config import RunConfig
from reglu.data.common import custom_data_collator
from reglu.data.tofu import TofuTextDataset
from reglu.trainers.common import load_model, load_tokenizer, set_seed


def run_finetune(config: RunConfig) -> dict:
    layout = ensure_run_layout(config.runtime.output_dir)
    write_config_snapshot(layout["config"], config)
    set_seed(config.training.seed)
    summary = {
        "command": "finetune",
        "task": config.task,
        "model_family": config.model_family,
        "output_dir": str(layout["root"]),
        "status": "dry_run" if config.runtime.dry_run else "ok",
    }
    if config.runtime.dry_run:
        write_summary(layout["summary"], summary)
        return summary

    from reglu.methods.reglu import build_lora_model, build_training_arguments

    tokenizer = load_tokenizer(config)
    model = load_model(config)
    model = build_lora_model(model, config)
    model.config.use_cache = False
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    dataset = TofuTextDataset(
        data_path=config.data.path,
        tokenizer=tokenizer,
        model_family=config.model_family,
        max_length=config.data.max_length,
        split=str(config.data.split),
        question_key=config.data.question_key,
        answer_key=config.data.answer_key,
        subset_indices_file=config.data.subset_indices_file,
    )
    training_args = build_training_arguments(config, str(layout["root"]), len(dataset))
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    if config.training.eval_only:
        train_result = trainer.evaluate()
    else:
        train_result = trainer.train()
    if config.training.save_model and not config.training.eval_only:
        from peft import PeftModel

        model_to_save = trainer.model
        if isinstance(model_to_save, PeftModel):
            model_to_save = model_to_save.merge_and_unload()
        model_to_save.save_pretrained(str(layout["root"]))
        tokenizer.save_pretrained(str(layout["root"]))
        model_to_save.save_pretrained(str(layout["checkpoints"] / "checkpoint-last"))
        tokenizer.save_pretrained(str(layout["checkpoints"] / "checkpoint-last"))
    append_metrics(
        layout["metrics"],
        {
            "event": "train_complete",
            "train_loss": float(getattr(train_result, "training_loss", 0.0)) if hasattr(train_result, "training_loss") else 0.0,
            "num_examples": len(dataset),
        },
    )
    summary.update(
        {
            "num_examples": len(dataset),
            "train_loss": float(getattr(train_result, "training_loss", 0.0)) if hasattr(train_result, "training_loss") else 0.0,
        }
    )
    if config.training.save_model and not config.training.eval_only:
        summary["checkpoint_dir"] = str(layout["checkpoints"] / "checkpoint-last")
    write_summary(layout["summary"], summary)
    return summary
