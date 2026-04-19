from __future__ import annotations

import shutil

from reglu.artifacts import append_metrics, ensure_run_layout, write_config_snapshot, write_summary
from reglu.config import RunConfig
from reglu.data.common import custom_data_collator_unlearn
from reglu.data.tofu import TofuUnlearnDataset
from reglu.data.wmdp import WmdpUnlearnDataset
from reglu.trainers.common import load_model, load_tokenizer, set_seed


def run_unlearn(config: RunConfig) -> dict:
    layout = ensure_run_layout(config.runtime.output_dir)
    write_config_snapshot(layout["config"], config)
    set_seed(config.training.seed)
    summary = {
        "command": "forget",
        "task": config.task,
        "model_family": config.model_family,
        "variant": config.method.variant.upper(),
        "output_dir": str(layout["root"]),
        "status": "dry_run" if config.runtime.dry_run else "ok",
    }
    if config.runtime.dry_run:
        write_summary(layout["summary"], summary)
        return summary

    from reglu.methods.reglu import (
        RegLUUnlearnTrainer,
        build_lora_model,
        build_training_arguments,
        initialize_rila,
        load_rol_bases,
    )

    tokenizer = load_tokenizer(config)
    if config.task == "tofu":
        dataset = TofuUnlearnDataset(
            data_path=config.data.path,
            tokenizer=tokenizer,
            model_family=config.model_family,
            max_length=config.data.max_length,
            split=str(config.data.split),
            subset_indices_file=config.data.subset_indices_file,
        )
    else:
        dataset = WmdpUnlearnDataset(
            data_path=config.data.path,
            tokenizer=tokenizer,
            model_family=config.model_family,
            split=str(config.data.split),
            max_length=config.data.max_length,
            subset_indices_file=config.data.subset_indices_file,
        )

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
    rol_basis_dict = None
    resolved_rila_cache = None
    if config.method.init_strategy == "rila":
        initialized, init_rol_bases, resolved_rila_cache = initialize_rila(
            model=model,
            config=config,
            dataset=dataset,
            output_dir=str(layout["root"]),
            strict_cache=bool(config.method.require_rila_cache),
        )
        if not initialized and config.method.require_rila_cache:
            raise FileNotFoundError("init_strategy=rila requires a valid RILA cache or a successful online initialization.")
        if init_rol_bases:
            rol_basis_dict = init_rol_bases
        if resolved_rila_cache and resolved_rila_cache.is_file():
            shutil.copyfile(resolved_rila_cache, layout["rila_cache"])

    if rol_basis_dict is None and config.method.rila_cache_path:
        rol_basis_dict = load_rol_bases(
            config.method.rila_cache_path,
            config.method.rol_rank,
        )
    training_args = build_training_arguments(config, str(layout["root"]), len(dataset))
    trainer = RegLUUnlearnTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=custom_data_collator_unlearn,
        method_config=config.method,
        run_config=config,
        tokenizer=tokenizer,
        rol_basis_dict=rol_basis_dict,
    )
    if config.training.eval_only:
        train_result = trainer.evaluate()
    else:
        train_result = trainer.train()
    if config.training.save_model and not config.training.eval_only:
        trainer.save_model(str(layout["root"]))
        tokenizer.save_pretrained(str(layout["root"]))
        trainer.save_model(str(layout["checkpoints"] / "checkpoint-last"))
        tokenizer.save_pretrained(str(layout["checkpoints"] / "checkpoint-last"))
    append_metrics(
        layout["metrics"],
        {
            "event": "forget_complete",
            "train_loss": float(getattr(train_result, "training_loss", 0.0)) if hasattr(train_result, "training_loss") else 0.0,
            "num_examples": len(dataset),
            "variant": config.method.variant.upper(),
            "rol_lambda": config.method.rol_lambda,
        },
    )
    summary.update(
        {
            "num_examples": len(dataset),
            "train_loss": float(getattr(train_result, "training_loss", 0.0)) if hasattr(train_result, "training_loss") else 0.0,
        }
    )
    if isinstance(train_result, dict):
        summary["evaluation"] = train_result
    if config.training.save_model and not config.training.eval_only:
        summary["checkpoint_dir"] = str(layout["checkpoints"] / "checkpoint-last")
    write_summary(layout["summary"], summary)
    return summary
