"""Progressive Self-Distillation (PSD) training script."""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_train_dataset, load_aime_dataset, load_math500_dataset, MathDataCollator
from src.psd_trainer import PSDTrainer, PSDConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--n_train_samples", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--n_rollouts", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--output_dir", type=str, default="outputs/psd")
    p.add_argument("--eval_dataset", type=str, default="aime2024")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_prompt_len", type=int, default=512)
    # PSD-specific
    p.add_argument("--buffer_size", type=int, default=5,
                   help="Max correct rollouts to keep per problem")
    p.add_argument("--buffer_strategy", type=str, default="random",
                   choices=["random", "latest", "shortest"],
                   help="How to sample from the buffer")
    p.add_argument("--buffer_path", type=str, default=None,
                   help="Path to persist/restore buffer (default: <output_dir>/buffer.json)")
    return p.parse_args()


def load_config(args):
    cfg = vars(args)
    if args.config:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f)
        cfg.update(file_cfg)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)

    accelerator = Accelerator(gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4))

    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        if cfg.get("wandb_project"):
            import wandb
            run_name = cfg.get("wandb_run_name") or f"psd_{Path(cfg['output_dir']).name}"
            wandb.init(project=cfg["wandb_project"], config=cfg, name=run_name)

    model_name = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01)
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    buffer_path = cfg.get("buffer_path") or os.path.join(cfg["output_dir"], "buffer.json")

    psd_config = PSDConfig(
        lr=cfg["lr"],
        n_rollouts=cfg["n_rollouts"],
        batch_size=cfg["batch_size"],
        max_completion_length=cfg["max_completion_length"],
        beta=cfg.get("beta", 0.0),
        temperature=cfg.get("temperature", 1.1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        max_steps=cfg["max_steps"],
        eval_steps=cfg["eval_steps"],
        max_prompt_len=cfg.get("max_prompt_len", 512),
        output_dir=cfg["output_dir"],
        wandb_project=cfg.get("wandb_project"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        buffer_size=cfg.get("buffer_size", 5),
        buffer_strategy=cfg.get("buffer_strategy", "random"),
        buffer_path=buffer_path,
    )

    trainer = PSDTrainer(model, tokenizer, optimizer, accelerator, psd_config)

    train_data = load_train_dataset(cfg["dataset"], n_samples=cfg["n_train_samples"])
    def _load_eval(name):
        if name == "aime2024":
            return load_aime_dataset([2024])
        if name == "aime2025":
            return load_aime_dataset([2025])
        if name == "math500":
            return load_math500_dataset()
        raise ValueError(f"unknown eval dataset: {name}")

    # --eval_dataset may be a comma-separated list, e.g. "aime2024,math500"
    eval_sets = {n.strip(): _load_eval(n.strip())
                 for n in cfg["eval_dataset"].split(",") if n.strip()}

    if accelerator.is_main_process:
        sizes = {k: len(v) for k, v in eval_sets.items()}
        print(
            f"PSD | model={model_name} | train={len(train_data)} | eval={sizes} "
            f"| buffer_size={psd_config.buffer_size} | strategy={psd_config.buffer_strategy}"
        )

    collator = MathDataCollator(tokenizer, max_prompt_len=cfg.get("max_prompt_len", 512))
    loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collator)
    loader = accelerator.prepare(loader)
    loader_iter = iter(loader)

    results = []

    for step in range(cfg["max_steps"]):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        metrics = trainer.train_step(batch)

        if accelerator.is_main_process:
            hit = metrics.get("buffer_hit_rate", 0.0)
            fill = metrics.get("buffer_fill", 0.0)
            print(
                f"Step {step}: loss={metrics['loss']:.4f}, reward={metrics['reward_mean']:.3f}, "
                f"buf_hit={hit:.1%}, buf_fill={fill:.1%}"
            )
            if cfg.get("wandb_project"):
                import wandb
                wandb.log({"train/" + k: v for k, v in metrics.items()}, step=step)

        if eval_sets and (step + 1) % cfg["eval_steps"] == 0:
            step_result = {"step": step + 1}
            log_metrics = {}
            for name, data in eval_sets.items():
                eval_subset = data[:min(30, len(data))]
                em = trainer.evaluate(eval_subset)  # runs on all ranks
                if accelerator.is_main_process:
                    print(f"Step {step+1} eval[{name}]: pass@1={em['pass@1']:.3f}")
                    for k, v in em.items():
                        step_result[f"{name}/{k}"] = v
                        log_metrics[f"eval/{name}/{k}"] = v
            if accelerator.is_main_process:
                results.append(step_result)
                with open(os.path.join(cfg["output_dir"], "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
                if cfg.get("wandb_project"):
                    import wandb
                    wandb.log(log_metrics, step=step + 1)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
        trainer._save_buffer(buffer_path)
        print(f"PSD model saved to {cfg['output_dir']}")
        print(f"Buffer saved to {buffer_path}")


if __name__ == "__main__":
    main()
