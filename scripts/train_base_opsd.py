"""Base OPSD training script."""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_train_dataset, load_aime_dataset, load_math500_dataset, MathDataCollator
from src.opsd_base import OPSDTrainer, OPSDConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", type=str, default="AI-MO/NuminaMath-CoT")
    p.add_argument("--n_train_samples", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--n_rollouts", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--output_dir", type=str, default="outputs/base_opsd")
    p.add_argument("--eval_dataset", type=str, default="aime2024")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--max_prompt_len", type=int, default=512)
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

    accelerator = Accelerator(gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2))

    if accelerator.is_main_process:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        if cfg.get("wandb_project"):
            import wandb
            wandb.init(project=cfg["wandb_project"], config=cfg, name=os.path.basename(cfg["output_dir"]))

    model_name = cfg["model"]
    if accelerator.is_main_process:
        print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)

    opsd_config = OPSDConfig(
        lr=cfg["lr"],
        n_rollouts=cfg["n_rollouts"],
        batch_size=cfg["batch_size"],
        max_completion_length=cfg["max_completion_length"],
        beta=cfg.get("beta", 0.0),
        temperature=cfg.get("temperature", 1.1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        max_steps=cfg["max_steps"],
        eval_steps=cfg["eval_steps"],
        max_prompt_len=cfg.get("max_prompt_len", 512),
        output_dir=cfg["output_dir"],
        wandb_project=cfg.get("wandb_project"),
    )

    trainer = OPSDTrainer(model, tokenizer, optimizer, accelerator, opsd_config)

    # Load datasets
    train_data = load_train_dataset(cfg["dataset"], n_samples=cfg["n_train_samples"])
    eval_data = []
    if cfg["eval_dataset"] == "aime2024":
        eval_data = load_aime_dataset([2024])
    elif cfg["eval_dataset"] == "aime2025":
        eval_data = load_aime_dataset([2025])
    elif cfg["eval_dataset"] == "math500":
        eval_data = load_math500_dataset()

    if accelerator.is_main_process:
        print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    collator = MathDataCollator(tokenizer, max_prompt_len=cfg.get("max_prompt_len", 512))
    loader = DataLoader(
        train_data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
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

        if accelerator.is_main_process and step % 5 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}, reward={metrics['reward_mean']:.3f}")
            if cfg.get("wandb_project"):
                import wandb
                wandb.log({**metrics, "step": step})

        if eval_data and (step + 1) % cfg["eval_steps"] == 0:
            eval_subset = eval_data[:min(30, len(eval_data))]
            eval_metrics = trainer.evaluate(eval_subset)
            if accelerator.is_main_process:
                print(f"Step {step+1} eval: pass@1={eval_metrics['pass@1']:.3f}")
                results.append({"step": step + 1, **eval_metrics})
                with open(os.path.join(cfg["output_dir"], "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
                if cfg.get("wandb_project"):
                    import wandb
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})

    # Save final model
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
        print(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
