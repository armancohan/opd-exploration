"""FED-OPD training script."""

import argparse
import copy
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
from src.fed import FEDTrainer, FEDConfig


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
    p.add_argument("--output_dir", type=str, default="outputs/fed_opsd")
    p.add_argument("--eval_dataset", type=str, default="aime2024")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_prompt_len", type=int, default=512)
    # FED-specific
    p.add_argument("--n_anchor_positions", type=int, default=2)
    p.add_argument("--n_continuations_per_anchor", type=int, default=8)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--beta_fed", type=float, default=0.5)
    p.add_argument("--tau_value", type=float, default=0.3)
    p.add_argument("--lambda_within", type=float, default=0.1)
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
            run_name = cfg.get("wandb_run_name", f"fed-opd_{Path(cfg['output_dir']).name}")
            wandb.init(project=cfg["wandb_project"], config=cfg, name=run_name)

    model_name = cfg["model"]
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

    # Create frozen reference model (initial policy)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    model, optimizer = accelerator.prepare(model, optimizer)
    ref_model = accelerator.prepare(ref_model)

    fed_config = FEDConfig(
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
        n_anchor_positions=cfg.get("n_anchor_positions", 2),
        n_continuations_per_anchor=cfg.get("n_continuations_per_anchor", 8),
        rho=cfg.get("rho", 0.5),
        beta_fed=cfg.get("beta_fed", 0.5),
        tau_value=cfg.get("tau_value", 0.3),
        lambda_within=cfg.get("lambda_within", 0.1),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
    )

    trainer = FEDTrainer(model, tokenizer, optimizer, accelerator, config=fed_config, ref_model=ref_model)

    train_data = load_train_dataset(cfg["dataset"], n_samples=cfg["n_train_samples"])
    eval_data = []
    if cfg.get("eval_dataset") == "aime2024":
        eval_data = load_aime_dataset([2024])
    elif cfg.get("eval_dataset") == "aime2025":
        eval_data = load_aime_dataset([2025])
    elif cfg.get("eval_dataset") == "math500":
        eval_data = load_math500_dataset()

    if accelerator.is_main_process:
        print(f"FED-OPD | model={model_name} | train={len(train_data)} | eval={len(eval_data)}")
        print(f"rho={fed_config.rho}, beta_fed={fed_config.beta_fed}, tau={fed_config.tau_value}")

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
            print(
                f"Step {step}: loss={metrics['loss']:.4f}, "
                f"reward={metrics['reward_mean']:.3f}, "
                f"n_classes={metrics.get('n_unique_classes', 0)}"
            )
            if cfg.get("wandb_project"):
                import wandb
                wandb.log({"train/" + k: v for k, v in metrics.items()}, step=step)

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
                    wandb.log({"eval/" + k: v for k, v in eval_metrics.items()}, step=step + 1)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
        print(f"FED-OPD model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
