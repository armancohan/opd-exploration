"""CH-OPD training script."""

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
from src.causal_hinge import CausalHingeOPSD, CHConfig


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
    p.add_argument("--output_dir", type=str, default="outputs/ch_opsd")
    p.add_argument("--eval_dataset", type=str, default="aime2024")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--max_prompt_len", type=int, default=512)
    # CH-specific
    p.add_argument("--n_probe_positions", type=int, default=2)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--n_probes", type=int, default=2)
    p.add_argument("--max_probe_tokens", type=int, default=150)
    p.add_argument("--tau_benefit", type=float, default=0.0)
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
            run_name = cfg.get("wandb_run_name", f"ch-opd_{Path(cfg['output_dir']).name}")
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)

    ch_config = CHConfig(
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
        n_probe_positions=cfg.get("n_probe_positions", 2),
        n_candidates=cfg.get("n_candidates", 4),
        n_probes=cfg.get("n_probes", 2),
        max_probe_tokens=cfg.get("max_probe_tokens", 150),
        tau_benefit=cfg.get("tau_benefit", 0.0),
    )

    trainer = CausalHingeOPSD(model, tokenizer, optimizer, accelerator, config=ch_config)

    train_data = load_train_dataset(cfg["dataset"], n_samples=cfg["n_train_samples"])
    eval_data = []
    if cfg.get("eval_dataset") == "aime2024":
        eval_data = load_aime_dataset([2024])
    elif cfg.get("eval_dataset") == "aime2025":
        eval_data = load_aime_dataset([2025])
    elif cfg.get("eval_dataset") == "math500":
        eval_data = load_math500_dataset()

    if accelerator.is_main_process:
        print(f"CH-OPD | model={model_name} | train={len(train_data)} | eval={len(eval_data)}")
        print(f"Probe positions={ch_config.n_probe_positions}, candidates={ch_config.n_candidates}, probes={ch_config.n_probes}")

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
                f"hinge_rate={metrics.get('hinge_positive_rate', 0):.3f}"
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
        print(f"CH-OPD model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
