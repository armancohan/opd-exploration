"""Shared utilities for OPSD analysis scripts."""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import TEACHER_PROMPT_TEMPLATE, STUDENT_PROMPT_TEMPLATE


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.eval()
    return model, tokenizer


def apply_chat_template(tokenizer, text: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


def tokenize(tokenizer, text: str, max_length: int, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def get_logits(model, input_ids, attention_mask):
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).logits


def generate_rollout(model, tokenizer, prompt_text: str, max_new_tokens: int,
                     max_prompt_len: int, device, temperature: float = 0.8):
    ids, mask = tokenize(tokenizer, prompt_text, max_prompt_len, device)
    prompt_len = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )
    completion_ids = out[0, prompt_len:].cpu()
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_ids, completion_text, prompt_len


def get_completion_logits(model, tokenizer, prompt_text: str,
                          completion_ids: torch.Tensor, max_prompt_len: int, device):
    """Run forward pass and return logits only over the completion tokens."""
    p_ids, p_mask = tokenize(tokenizer, prompt_text, max_prompt_len, device)
    p_len = p_ids.shape[1]
    full = torch.cat([p_ids[0], completion_ids.to(device)]).unsqueeze(0)
    full_mask = torch.ones(1, full.shape[1], dtype=torch.long, device=device)
    logits = get_logits(model, full, full_mask)  # [1, L, V]
    T_comp = completion_ids.shape[0]
    return logits[0, p_len - 1: p_len - 1 + T_comp, :]  # [T_comp, V]


def chunked_jsd(logits_a: torch.Tensor, logits_b: torch.Tensor,
                chunk_size: int = 64, temperature: float = 1.0) -> torch.Tensor:
    """Per-token JSD(P||Q). logits_a, logits_b: [T, V] → [T]"""
    T = logits_a.shape[0]
    out = torch.zeros(T, device=logits_a.device, dtype=torch.float32)
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        la = logits_a[s:e].float() / temperature
        lb = logits_b[s:e].float() / temperature
        lpa = F.log_softmax(la, dim=-1)
        lpb = F.log_softmax(lb, dim=-1)
        lm = torch.logaddexp(lpa - 0.693147, lpb - 0.693147)
        out[s:e] = 0.5 * ((lpa.exp() * (lpa - lm)).sum(-1) +
                          (lpb.exp() * (lpb - lm)).sum(-1))
    return out


def chunked_entropy(logits: torch.Tensor, chunk_size: int = 64,
                    temperature: float = 1.0) -> torch.Tensor:
    """Per-token entropy. logits: [T, V] → [T]"""
    T = logits.shape[0]
    out = torch.zeros(T, device=logits.device, dtype=torch.float32)
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        l = logits[s:e].float() / temperature
        lp = F.log_softmax(l, dim=-1)
        out[s:e] = -(lp.exp() * lp).sum(-1)
    return out


def extract_answer(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]+)\}", solution)
    return m.group(1).strip() if m else solution[-60:].strip()


def smooth(arr: np.ndarray, w: int = 8) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid")
