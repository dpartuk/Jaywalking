# -*- coding: utf-8 -*-
"""
gptFromScratch_on_imdb.py

A compact, single-file, pedagogical GPT implementation with configurable pretraining dataset.
Key features:
- Configurable pretraining dataset via Config.pretraining_dataset ("stanfordnlp/imdb" or "toy").
- Tokenizer built with huge model_max_length to avoid warnings; we control length via chunking.
- Concatenate+chunk language-modeling dataset creation on CPU with optional token caps for fast demos.
- DataLoader tuned for throughput (num_workers, pin_memory, persistent_workers) and moving to device per batch.
- "pedagogical_mode" forward hooks to print shapes once on first pass.
- Optional fine-tuning section: "Fine-tuning: Coldplay lyrics".
- No Colab-only dependencies; character-level tokenizer section removed.
- Benchmarking of model for quality estimation

### GPT2-124M
| Parameter      | Value   | Description                 |
|----------------|---------|-----------------------------|
| `vocab_size`   | 50257   | Vocabulary size             |
| `context_length`| 1024    | Context length              |
| `emb_dim`      | 768     | Embedding dimension         |
| `n_heads`      | 12      | Number of attention heads   |
| `n_layers`     | 12      | Number of layers            |
| `drop_rate`    | 0.1     | Dropout rate                |
| `qkv_bias`     | False   | Query-Key-Value bias        |

We'll use somewhat smaller numbers here
"""

# ========== 0) Imports ==========
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datasets import load_dataset
import math
import random
import logging
import os

import torch
from torch import nn # The foundation class for Neural Networks
from torch.utils.data import Dataset, DataLoader, TensorDataset

try:
    from transformers import AutoTokenizer  # optional but recommended
except Exception:
    AutoTokenizer = None
    logging.warning("transformers not available; using a simple whitespace tokenizer fallback.")

from datasets import load_dataset

# (optional) Hugging Face datasets for IMDB and Coldplay demos
try:
    import datasets as hf_datasets
except Exception:
    hf_datasets = None
    logging.warning("datasets not available; IMDB pretraining and Coldplay fine-tuning will be disabled.")

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version >=1\.16\.5 and <1\.23\.0 is required",
    category=UserWarning,
    module="scipy"
)

# ========== 1) Config & Utilities ==========

@dataclass
class Config:
    # Dataset control
    pretraining_dataset: str = "stanfordnlp/imdb"  # or "toy"
    imdb_val_fraction: float = 0.05

    # Data / Tokenizer
    tokenizer_name: str = "gpt2"
    context_length: int = 128

    # Optional caps to keep classroom demos brisk
    max_train_tokens: Optional[int] = 500_000
    max_val_tokens:   Optional[int] = 100_000

    # Model
    vocab_size: int = 50257  # overwritten by tokenizer if available
    embed_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    qkv_bias: bool = False

    # Training
    batch_size: int = 32
    max_steps: int = 500
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    grad_clip: Optional[float] = 1.0

    # Pedagogical
    seed: int = 1337
    pedagogical_mode: bool = True  # controls extra prints / shape checks
    plot_curves: bool = False       # keep off by default in a minimal script
    device: Optional[str] = None    # "cuda", "mps", or "cpu" (auto if None)

def get_device(cfg: Config) -> torch.device:
    if cfg.device is not None:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========== 2) Tokenization & Data ==========

def build_tokenizer(cfg: Config):
    """
    Builds/loads a tokenizer. If transformers isn't available, a simple whitespace tokenizer fallback is used.
    We set a huge model_max_length so the tokenizer doesn't warn; we control lengths via chunking.
    """
    if AutoTokenizer is None:
        logging.warning("transformers not available; using a simple whitespace tokenizer fallback.")
        class SimpleTok:
            bos_token_id = 0
            eos_token_id = 1
            pad_token_id = 1
            vocab = {"<BOS>":0, "<PAD>":1}
            def __init__(self): pass
            def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None, add_special_tokens=True, return_attention_mask=False):
                ids = []
                for t in texts:
                    toks = t.strip().split()
                    arr = [self.bos_token_id] + [hash(w)%10000+2 for w in toks]
                    if max_length and padding == "max_length":
                        arr = arr[:max_length]
                        arr = arr + [self.pad_token_id]*(max_length-len(arr))
                    ids.append(arr)
                return {"input_ids": torch.tensor(ids, dtype=torch.long)}
            @property
            def vocab_size(self): return 10002
            @property
            def pad_token(self): return "<PAD>"
            @property
            def eos_token(self): return "<PAD>"
            @pad_token.setter
            def pad_token(self, v): pass
        tok = SimpleTok()
    else:
        # Large cap to avoid model_max_length warnings; we manage length in chunking
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, model_max_length=10**9)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    return tok

def make_toy_corpus() -> List[str]:
    # A tiny toy corpus to keep the demo runnable in minutes.
    return [
        "Transformers use self attention to mix information across positions.",
        "Attention allows each token to look at previous tokens and compute a context aware representation.",
        "Language modeling trains a network to predict the next token from previous ones.",
        "Mini GPTs are great for teaching and for sanity checking ideas before training large models.",
    ]

def encode_texts(texts: List[str], tok, max_len: int, device: torch.device) -> torch.Tensor:
    out = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    return out["input_ids"].to(device)

def build_dataloaders(cfg: Config, tok, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train/val loaders.
    """
    if (
        cfg.pretraining_dataset.lower() == "stanfordnlp/imdb"
        and hf_datasets is not None
        and AutoTokenizer is not None
    ):
        raw = load_dataset("stanfordnlp/imdb")
        train_split = raw["train"]
        val_split   = raw.get("test", None)

        def tok_fn(batch):
            # No truncation/padding here; we will chunk later.
            return tok(
                batch["text"],
                add_special_tokens=True,
                return_attention_mask=False,
                truncation=False,
                padding=False,
            )

        train_tok = train_split.map(tok_fn, batched=True, remove_columns=train_split.column_names)
        val_tok   = val_split.map(tok_fn,   batched=True, remove_columns=val_split.column_names) if val_split else None

        block_size = cfg.context_length + 1

        def chunker(ds, token_cap: Optional[int] = None):
            concat = []
            eos = tok.eos_token_id if getattr(tok, "eos_token_id", None) is not None else 0
            for ids in ds["input_ids"]:
                concat.extend(ids + [eos])  # add EOS between docs
                if token_cap is not None and len(concat) >= token_cap:
                    break
            if token_cap is not None and len(concat) > token_cap:
                concat = concat[:token_cap]
            total = (len(concat) // block_size) * block_size
            concat = concat[:total]
            blocks = [concat[i:i+block_size] for i in range(0, total, block_size)]
            arr = torch.tensor(blocks, dtype=torch.long)    # CPU tensor
            x = arr[:, :-1].contiguous()
            y = arr[:,  1:].contiguous()
            return x, y

        x_tr, y_tr = chunker(train_tok, cfg.max_train_tokens)
        if val_tok is not None and len(val_tok) > 0:
            x_va, y_va = chunker(val_tok, cfg.max_val_tokens)
        else:
            # fallback: small validation slice from train
            n = x_tr.size(0)
            n_val = max(1, int(cfg.imdb_val_fraction * n))
            x_va, y_va = x_tr[-n_val:], y_tr[-n_val:]
            x_tr, y_tr = x_tr[:-n_val], y_tr[:-n_val]

        # DataLoader on CPU tensors; move to device in the training loop
        is_cuda = (device.type == "cuda")
        nw = max(1, (os.cpu_count() or 4) // 2)
        persistent = True if nw > 0 else False

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=nw,
            pin_memory=is_cuda,
            persistent_workers=persistent,
        )
        val_loader = DataLoader(
            TensorDataset(x_va, y_va),
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=nw,
            pin_memory=is_cuda,
            persistent_workers=persistent,
        )
        return train_loader, val_loader

    # --------- Fallback: toy corpus ----------
    texts = make_toy_corpus()
    n_train = max(2, int(0.75*len(texts)))
    train_ids = encode_texts(texts[:n_train], tok, cfg.context_length, device)
    val_ids   = encode_texts(texts[n_train:], tok, cfg.context_length, device)
    x_tr = train_ids[:, :-1].contiguous()
    y_tr = train_ids[:,  1:].contiguous()
    x_va = val_ids[:, :-1].contiguous()
    y_va = val_ids[:,  1:].contiguous()
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader

# ========== 3) Basic Architecture - Model Blocks ==========

class SelfAttention(nn.Module):
    """
    Single-head **masked** (causal) self-attention.

    - Takes token embeddings x of shape (B, T, C)
    - Projects them to queries, keys, values of shape (B, T, D_head)
    - Computes scaled dot-product attention with a causal mask
    - Returns attended values of shape (B, T, D_head)
    """
    def __init__(self, embed_dim: int, head_dim: int, bias: bool = False, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim

        # Learnable projections from model dimension -> head dimension
        self.w_query = nn.Linear(embed_dim, head_dim, bias=bias)
        self.w_key   = nn.Linear(embed_dim, head_dim, bias=bias)
        self.w_value = nn.Linear(embed_dim, head_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) token embeddings
        returns: (B, T, D_head) attended features for this head
        """
        B, T, _ = x.size()

        # 1) Project to q, k, v
        q = self.w_query(x)  # (B, T, D_head)
        k = self.w_key(x)    # (B, T, D_head)
        v = self.w_value(x)  # (B, T, D_head)

        # 2) Scaled dot-product attention scores
        #    scores[b, t_q, t_k] = <q[b, t_q], k[b, t_k]> / sqrt(D_head)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, T, T)

        # 3) Causal mask: prevent looking into the future
        #    mask[t_q, t_k] = True if t_k > t_q
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-1e10"))

        # 4) Convert scores to probabilities
        attn = scores.softmax(dim=-1)  # (B, T, T)
        attn = self.dropout(attn)

        # 5) Weighted sum of values
        out = attn @ v                 # (B, T, D_head)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head masked self-attention built from SelfAttention heads.

    - Splits the model dimension C into H heads of size D_head = C / H
    - Each head runs its own SelfAttention (independent projections)
    - Concatenates head outputs back to (B, T, C)
    - Optional final linear projection can be added if desired
    """
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False, dropout: float = 0.1, max_ctx: int = 2048):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # List of independent attention heads
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim=embed_dim, head_dim=self.head_dim, bias=bias, dropout=dropout)
            for _ in range(num_heads)
        ])

        # Final projection back to embed_dim (keeps interface consistent)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) where C = embed_dim
        returns: (B, T, C)
        """
        # Run each head on the same input x
        head_outputs = [head(x) for head in self.heads]     # list of (B, T, D_head)
        # Concatenate along the feature dimension to recover (B, T, C)
        concat = torch.cat(head_outputs, dim=-1)            # (B, T, num_heads * D_head) = (B, T, C)
        # Optional output projection mixes head information
        return self.out_proj(concat)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float, qkv_bias: bool, max_ctx: int = 2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, bias=qkv_bias, dropout=dropout, max_ctx=max_ctx)
        self.ffn  = FeedForward(embed_dim, hidden_dim=4*embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    """
    Minimal GPT-style decoder-only transformer.
    """
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, n_layers: int,
                 n_heads: int, dropout: float=0.0, qkv_bias: bool=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, n_heads, dropout, qkv_bias, max_ctx=context_length) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.context_length = context_length
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)  # (1,T)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B,T,C)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,vocab)
        return logits

# ========== 4) Shape-print hooks ==========

class _PrintOnce:
    def __init__(self): self.done = False
    def should(self):
        if self.done: return False
        self.done = True
        return True

def register_pedagogical_hooks(model: nn.Module, enabled: bool=True) -> List[torch.utils.hooks.RemovableHandle]:
    """
    If enabled, prints shapes at key points during the first forward pass.
    """
    handles = []
    if not enabled:
        return handles

    printer = _PrintOnce()

    def make_hook(name):
        def hook(mod, inp, out):
            if printer.should():
                def shape(x):
                    return tuple(x.shape) if isinstance(x, torch.Tensor) else str(type(x))
                in_shapes = [shape(t) for t in (inp if isinstance(inp, (list,tuple)) else [inp])]
                out_shape = shape(out)
                print(f"[pedagogical] {name}: input {in_shapes} -> output {out_shape}")
        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.ln1.register_forward_hook(make_hook(f"Block{i}.LN1")))
        handles.append(blk.attn.register_forward_hook(make_hook(f"Block{i}.MHA")))
        handles.append(blk.ln2.register_forward_hook(make_hook(f"Block{i}.LN2")))
        handles.append(blk.ffn.register_forward_hook(make_hook(f"Block{i}.FFN")))
    handles.append(model.head.register_forward_hook(make_hook("LMHead")))
    return handles

# ========== 5) Loss, Evaluation, Scheduler ==========

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (B,T,V), targets: (B,T)
    B, T, V = logits.shape
    return nn.functional.cross_entropy(logits.view(B*T, V), targets.view(B*T))

class CosineWithWarmup:
    # Lightweight LR scheduler: linear warmup then cosine decay to 10% of initial LR.
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, base_lr: float):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.t = 0

    def step(self):
        self.t += 1
        if self.t < self.warmup:
            lr = self.base_lr * self.t / self.warmup
        else:
            # cosine from base_lr -> 0.1*base_lr
            progress = (self.t - self.warmup) / max(1, (self.max_steps - self.warmup))
            min_lr = 0.1 * self.base_lr
            lr = min_lr + 0.5*(self.base_lr - min_lr)*(1 + math.cos(math.pi * progress))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))

# ========== 6) Training & Generation ==========

def train(cfg: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineWithWarmup(opt, cfg.warmup_steps, cfg.max_steps, cfg.lr)

    hooks = register_pedagogical_hooks(model, cfg.pedagogical_mode)

    step = 0
    while step < cfg.max_steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = cross_entropy_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            lr = sched.step()

            if step % 10 == 0:
                val_loss = evaluate(model, val_loader, device)
                logging.info(f"step {step:4d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | lr {lr:.2e}")
            step += 1
            if step >= cfg.max_steps:
                break

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_length: int,
             temperature: float=1.0, top_k: Optional[int]=None) -> torch.Tensor:
    """
    Autoregressive generation from context idx (B,T). Returns (B, T+max_new_tokens).
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)[:, -1, :] / max(1e-8, temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[..., [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    model.train()
    return idx

# ========== 7) Main: the build the entire flow, with (optional) fine-tune, generate ==========

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = Config()
    set_seed(cfg.seed)
    device = get_device(cfg)
    logging.info(f"Using device: {device}")

    # Tokenizer
    tok = build_tokenizer(cfg)
    vocab_size = getattr(tok, "vocab_size", cfg.vocab_size)
    cfg.vocab_size = vocab_size
    logging.info(f"Tokenizer vocab size: {cfg.vocab_size}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg, tok, device)

    # Model
    model = GPT(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        qkv_bias=cfg.qkv_bias,
    ).to(device)

    # Pretraining
    title = "IMDB" if cfg.pretraining_dataset.lower() == "stanfordnlp/imdb" else "toy corpus"
    logging.info(f"=== Pretraining ({title}) ===")
    train(cfg, model, train_loader, val_loader, device)

    # ---------- Optional: Fine-tuning on Coldplay lyrics ----------
    print("\n=== Fine-tuning: Coldplay lyrics ===")
    if hf_datasets is not None and AutoTokenizer is not None:
        try:
            ds = hf_datasets.load_dataset("huggingartists/coldplay")["train"]
            sample_texts = [ex["lyrics"] for ex in ds.select(range(min(64, len(ds))))]
            cold_ids = tok(sample_texts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=cfg.context_length)["input_ids"].to(device)
            x_co = cold_ids[:, :-1].contiguous()
            y_co = cold_ids[:, 1:].contiguous()
            cold_loader = DataLoader(TensorDataset(x_co, y_co), batch_size=cfg.batch_size, shuffle=True)

            # short tune
            old_steps = cfg.max_steps
            cfg.max_steps = min(200, old_steps)  # quick fine-tune
            train(cfg, model, cold_loader, val_loader, device)
            cfg.max_steps = old_steps
        except Exception as e:
            logging.info(f"(Skipping Coldplay fine-tune; reason: {e})")
    else:
        logging.info("(Hugging Face datasets/transformers not available; skipping Coldplay fine-tune.)")

    # ---------- Generation demo ----------
    logging.info("\n=== Generation demo ===")
    prompt = "In a world where small models teach big ideas"
    ctx = tok([prompt], return_tensors="pt", padding="max_length", truncation=True,
              max_length=cfg.context_length)["input_ids"][:, :9].to(device)  # small context slice
    out = generate(model, ctx, max_new_tokens=32, context_length=cfg.context_length, temperature=0.9, top_k=50)
    if hasattr(tok, "decode"):
        try:
            text = tok.batch_decode(out.tolist(), skip_special_tokens=True)[0]
        except Exception:
            text = str(out[0].tolist())
    else:
        text = str(out[0].tolist())
    print(text)

    # ========== 8) (Optional) Benchmark the model ==========

    #dataset = load_dataset("lambada", split="validation[:200]") #lambada_development_plain_text
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:400]")

    vocab_size = model.tok_emb.num_embeddings
    max_ctx = model.context_length  # from your GPT class

    correct = total = skipped = 0

    for sample in dataset:
        text = sample["text"].strip()
        if not text or " " not in text:
            continue

        # Split into context and target last word
        context, target = text.rsplit(" ", 1)

        # 1) Tokenize context, truncate to model's context window
        enc = tok(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=max_ctx,
        )
        ids = enc["input_ids"].to(device)  # (1, T')
        ids = ids[:, -max_ctx:]  # safety: ensure T' <= max_ctx

        # 2) Tokenize target (may be multi-token)
        tgt_ids = tok(" " + target, add_special_tokens=False).input_ids
        if not tgt_ids:
            continue

        # 3) Guard against any vocab mismatch (shouldn’t normally trigger)
        if torch.max(ids).item() >= vocab_size or max(tgt_ids) >= vocab_size:
            skipped += 1
            continue

        # 4) Model forward: we want logits for the **next token** after the context
        with torch.no_grad():
            logits_all = model(ids)  # (1, T', V)
            logits = logits_all[0, -1, :]  # logits for next token

        # 5) Greedy prediction
        pred_id = int(torch.argmax(logits).item())
        pred = tok.decode([pred_id]).strip()

        # 6) Clean tokenization artifacts for fair comparison
        target_ids = tok(" " + target, add_special_tokens=False).input_ids
        topk = torch.topk(logits, k=20).indices.tolist()
        if any(t in topk for t in target_ids):
            correct += 1
        total += 1

        #print("CONTEXT:", context)
        #print("TARGET:", target)
        #print("PRED:  ", pred)

    print(f"LAMBADA accuracy: {correct}/{total} = {correct / max(1, total):.3f}, skipped {skipped}")

if __name__ == "__main__":
    main()