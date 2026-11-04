"""From-scratch PyTorch modules: RoPE, causal attention, transformer blocks, TinyGPT.

The modules are intentionally small and literal. Their numerical behaviour is
validated against PyTorch's fused attention and against analytical properties
of rotary embeddings in experiment E3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_frequencies(
    head_dim: int, base: float = 10_000.0, device=None, dtype=torch.float32
) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
    return 1.0 / (base**exponent)


def rope_cos_sin(
    seq_len: int, head_dim: int, base: float = 10_000.0, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = rope_frequencies(head_dim, base=base, device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent dimension pairs; x has shape (..., T, head_dim)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    out = torch.stack((rotated_even, rotated_odd), dim=-1)
    return out.flatten(-2)


def attention_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Plain scaled dot-product attention. Shapes: (..., T, D)."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        t = q.size(-2)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        shape = (batch, seq_len, self.n_head, self.head_dim)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)
        if self.use_rope:
            cos, sin = rope_cos_sin(seq_len, self.head_dim, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.proj(out)
        return out, (weights if return_attn else None)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, multiplier: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, multiplier * n_embd),
            nn.GELU(),
            nn.Linear(multiplier * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        self.norm2 = RMSNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout=dropout)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, weights


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 128,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout=dropout) for _ in range(n_layer)]
        )
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if idx.size(1) > self.block_size:
            raise ValueError("sequence longer than block_size")
        x = self.token_embedding(idx)
        all_weights: list[torch.Tensor] | None = [] if return_attn else None
        for block in self.blocks:
            x, weights = block(x, return_attn=return_attn)
            if all_weights is not None and weights is not None:
                all_weights.append(weights.detach())
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss, all_weights

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return int(n)
