"""Custom decoder-only transformer trained from scratch.

Architecture: GPT-style with modern improvements:
  - RoPE (Rotary Position Embeddings)
  - RMSNorm instead of LayerNorm
  - SwiGLU FFN activation
  - Optional Flash Attention 2
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_rotary_embedding(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = get_rotary_embedding(T, self.head_dim, x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        hidden = int(config.n_embd * config.ffn_mult * 2 / 3)
        hidden = ((hidden + config.multiple_of - 1) // config.multiple_of) * config.multiple_of
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ModelConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        block_size: int = 4096,
        n_layer: int = 24,
        n_head: int = 16,
        n_embd: int = 1024,
        dropout: float = 0.0,
        ffn_mult: float = 4.0,
        multiple_of: int = 256,
        tie_weights: bool = True,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.ffn_mult = ffn_mult
        self.multiple_of = multiple_of
        self.tie_weights = tie_weights

    def estimate_params(self) -> int:
        """Rough parameter count."""
        emb = self.vocab_size * self.n_embd
        if self.tie_weights:
            emb = self.vocab_size * self.n_embd  # shared
        else:
            emb *= 2
        # Per layer: 4 linear projections (q,k,v,o) + 3 FFN linear
        # qkv: 3 * n_embd * n_embd, o: n_embd * n_embd
        # ffn: w1, w3: n_embd * hidden, w2: hidden * n_embd
        hidden = int(self.n_embd * self.ffn_mult * 2 / 3)
        hidden = ((hidden + self.multiple_of - 1) // self.multiple_of) * self.multiple_of
        attn = 4 * self.n_embd * self.n_embd
        ffn = 3 * self.n_embd * hidden
        norms = 2 * self.n_embd  # RMSNorm weights
        layer_params = attn + ffn + norms
        total = emb + self.n_layer * layer_params + self.n_embd  # final norm
        return total


class AssBrainTransformer(nn.Module):
    """Custom decoder-only transformer for ASSBRAIN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        B, T = input_ids.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        x = self.token_emb(input_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.block_size:]
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token in set(input_ids[i].tolist()):
                        logits[i, token] /= repetition_penalty

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float("-inf")

            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def get_num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            n -= self.token_emb.weight.numel()  # don't double count
        return n

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU)."""
        N = self.get_num_params()
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A40 FP16 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
