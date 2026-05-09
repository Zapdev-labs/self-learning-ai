"""9B Multimodal ASSBRAIN Model: Vision + Text + Tool Calling."""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .custom_transformer import RMSNorm, CausalSelfAttention, SwiGLU, get_rotary_embedding, apply_rotary_pos_emb
from .vision_encoder import VisionEncoder, VisionProjector


class MultimodalConfig:
    """Configuration for the 9B multimodal model."""

    def __init__(
        self,
        # Text model
        vocab_size: int = 32000,
        block_size: int = 8192,
        n_layer: int = 48,
        n_head: int = 32,
        n_embd: int = 4096,
        dropout: float = 0.0,
        ffn_mult: float = 4.0,
        multiple_of: int = 256,
        tie_weights: bool = True,
        use_gradient_checkpointing: bool = True,
        # Vision model
        img_size: int = 336,
        patch_size: int = 14,
        vision_embed_dim: int = 1024,
        vision_n_layers: int = 24,
        vision_n_heads: int = 16,
        # Multimodal
        vision_token_id: int = 32000,
        image_start_token_id: int = 32001,
        image_end_token_id: int = 32002,
        # Tool calling
        tool_call_begin_id: int = 32003,
        tool_call_end_id: int = 32004,
        tool_result_begin_id: int = 32005,
        tool_result_end_id: int = 32006,
        # Special
        pad_token_id: int = 0,
        eos_token_id: int = 2,
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
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.img_size = img_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.vision_n_layers = vision_n_layers
        self.vision_n_heads = vision_n_heads

        self.vision_token_id = vision_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.tool_call_begin_id = tool_call_begin_id
        self.tool_call_end_id = tool_call_end_id
        self.tool_result_begin_id = tool_result_begin_id
        self.tool_result_end_id = tool_result_end_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Derived
        self.num_image_tokens = (img_size // patch_size) ** 2 + 1  # +1 for cls

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Text model
        emb = self.vocab_size * self.n_embd
        if self.tie_weights:
            emb = self.vocab_size * self.n_embd
        else:
            emb *= 2
        hidden = ((int(self.n_embd * self.ffn_mult * 2 / 3) + self.multiple_of - 1) // self.multiple_of) * self.multiple_of
        attn = 4 * self.n_embd * self.n_embd
        ffn = 3 * self.n_embd * hidden
        norms = 2 * self.n_embd
        layer_params = attn + ffn + norms
        text_total = emb + self.n_layer * layer_params + self.n_embd

        # Vision model
        num_patches = (self.img_size // self.patch_size) ** 2
        v_emb = self.patch_size * self.patch_size * 3 * self.vision_embed_dim
        v_pos = (num_patches + 1) * self.vision_embed_dim
        v_layer = (4 * self.vision_embed_dim * self.vision_embed_dim +
                   self.vision_embed_dim * int(self.vision_embed_dim * 4) * 2)
        vision_total = v_emb + v_pos + self.vision_n_layers * v_layer + self.vision_embed_dim

        # Projector
        proj = self.vision_embed_dim * self.n_embd + self.n_embd * self.n_embd

        return text_total + vision_total + proj


class MultimodalTransformerBlock(nn.Module):
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
        self.use_checkpoint = config.use_gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            x = x + torch.utils.checkpoint.checkpoint(self.attn, self.attn_norm(x))
            x = x + torch.utils.checkpoint.checkpoint(self.ffn, self.ffn_norm(x))
        else:
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
        return x


class AssBrainMultimodal(nn.Module):
    """9B Multimodal model: text + vision + tool calling."""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # Text embedding
        self.token_emb = nn.Embedding(config.vocab_size + 7, config.n_embd)  # +7 special tokens
        self.drop = nn.Dropout(config.dropout)

        # Vision encoder + projector
        self.vision_encoder = VisionEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_embed_dim,
            n_layers=config.vision_n_layers,
            n_heads=config.vision_n_heads,
            output_dim=config.vision_embed_dim,
        )
        self.vision_projector = VisionProjector(config.vision_embed_dim, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MultimodalTransformerBlock(config)
            for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size + 7, bias=False)

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

    def _embed_images(self, images: Optional[List[Image.Image]], device: torch.device) -> Optional[torch.Tensor]:
        """Encode images and project to LLM space."""
        if images is None or len(images) == 0:
            return None
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        tensors = torch.stack([transform(img) for img in images]).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            vision_out = self.vision_encoder(tensors)  # (B, N_patches+1, vision_dim)
            projected = self.vision_projector(vision_out)  # (B, N_patches+1, n_embd)
        return projected

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[List[Image.Image]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, T = input_ids.size()

        # Embed text tokens
        x = self.token_emb(input_ids)

        # Replace image placeholder tokens with actual vision features
        if images is not None and len(images) > 0:
            vision_features = self._embed_images(images, input_ids.device)
            if vision_features is not None:
                # Find image_start / image_end positions
                # For simplicity: assume single image per batch at consistent position
                # A full implementation would handle arbitrary positions
                start_mask = (input_ids == self.config.image_start_token_id)
                end_mask = (input_ids == self.config.image_end_token_id)
                # Replace tokens between start and end with vision features
                # This is a simplified version
                pass  # Complex insertion logic handled in generate()

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
        images: Optional[List[Image.Image]] = None,
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
        cfg = self.config

        # Pre-encode images if provided
        vision_features = self._embed_images(images, device)

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -cfg.block_size:]

            # Forward pass
            x = self.token_emb(idx_cond)

            # Insert vision features at image token positions for this window
            if vision_features is not None:
                # Find image token in current window
                for b in range(x.size(0)):
                    img_positions = (idx_cond[b] == cfg.image_start_token_id).nonzero(as_tuple=True)[0]
                    if len(img_positions) > 0:
                        pos = img_positions[0].item()
                        n_vis = min(vision_features.size(1), cfg.block_size - pos)
                        x[b, pos:pos+n_vis] = vision_features[b, :n_vis]

            x = self.drop(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            logits = self.lm_head(x)

            logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token in set(input_ids[i].tolist()):
                        logits[i, token] /= repetition_penalty

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p
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

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from generated text."""
        cfg = self.config
        calls = []
        begin_tag = f"<{cfg.tool_call_begin_id}>"
        end_tag = f"<{cfg.tool_call_end_id}>"

        # Search for tool call blocks
        import re
        pattern = rf"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                call = json.loads(match.strip())
                calls.append(call)
            except json.JSONDecodeError:
                continue
        return calls

    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """Format a tool result for feeding back into the model."""
        return f"<|tool_result_begin|>\n{{\"name\": \"{tool_name}\", \"result\": {json.dumps(result)}}}\n<|tool_result_end|>"

    def get_num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            n -= self.token_emb.weight.numel()
        return n

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) for A40."""
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
