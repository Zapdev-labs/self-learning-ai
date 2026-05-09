"""Vision encoder (ViT-style) for multimodal ASSBRAIN."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size: int = 336, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//p, W//p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-based vision encoder for multimodal understanding."""

    def __init__(
        self,
        img_size: int = 336,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project to LLM dimension if needed
        self.projection = nn.Identity()
        if output_dim is not None and output_dim != embed_dim:
            self.projection = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, C, H, W) normalized to [-1, 1] or [0, 1]
        Returns: (B, num_patches + 1, output_dim) — patch tokens + cls token
        """
        B = images.size(0)
        x = self.patch_embed(images)  # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.projection(x)
        return x

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches


class VisionProjector(nn.Module):
    """MLP that projects vision features to LLM space with per-patch pooling."""

    def __init__(self, vision_dim: int, llm_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = vision_dim if i == 0 else llm_dim
            out_d = llm_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(vision_features)
