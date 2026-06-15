"""SegFormer backbone (frozen) + custom floorplan decoder for semantic segmentation (spec_v005).

Architecture overview:
  SegFormerBackboneExtractor  — frozen pretrained SegFormer encoder
                                run ONCE per image to extract multi-scale features,
                                then cached to disk.  NOT called during training loops.
  FloorplanDecoder            — fully trainable custom decoder (spec §7)
                                projection → 2 conv hidden layers → 1×1 final → upsample
  FloorplanSegModel           — backbone + decoder combined for inference / preview

Phase 1 training workflow (spec §7 "Frozen Backbone Feature Cache Mode"):
  1. Load pretrained SegFormer backbone
  2. Freeze backbone parameters
  3. Run each image through backbone once → save to features/<sample_id>.pt
  4. Train only FloorplanDecoder using cached features
  5. Backbone forward pass is NOT called during head-only training
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

# HuggingFace model IDs for each SegFormer backbone variant
SEGFORMER_VARIANTS: dict[str, str] = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b1": "nvidia/mit-b1",
    "segformer_b2": "nvidia/mit-b2",
}

# Hidden channel sizes of the 4 encoder stages for each variant
BACKBONE_HIDDEN_SIZES: dict[str, list[int]] = {
    "segformer_b0": [32, 64, 160, 256],
    "segformer_b1": [64, 128, 320, 512],
    "segformer_b2": [64, 128, 320, 512],
}


# ---------------------------------------------------------------------------
# Frozen backbone
# ---------------------------------------------------------------------------


class SegFormerBackboneExtractor(nn.Module):
    """Frozen SegFormer encoder — used only for feature extraction / caching.

    All parameters have ``requires_grad=False``.
    The backbone is run ONCE per image (not per epoch) and the multi-scale
    feature tensors are saved to disk.  During head-only training the
    backbone is never invoked.

    Args:
        variant:    One of the keys in ``SEGFORMER_VARIANTS``.
        pretrained: Load pretrained ImageNet weights from HuggingFace.
    """

    def __init__(self, variant: str = "segformer_b0", pretrained: bool = True) -> None:
        super().__init__()
        model_id = SEGFORMER_VARIANTS.get(variant, SEGFORMER_VARIANTS["segformer_b0"])
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(model_id)
        else:
            config = SegformerConfig.from_pretrained(model_id)
            self.encoder = SegformerModel(config)

        # Freeze every parameter
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.variant = variant

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract multi-scale backbone features.

        Args:
            pixel_values: ``[B, 3, H, W]`` normalized input images.

        Returns:
            Tuple of 4 hidden-state tensors (one per encoder stage).
            Shape per tensor: ``[B, N_i, C_i]`` (flat) where
            ``N_i = (H / stride_i) * (W / stride_i)`` and ``C_i`` is
            the stage channel count (e.g. 32, 64, 160, 256 for B0).
            ``FloorplanDecoder.forward()`` handles the flat→spatial reshape.
        """
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        return outputs.hidden_states  # tuple of 4 tensors


# ---------------------------------------------------------------------------
# Custom decoder / head
# ---------------------------------------------------------------------------


class FloorplanDecoder(nn.Module):
    """Custom trainable decoder for floorplan semantic segmentation.

    All parameters are trainable (``requires_grad=True``).

    Spec §7 — Decoder / Head Architecture:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Fusion                                                           │
    │    • Project each of the 4 backbone stages to PROJ_DIM with 1×1  │
    │    • Reshape to spatial [B, PROJ_DIM, H_i, W_i]                  │
    │    • Upsample all to stage-1 resolution (H/4 × W/4)              │
    │    • Element-wise sum → [B, PROJ_DIM, H/4, W/4]                  │
    │  Hidden Layer 1                                                    │
    │    Conv 3×3 → 256ch, BatchNorm2d, GELU, Dropout2d(0.1)           │
    │  Hidden Layer 2                                                    │
    │    Conv 3×3 → 128ch, BatchNorm2d, GELU, Dropout2d(0.1)           │
    │  Final Classification Layer                                        │
    │    1×1 Conv → num_classes logits                                  │
    │  Upsample bilinear → [B, num_classes, output_size, output_size]   │
    └───────────────────────────────────────────────────────────────────┘

    Args:
        encoder_hidden_sizes: Channel counts of each backbone stage
            (e.g. ``[32, 64, 160, 256]`` for SegFormer-B0).
        num_classes:   Number of semantic classes.
        output_size:   Spatial size of the output logit map (square).
    """

    PROJ_DIM: int = 256  # channels after per-stage projection

    def __init__(
        self,
        encoder_hidden_sizes: list[int],
        num_classes: int,
        output_size: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_size = output_size

        # --- Projection: one 1×1 conv per encoder stage ---
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(c, self.PROJ_DIM, kernel_size=1, bias=False)
            for c in encoder_hidden_sizes
        ])

        # --- Hidden Layer 1: Conv 3×3, 256ch, BN, GELU, Dropout2d ---
        self.hidden1 = nn.Sequential(
            nn.Conv2d(self.PROJ_DIM, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
        )

        # --- Hidden Layer 2: Conv 3×3, 128ch, BN, GELU, Dropout2d ---
        self.hidden2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
        )

        # --- Final Classification Layer: 1×1 Conv → num_classes ---
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Decode backbone features to pixel-level semantic logits.

        Args:
            hidden_states: Tuple of 4 tensors from ``SegFormerBackboneExtractor``.
                Each tensor may be:
                  - ``[B, N_i, C_i]``  flat format  (from encoder hidden_states)
                  - ``[B, C_i, H_i, W_i]`` spatial format  (older transformers versions)

        Returns:
            Logits ``[B, num_classes, output_size, output_size]``.
        """
        projected: list[torch.Tensor] = []
        target_h: int | None = None
        target_w: int | None = None

        for i, (hs, proj) in enumerate(zip(hidden_states, self.proj_layers)):
            # --- Normalise to spatial format [B, C, H, W] ---
            if hs.dim() == 3:
                # Flat format [B, N, C] from SegFormer encoder hidden_states
                B, N, C = hs.shape
                h = w = int(math.sqrt(N))
                spatial = hs.transpose(1, 2).reshape(B, C, h, w)
            else:
                # Already spatial [B, C, H, W]
                spatial = hs

            # Project all stages to PROJ_DIM channels
            spatial = proj(spatial)  # [B, PROJ_DIM, H_i, W_i]

            # Stage 0 (finest) defines the fusion resolution
            if i == 0:
                target_h = spatial.shape[2]
                target_w = spatial.shape[3]

            projected.append(spatial)

        # --- Upsample all to stage-0 resolution and sum ---
        fused = torch.zeros_like(projected[0])
        for feat in projected:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            fused = fused + feat  # [B, PROJ_DIM, H/4, W/4]

        # --- Apply hidden layers ---
        x = self.hidden1(fused)    # [B, 256, H/4, W/4]
        x = self.hidden2(x)        # [B, 128, H/4, W/4]
        x = self.classifier(x)    # [B, num_classes, H/4, W/4]

        # --- Upsample to output resolution ---
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return x  # [B, num_classes, output_size, output_size]


# ---------------------------------------------------------------------------
# Full model (backbone + decoder) — used for inference and preview
# ---------------------------------------------------------------------------


class FloorplanSegModel(nn.Module):
    """Full segmentation model: frozen backbone + trainable decoder.

    During Phase-1 training the decoder is trained separately using cached
    backbone features.  This combined model is used only for inference,
    preview generation, and future full-fine-tuning phases.
    """

    def __init__(
        self,
        backbone: SegFormerBackboneExtractor,
        decoder: FloorplanDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Full forward pass: backbone → decoder → logits."""
        hidden_states = self.backbone(pixel_values)
        return self.decoder(hidden_states)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def build_backbone(
    variant: str = "segformer_b0",
    pretrained: bool = True,
) -> SegFormerBackboneExtractor:
    """Build the frozen SegFormer backbone extractor."""
    return SegFormerBackboneExtractor(variant=variant, pretrained=pretrained)


def build_decoder(
    variant: str = "segformer_b0",
    num_classes: int = 5,
    output_size: int = 512,
) -> FloorplanDecoder:
    """Build the custom floorplan segmentation decoder."""
    hidden_sizes = BACKBONE_HIDDEN_SIZES.get(variant, BACKBONE_HIDDEN_SIZES["segformer_b0"])
    return FloorplanDecoder(
        encoder_hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        output_size=output_size,
    )


def build_model(
    model_name: str = "segformer_b0",
    num_classes: int = 5,
    pretrained: bool = True,
    output_size: int = 512,
) -> FloorplanSegModel:
    """Build the full segmentation model (frozen backbone + trainable decoder).

    This is the primary entry point used by training and test scripts.

    Args:
        model_name:  SegFormer variant key (e.g. ``"segformer_b0"``).
        num_classes: Number of semantic segmentation classes.
        pretrained:  Load pretrained backbone weights from HuggingFace.
        output_size: Output spatial resolution (square, pixels).

    Returns:
        ``FloorplanSegModel`` with frozen backbone and trainable decoder.
    """
    backbone = build_backbone(variant=model_name, pretrained=pretrained)
    decoder = build_decoder(variant=model_name, num_classes=num_classes, output_size=output_size)
    return FloorplanSegModel(backbone=backbone, decoder=decoder)
