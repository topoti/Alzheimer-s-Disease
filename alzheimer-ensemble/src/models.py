"""Model factory: ResNet50 / EfficientNetB3 / DenseNet121 backbones via timm (Phase 2).

Three genuinely different CNN *design paradigms* — residual learning (ResNet50),
compound scaling + MBConv (EfficientNetB3), and dense connectivity (DenseNet121) — so the
ensemble's errors are less correlated than a same-family ensemble's. See RESEARCH_PLAN.md
(Phase 2). Every backbone is loaded ImageNet-pretrained from ``timm`` and gets the same
fresh classifier head::

    GlobalAvgPool -> Dropout(0.3) -> Linear(in_features, num_classes)

The global-average-pool is provided by ``timm.create_model(..., num_classes=0,
global_pool="avg")``, which returns pooled penultimate-layer features of shape
``(batch, in_features)``; we attach Dropout + Linear on top. Loading the backbone with
``num_classes=0`` also gives a clean 2048/1536/1024-dim embedding (via ``forward_features``
+ pool) for the frozen-backbone ADASYN step in Phase 3.

Run directly to build all three and print parameter counts::

    python -m src.models
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
        "timm is required for src/models.py. Install with `pip install timm`."
    ) from exc


# Map our canonical short name -> the timm model identifier + expected input size.
# Input sizes are locked per backbone: 224 for ResNet/DenseNet, 300 for EfficientNetB3
# (RESEARCH_PLAN.md Phase 2 / configs/default.yaml -> data.input_sizes).
_MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "resnet50": {"timm_name": "resnet50", "input_size": 224},
    "efficientnet_b3": {"timm_name": "efficientnet_b3", "input_size": 300},
    "densenet121": {"timm_name": "densenet121", "input_size": 224},
}

SUPPORTED_MODELS = tuple(_MODEL_REGISTRY)

DROPOUT_P = 0.3


def get_input_size(name: str) -> int:
    """Return the expected square input size (pixels) for a backbone.

    224 for ``resnet50`` / ``densenet121``, 300 for ``efficientnet_b3``.
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {', '.join(SUPPORTED_MODELS)}."
        )
    return int(_MODEL_REGISTRY[name]["input_size"])


def build_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """Build an ImageNet-pretrained backbone with a fresh 4-class head.

    Parameters
    ----------
    name:
        One of ``{"resnet50", "efficientnet_b3", "densenet121"}``.
    num_classes:
        Output classes (default 4 for the OASIS task).
    pretrained:
        Load ImageNet weights (default True).

    Returns
    -------
    nn.Module
        A module whose ``forward(x) -> logits`` of shape ``(batch, num_classes)``.
        The classifier head is ``GlobalAvgPool -> Dropout(0.3) -> Linear(in_features,
        num_classes)``; global-average-pool comes from the timm backbone.
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {', '.join(SUPPORTED_MODELS)}."
        )

    # num_classes=0 + global_pool="avg" -> backbone outputs pooled features (B, in_features),
    # i.e. everything up to and including GlobalAvgPool, with the original head removed.
    backbone = timm.create_model(
        str(_MODEL_REGISTRY[name]["timm_name"]),
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    in_features = backbone.num_features
    head = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    return nn.Sequential(backbone, head)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main() -> None:
    """Build all three backbones and print parameter counts (build smoke test)."""
    print("Building all three backbones (pretrained=False for a fast offline check)...\n")
    print(f"{'model':<18}{'input':>7}{'total params':>16}{'trainable':>16}")
    print("-" * 57)
    for name in SUPPORTED_MODELS:
        model = build_model(name, num_classes=4, pretrained=False)
        counts = count_parameters(model)
        size = get_input_size(name)
        # Sanity forward pass on a single dummy image -> verify head shape (B, 4).
        with torch.no_grad():
            out = model(torch.zeros(1, 3, size, size))
        assert out.shape == (1, 4), f"{name} produced unexpected output shape {tuple(out.shape)}"
        print(
            f"{name:<18}{size:>7}{counts['total']:>16,}{counts['trainable']:>16,}"
        )
    print("\nAll models built and produced (1, 4) logits.")


if __name__ == "__main__":
    main()
