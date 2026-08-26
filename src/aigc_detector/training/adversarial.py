"""Adversarial formality training (W7) — gradient-reversal extension of LoRATrainer.

Rationale (DETECTOR_NOTES_2026-08.md FN-1 / 1D-collapse literature): fine-tuned
detectors collapse their representation onto a formality axis (cos 0.73-0.99)
because formal register correlates with the human label in HC3-style data.
The remedy with published support: train with a formality adversary through a
gradient-reversal layer so the representation cannot keep a formality direction.

Design:
- Formality score is CONTINUOUS and self-supervised (no manual annotation):
  lexical formality measure combining formal 公文 markers (positive) with
  colloquial/internet-register markers (negative), normalized to [0,1].
- Adversary: small MLP on the pooled [CLS] embedding predicting the formality
  score, connected via GradientReversal(λ) with λ ramped over training.
- Loss = CE(classification) + beta * MSE(adversary(formality)); GRL makes the
  backbone receive reversed gradients w.r.t. formality prediction.

Automation contract (plan v2.1 W7): the adversarial trainer NEVER writes to
the production adapter dir; candidates go to `*-adversarial-candidate/` and
the automated gate (`scripts/adversarial_gate.py`) decides promotion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

# ---------------------------------------------------------------------------
# Self-supervised continuous formality measure (zh-heavy, en-tolerant)
# ---------------------------------------------------------------------------

FORMAL_MARKERS = [
    "特此", "兹因", "兹有", "兹就", "郑重", "依据", "根据", "严格遵守",
    " hereby", "承諾", "承诺", " undertook", "应当", "予以", "如下",
    "如有违反", "承担", "监督管理", "履行", " hereby", "声明", "公告",
    "尊敬的", "拟", "经研究", "经核实", "现将", "上述", "本承诺",
]
INFORMAL_MARKERS = [
    "哈哈", "嘻嘻", "唉", "啊", "呀", "哦", "嗯", "诶", "哈哈哈", "笑死",
    "绝了", "牛", "离谱", "无语", "吐槽", "种草", "拔草", "打工人", "内卷",
    "yyds", "awsl", "绝绝子", "反正", "随便", "说白了", "我觉得吧", "讲真",
    "lol", "omg", "haha", "btw", "kinda", "wanna", "gonna", "u know",
]


def formality_score(text: str) -> float:
    """Continuous formality measure in [0,1]; self-supervised, no annotation.

    Deliberately lexical and cheap — its purpose is an adversary TARGET, not a
    classifier. Bounded quality is acceptable: the adversary penalizes the
    representation for carrying ANY direction correlated with this measure.
    """
    t = text.lower()
    hits = sum(t.count(m.lower()) for m in FORMAL_MARKERS)
    misses = sum(t.count(m.lower()) for m in INFORMAL_MARKERS)
    n = max(20, len(t))
    # per-1000-chars density, squashed
    f = min(1.0, hits * 1000.0 / n / 8.0)
    i = min(1.0, misses * 1000.0 / n / 4.0)
    return max(0.0, min(1.0, 0.5 + 0.5 * f - 0.5 * i))


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


# ---------------------------------------------------------------------------
# Adversary head
# ---------------------------------------------------------------------------


class FormalityAdversary(nn.Module):
    """MLP predicting the continuous formality score from pooled features."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(pooled, lambd)).squeeze(-1)
