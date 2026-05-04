from abc import abstractmethod

import torch
from torch import Tensor

from .base import Metric


class Accuracy(Metric):
    """전체 정확도 = 맞춘 개수 / 전체 개수"""

    def __init__(self):
        self.reset()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        # logits (B, C)이면 argmax, 이미 class indices (B,)면 그대로 사용
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class ConfusionMatrix(Metric):
    """혼동 행렬. matrix[t, p] = 실제 t를 p로 예측한 횟수."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        K = self.num_classes
        # bincount 트릭: (t, p) 쌍을 K*t+p로 평탄화한 뒤 KxK로 reshape
        idx = K * targets + preds
        bincount = torch.bincount(idx, minlength=K * K).reshape(K, K)
        if self.matrix.device != bincount.device:
            self.matrix = self.matrix.to(bincount.device)
        self.matrix += bincount

    def compute(self) -> Tensor:
        return self.matrix.clone()

    def reset(self) -> None:
        self.matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)


class _ConfusionBased(Metric):
    """ConfusionMatrix 기반의 Precision/Recall/F1 공통 부모.

    자식은 _per_class만 구현 — cm 텐서를 받아 클래스별 점수 (K,)를 반환.
    average로 클래스별 점수를 어떻게 합칠지 결정한다.
        - 'macro'   : 단순 평균 (클래스 불균형 무시)
        - 'weighted': 클래스별 실제 샘플 수로 가중 평균
        - 'none'    : 클래스별 점수 텐서 (K,) 그대로 반환
    micro는 멀티클래스에서 Accuracy와 동일하므로 별도 옵션으로 두지 않음.
    """

    def __init__(self, num_classes: int, average: str = "macro"):
        if average not in ("macro", "weighted", "none"):
            raise ValueError(
                f"average는 'macro' | 'weighted' | 'none' 중 하나 (got {average!r}). "
                "micro는 멀티클래스에서 Accuracy와 동일하므로 Accuracy를 사용하세요."
            )
        self.num_classes = num_classes
        self.average = average
        self.cm = ConfusionMatrix(num_classes)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        self.cm.update(preds, targets)

    def reset(self) -> None:
        self.cm.reset()

    def compute(self) -> float | Tensor:
        cm = self.cm.compute().float()
        per_class = self._per_class(cm)        # (K,)
        support = cm.sum(dim=1)                # 각 클래스의 실제 샘플 수

        if self.average == "none":
            return per_class
        if self.average == "macro":
            return per_class.mean().item()
        # weighted
        total = support.sum()
        if total == 0:
            return 0.0
        return (per_class * support / total).sum().item()

    @abstractmethod
    def _per_class(self, cm: Tensor) -> Tensor:
        ...


class Precision(_ConfusionBased):
    """Precision = TP / (TP + FP)"""
    def _per_class(self, cm: Tensor) -> Tensor:
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp                # 클래스 i로 예측한 것 중 실제는 i가 아닌 것
        denom = tp + fp
        return torch.where(denom > 0, tp / denom, torch.zeros_like(tp))


class Recall(_ConfusionBased):
    """Recall = TP / (TP + FN)"""
    def _per_class(self, cm: Tensor) -> Tensor:
        tp = cm.diag()
        fn = cm.sum(dim=1) - tp                # 실제 i인데 다른 클래스로 예측한 것
        denom = tp + fn
        return torch.where(denom > 0, tp / denom, torch.zeros_like(tp))


class F1(_ConfusionBased):
    """F1 = 2 * P * R / (P + R) = 2*TP / (2*TP + FP + FN)"""
    def _per_class(self, cm: Tensor) -> Tensor:
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = 2 * tp + fp + fn
        return torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
