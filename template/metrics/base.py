from abc import ABC, abstractmethod
from torch import Tensor


class Metric(ABC):
    """배치 단위로 누적해 epoch 끝에 값을 계산하는 메트릭의 베이스 클래스.

    사용 패턴:
        metric = Accuracy()
        for x, y in loader:
            logits = model(x)
            metric.update(logits, y)   # 배치마다 상태 누적
        score = metric.compute()       # epoch 끝에 최종값
        metric.reset()                 # 다음 epoch 전에 초기화
    """

    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor) -> None:
        """배치의 예측/정답을 받아 내부 상태를 누적한다."""
        ...

    @abstractmethod
    def compute(self) -> float | Tensor:
        """누적된 상태로부터 최종 메트릭 값을 계산한다."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """내부 상태를 초기화한다."""
        ...
