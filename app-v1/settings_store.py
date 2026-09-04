from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock


@dataclass(frozen=True)
class ResearchSettings:
    """Immutable inference and research-indicator settings."""

    pivc_confidence: float = 0.50
    mark_confidence: float = 0.30
    iou: float = 0.70
    tolerance_cm: float = 0.10
    imgsz: int = 960

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchSettingsStore:
    """Thread-safe in-memory defaults for newly created sessions."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._settings = ResearchSettings()

    @staticmethod
    def _validate(
        pivc_confidence: float,
        mark_confidence: float,
        iou: float,
        tolerance_cm: float,
    ) -> None:
        bounds = {
            "pivc_confidence": (
                pivc_confidence,
                0.05,
                0.95,
            ),
            "mark_confidence": (
                mark_confidence,
                0.05,
                0.95,
            ),
            "iou": (
                iou,
                0.10,
                0.95,
            ),
            "tolerance_cm": (
                tolerance_cm,
                0.01,
                1.00,
            ),
        }

        for name, (value, minimum, maximum) in bounds.items():
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be between "
                    f"{minimum} and {maximum}."
                )

    def get(self) -> ResearchSettings:
        with self._lock:
            return self._settings

    def update(
        self,
        *,
        pivc_confidence: float,
        mark_confidence: float,
        iou: float,
        tolerance_cm: float,
    ) -> ResearchSettings:
        self._validate(
            pivc_confidence,
            mark_confidence,
            iou,
            tolerance_cm,
        )

        replacement = ResearchSettings(
            pivc_confidence=pivc_confidence,
            mark_confidence=mark_confidence,
            iou=iou,
            tolerance_cm=tolerance_cm,
        )

        with self._lock:
            self._settings = replacement

        return replacement

    def reset(self) -> ResearchSettings:
        with self._lock:
            self._settings = ResearchSettings()
            return self._settings