from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4


class BaselineSessionStore:
    """Thread-safe in-memory storage for demonstration sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._lock = Lock()

    def create(
        self,
        baseline_measurement: dict,
        settings: dict,
    ) -> dict:
        if baseline_measurement.get("measurement_status") != "MEASURED":
            raise ValueError(
                "A baseline session requires a successful measurement."
            )

        session_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        session = {
            "session_id": session_id,
            "session_status": "BASELINE_ESTABLISHED",
            "created_at": created_at,
            "baseline": deepcopy(baseline_measurement),
            "settings": deepcopy(settings),
            "follow_ups": [],
        }

        with self._lock:
            self._sessions[session_id] = session

        return deepcopy(session)

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            session = self._sessions.get(session_id)

        return deepcopy(session) if session is not None else None

    def add_follow_up(
        self,
        session_id: str,
        follow_up_measurement: dict,
        research_indicator: dict,
    ) -> dict | None:
        if follow_up_measurement.get("measurement_status") != "MEASURED":
            raise ValueError(
                "A follow-up entry requires a successful measurement."
            )

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            baseline_length = float(
                session["baseline"]["external_length_cm"]
            )
            follow_up_length = float(
                follow_up_measurement["external_length_cm"]
            )
            signed_change = round(follow_up_length - baseline_length, 3)

            entry = {
                "follow_up_id": uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "measurement": deepcopy(follow_up_measurement),
                "signed_change_cm": signed_change,
                "absolute_change_cm": round(abs(signed_change), 3),
                "research_indicator": deepcopy(research_indicator),
            }
            session["follow_ups"].append(entry)
            session["session_status"] = "FOLLOW_UP_MEASURED"

            return deepcopy(session)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def list_summaries(self) -> list[dict]:
        with self._lock:
            sessions = sorted(
                reversed(list(self._sessions.values())),
                key=lambda session: session["created_at"],
                reverse=True,
            )
            summaries = []
            for session in sessions:
                latest = (
                    session["follow_ups"][-1]
                    if session["follow_ups"]
                    else None
                )
                summaries.append({
                    "session_id": session["session_id"],
                    "created_at": session["created_at"],
                    "baseline_length_cm": session["baseline"][
                        "external_length_cm"
                    ],
                    "successful_follow_up_count": len(
                        session["follow_ups"]
                    ),
                    "latest_signed_change_cm": (
                        latest["signed_change_cm"] if latest else None
                    ),
                    "latest_indicator": deepcopy(
                        latest["research_indicator"]
                    ) if latest else None,
                })

        return summaries
