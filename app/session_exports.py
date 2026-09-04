from __future__ import annotations

import csv
from copy import deepcopy
from io import StringIO


WARNING = (
    "Research demonstration only; "
    "not clinically validated."
)

CSV_FIELDS = [
    "record_type",
    "record_id",
    "created_at",
    "external_length_cm",
    "signed_change_cm",
    "absolute_change_cm",
    "indicator_code",
    "pivc_confidence",
    "mark_confidence",
    "iou",
    "tolerance_cm",
    "imgsz",
]


def session_export_payload(session: dict) -> dict:
    payload = deepcopy(session)
    payload["warning"] = WARNING
    return payload


def session_csv(session: dict) -> str:
    settings = session["settings"]

    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_FIELDS,
    )
    writer.writeheader()

    writer.writerow({
        "record_type": "baseline",
        "record_id": session["session_id"],
        "created_at": session["created_at"],
        "external_length_cm": (
            session["baseline"]["external_length_cm"]
        ),
        "signed_change_cm": "",
        "absolute_change_cm": "",
        "indicator_code": "",
        **settings,
    })

    for follow_up in session["follow_ups"]:
        writer.writerow({
            "record_type": "follow_up",
            "record_id": follow_up["follow_up_id"],
            "created_at": follow_up["created_at"],
            "external_length_cm": (
                follow_up["measurement"][
                    "external_length_cm"
                ]
            ),
            "signed_change_cm": (
                follow_up["signed_change_cm"]
            ),
            "absolute_change_cm": (
                follow_up["absolute_change_cm"]
            ),
            "indicator_code": (
                follow_up["research_indicator"]["code"]
            ),
            **settings,
        })

    return output.getvalue()