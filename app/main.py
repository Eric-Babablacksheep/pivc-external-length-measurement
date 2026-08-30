from __future__ import annotations

import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO
from session_store import BaselineSessionStore
from session_exports import session_csv, session_export_payload
from decimal import Decimal
from settings_store import ResearchSettings, ResearchSettingsStore

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

PIVC_MODULE_DIR = Path(
    os.getenv("PIVC_MODULE_DIR", PROJECT_ROOT / "pi-cent_conv")
).resolve()

MODEL_PATH = Path(
    os.getenv("PIVC_MODEL_PATH", PIVC_MODULE_DIR / "best.pt")
).resolve()

RUNTIME_DIR = APP_DIR / "runtime"
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
DIAGNOSTIC_DIR = RUNTIME_DIR / "diagnostics"
ULTRALYTICS_DIR = RUNTIME_DIR / "ultralytics"

for directory in (
    RUNTIME_DIR,
    UPLOAD_DIR,
    DIAGNOSTIC_DIR,
    ULTRALYTICS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Prevent Ultralytics from trying to write into a restricted user folder.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))

if not PIVC_MODULE_DIR.is_dir():
    raise RuntimeError(
        f"PIVC module directory was not found: {PIVC_MODULE_DIR}"
    )

if str(PIVC_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(PIVC_MODULE_DIR))


# Import the existing tested measurement implementation.
from pivc_validation import (  # noqa: E402
    ValidationCase,
    ValidationConfig,
    process_validation_case,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

CONTENT_TYPE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}

SESSION_STORE = BaselineSessionStore()
SETTINGS_STORE = ResearchSettingsStore()


class SettingsUpdate(BaseModel):
    confidence: float = Field(ge=0.05, le=0.95)
    iou: float = Field(ge=0.10, le=0.95)
    tolerance_cm: float = Field(ge=0.01, le=1.00)


def validation_config_for(
    settings: ResearchSettings,
) -> ValidationConfig:
    return ValidationConfig(
        imgsz=settings.imgsz,
        confidence=settings.confidence,
        iou=settings.iou,
        device="cpu",
        known_mark_spacing_cm=1.0,
        accuracy_tolerance_cm=settings.tolerance_cm,
    )

# ---------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file was not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    if getattr(model, "task", None) != "segment":
        raise RuntimeError(
            f"Expected a segmentation model, but model task is "
            f"{getattr(model, 'task', None)!r}."
        )

    class_names = {
        int(class_id): str(name).lower()
        for class_id, name in model.names.items()
    }

    if "mark" not in class_names.values():
        raise RuntimeError("The model does not contain the 'mark' class.")

    if "picc" not in class_names.values():
        raise RuntimeError("The model does not contain the 'picc' class.")

    app.state.model = model
    app.state.class_names = class_names

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Model task: {model.task}")
    print(f"Classes: {class_names}")
    print("Device: cpu")

    yield

    app.state.model = None


app = FastAPI(
    title="PIVC External-Length Measurement API",
    description=(
        "Research demonstration API for estimating external PIVC length "
        "from segmentation masks and built-in 1 cm graduation marks."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)

# This is convenient during local frontend development.
# Restrict these origins before any non-demo deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/diagnostics",
    StaticFiles(directory=str(DIAGNOSTIC_DIR)),
    name="diagnostics",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def validate_image_bytes(image_bytes: bytes) -> None:
    """Confirm that the uploaded bytes contain a readable image."""

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image exceeds the 20 MB limit.",
        )

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable JPG or PNG image.",
        )

    height, width = decoded.shape[:2]

    if height < 256 or width < 256:
        raise HTTPException(
            status_code=400,
            detail="The image is too small. Minimum size is 256 × 256 pixels.",
        )


def diagnostic_url(request: Request, diagnostic_path: str) -> str | None:
    if not diagnostic_path:
        return None

    path = Path(diagnostic_path)

    if not path.is_file():
        return None

    return str(
        request.url_for(
            "diagnostics",
            path=path.name,
        )
    )

def classify_research_change(
    signed_change_cm: float | None,
    tolerance_cm: float,
) -> dict:
    """
    Classify measured external-length change for research demonstration.

    This uses the algorithm's ±0.10 cm measurement tolerance. It is not
    a clinically validated dislodgement threshold.
    """

    threshold = Decimal(str(tolerance_cm))

    if signed_change_cm is None:
        return {
            "code": "UNABLE_TO_ASSESS",
            "label": "Unable to assess",
            "direction": None,
            "threshold_cm": float(threshold),
            "message": (
                "A safe follow-up measurement was not available. "
                "Retake the image before interpreting change."
            ),
        }

    change = Decimal(str(signed_change_cm))

    if abs(change) <= threshold:
        return {
            "code": "WITHIN_TOLERANCE",
            "label": "Within measurement tolerance",
            "direction": "stable",
            "threshold_cm": float(threshold),
            "message": (
                f"The measured change is within the ±{threshold} cm "
                "research tolerance."
            ),
        }

    if change > threshold:
        return {
            "code": "OUTWARD_INCREASE",
            "label": "Outward-length increase detected",
            "direction": "increase",
            "threshold_cm": float(threshold),
            "message": (
                "The measured external PIVC length increased beyond "
                "the research tolerance. Review the images and "
                "measurement evidence."
            ),
        }

    return {
        "code": "INWARD_DECREASE",
        "label": "Inward-length decrease detected",
        "direction": "decrease",
        "threshold_cm": float(threshold),
        "message": (
            "The measured external PIVC length decreased beyond "
            "the research tolerance. Review the images and "
            "measurement evidence."
        ),
    }

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    settings = SETTINGS_STORE.get()
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model", None) is not None,
        "model_task": getattr(
            getattr(request.app.state, "model", None),
            "task",
            None,
        ),
        "classes": getattr(request.app.state, "class_names", {}),
        "device": "cpu",
        "imgsz": settings.imgsz,
        "confidence": settings.confidence,
        "iou": settings.iou,
        "active_sessions": SESSION_STORE.count(),
    }


def settings_response(request: Request, settings: ResearchSettings) -> dict:
    model = getattr(request.app.state, "model", None)
    return {
        "editable": {
            "confidence": settings.confidence,
            "iou": settings.iou,
            "tolerance_cm": settings.tolerance_cm,
        },
        "runtime": {
            "imgsz": settings.imgsz,
            "device": "cpu",
            "model_filename": MODEL_PATH.name,
            "model_loaded": model is not None,
            "model_task": getattr(model, "task", None),
            "classes": getattr(request.app.state, "class_names", {}),
            "application_version": app.version,
        },
        "applies_to": "new_sessions",
        "warning": "Research demonstration only; not clinically validated.",
    }


@app.get("/api/v1/settings")
def get_settings(request: Request):
    return settings_response(request, SETTINGS_STORE.get())


@app.put("/api/v1/settings")
def update_settings(request: Request, update: SettingsUpdate):
    settings = SETTINGS_STORE.update(
        confidence=update.confidence,
        iou=update.iou,
        tolerance_cm=update.tolerance_cm,
    )
    return settings_response(request, settings)


@app.post("/api/v1/settings/reset")
def reset_settings(request: Request):
    return settings_response(request, SETTINGS_STORE.reset())


@app.post("/api/v1/measure")
async def measure_pivc(
    request: Request,
    image: UploadFile = File(...),
    settings: ResearchSettings | None = None,
):
    selected_settings = settings or SETTINGS_STORE.get()
    measurement_config = validation_config_for(selected_settings)
    content_type = (image.content_type or "").lower()

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPG and PNG images are supported.",
        )

    image_bytes = await image.read()
    await image.close()

    validate_image_bytes(image_bytes)

    case_token = uuid.uuid4().hex
    suffix = CONTENT_TYPE_SUFFIX[content_type]
    upload_path = UPLOAD_DIR / f"{case_token}{suffix}"

    upload_path.write_bytes(image_bytes)

    # ValidationCase requires ground-truth metadata because it was originally
    # designed for validation. The placeholder value is not returned or used
    # to judge application measurements. Only estimated_length_cm is used.
    case = ValidationCase(
        image_path=upload_path,
        length_group="APPLICATION",
        known_length_cm=0.0,
        dressing_condition="unspecified",
    )

    try:
        result = process_validation_case(
            case=case,
            model=request.app.state.model,
            config=measurement_config,
            diagnostics_dir=DIAGNOSTIC_DIR,
        )
    finally:
        # The original upload is no longer needed after processing.
        upload_path.unlink(missing_ok=True)

    overlay_url = diagnostic_url(
        request,
        result.diagnostic_path,
    )

    measured = (
        result.status != "REJECTED"
        and result.estimated_length_cm is not None
    )

    if not measured:
        return {
            "measurement_status": "REJECTED",
            "external_length_cm": None,
            "pivc_detected": result.pivc_detected,
            "pivc_confidence": result.pivc_confidence,
            "marks_detected": result.marks_detected,
            "corrected_centreline_px": None,
            "consecutive_mark_spacings_px": [],
            "pixels_per_cm": None,
            "rejection_stage": result.rejection_stage,
            "rejection_reason": result.rejection_reason,
            "diagnostic_available": result.diagnostic_available,
            "diagnostic_url": overlay_url,
            "warning": "Research demonstration only; not clinically validated.",
        }

    return {
        "measurement_status": "MEASURED",
        "external_length_cm": round(result.estimated_length_cm, 3),
        "pivc_detected": result.pivc_detected,
        "pivc_confidence": (
            round(result.pivc_confidence, 4)
            if result.pivc_confidence is not None
            else None
        ),
        "marks_detected": result.marks_detected,
        "corrected_centreline_px": (
            round(result.corrected_centreline_px, 2)
            if result.corrected_centreline_px is not None
            else None
        ),
        "consecutive_mark_spacings_px": [
            round(value, 2)
            for value in result.consecutive_mark_spacings_px
        ],
        "pixels_per_cm": (
            round(result.pixels_per_cm, 2)
            if result.pixels_per_cm is not None
            else None
        ),
        "rejection_stage": None,
        "rejection_reason": None,
        "diagnostic_available": result.diagnostic_available,
        "diagnostic_url": overlay_url,
        "warning": "Research demonstration only; not clinically validated.",
    }

@app.post("/api/v1/sessions/baseline")
async def establish_baseline(
    request: Request,
    image: UploadFile = File(...),
):
    """
    Analyse an uploaded image and establish it as the session baseline.

    A rejected measurement never creates a session.
    """

    settings_snapshot = SETTINGS_STORE.get()
    measurement = await measure_pivc(
        request=request,
        image=image,
        settings=settings_snapshot,
    )

    if measurement["measurement_status"] != "MEASURED":
        return {
            "session_id": None,
            "session_status": "BASELINE_REJECTED",
            "created_at": None,
            "baseline": measurement,
            "message": (
                "The baseline was not created because the image did not "
                "produce a safe measurement. Please retake the image."
            ),
        }

    session = SESSION_STORE.create(
        measurement,
        settings_snapshot.to_dict(),
    )

    return {
        **session,
        "message": "Baseline measurement established successfully.",
    }


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str):
    session = SESSION_STORE.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="The requested baseline session was not found.",
        )

    return session


@app.get("/api/v1/sessions")
def list_sessions():
    return {"sessions": SESSION_STORE.list_summaries()}


@app.get("/api/v1/sessions/{session_id}/export.json")
def export_session_json(session_id: str):
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(
        content=session_export_payload(session),
        headers={
            "Content-Disposition": (
                f'attachment; filename="pivc-session-{session_id}.json"'
            )
        },
    )


@app.get("/api/v1/sessions/{session_id}/export.csv")
def export_session_csv(session_id: str):
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return StreamingResponse(
        iter([session_csv(session)]),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="pivc-session-{session_id}.csv"'
            )
        },
    )


@app.post("/api/v1/sessions/{session_id}/follow-up")
async def analyse_follow_up(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
):
    """Compare one safely measured follow-up image with its baseline."""

    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="The requested baseline session was not found.",
        )

    settings_snapshot = ResearchSettings(**session["settings"])
    measurement = await measure_pivc(
        request=request,
        image=image,
        settings=settings_snapshot,
    )

    if measurement["measurement_status"] != "MEASURED":
        return {
            "session_id": session_id,
            "session_status": "FOLLOW_UP_REJECTED",
            "baseline": session["baseline"],
            "follow_up": measurement,
            "comparison": None,
            "research_indicator": classify_research_change(
                None,
                settings_snapshot.tolerance_cm,
            ),
            "successful_follow_up_count": len(session["follow_ups"]),
            "message": (
                "The follow-up was not stored because it did not produce "
                "a safe measurement. Please retake the image."
            ),
        }

    signed_change = round(
        float(measurement["external_length_cm"])
        - float(session["baseline"]["external_length_cm"]),
        3,
    )
    research_indicator = classify_research_change(
        signed_change,
        settings_snapshot.tolerance_cm,
    )
    updated = SESSION_STORE.add_follow_up(
        session_id,
        measurement,
        research_indicator,
    )
    entry = updated["follow_ups"][-1]

    return {
        "session_id": session_id,
        "session_status": "FOLLOW_UP_MEASURED",
        "baseline": updated["baseline"],
        "follow_up": entry["measurement"],
        "comparison": {
            "signed_change_cm": entry["signed_change_cm"],
            "absolute_change_cm": entry["absolute_change_cm"],
        },
        "research_indicator": research_indicator,
        "follow_up_id": entry["follow_up_id"],
        "created_at": entry["created_at"],
        "successful_follow_up_count": len(updated["follow_ups"]),
        "message": "Follow-up compared with the baseline successfully.",
    }
