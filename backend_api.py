import os
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, bytes_to_image
from src.face_engine import FaceEngine
from src.database import EmployeeDatabase
from src.liveness import check_liveness
from src.attendance import append_record

CONFIG = load_config("config.yaml")
app = FastAPI(title="Face Recognition Attendance API")

engine = FaceEngine(
    model_name=CONFIG["model"]["name"],
    detection_size=tuple(CONFIG["model"]["detection_size"]),
    ctx_id=CONFIG["model"]["ctx_id"],
)
database = EmployeeDatabase(CONFIG["paths"]["embeddings_db"])


class VerifyResponse(BaseModel):
    identified: bool
    employee_id: str | None
    name: str | None
    similarity: float
    confidence_percent: float
    status: str
    liveness_passed: bool
    liveness_reasons: list[str]


@app.post("/verify", response_model=VerifyResponse)
async def verify(image: UploadFile = File(...)):
    content = await image.read()
    frame = bytes_to_image(content)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    embedding, face = engine.get_embedding(frame)
    if embedding is None:
        raise HTTPException(status_code=422, detail="No face detected")

    rec_cfg = CONFIG["recognition"]
    live_cfg = CONFIG["liveness"]

    liveness_passed, liveness_reasons, _ = True, [], None
    if live_cfg["enabled"]:
        liveness_passed, liveness_reasons, _ = check_liveness(
            frame, face,
            blur_threshold=live_cfg["blur_threshold"],
            min_face_size=live_cfg["min_face_size"],
        )

    matches = engine.search(embedding, database, top_k=1)
    if not matches:
        raise HTTPException(status_code=404, detail="No enrolled employees to match against")

    employee_id, name, similarity = matches[0]
    confidence = engine.similarity_to_confidence(
        similarity, rec_cfg["similarity_low_bound"], rec_cfg["similarity_high_bound"]
    )

    threshold = rec_cfg["confidence_threshold_percent"]
    identified = confidence >= threshold and liveness_passed
    status = "auto_verified" if identified else "manual_required"

    append_record(
        CONFIG["paths"]["attendance_log"], employee_id, name, similarity,
        confidence, status, "face_recognition",
    )

    return VerifyResponse(
        identified=identified,
        employee_id=employee_id,
        name=name,
        similarity=similarity,
        confidence_percent=confidence,
        status=status,
        liveness_passed=liveness_passed,
        liveness_reasons=liveness_reasons,
    )


@app.get("/employees")
async def list_employees():
    return database.list_employees()


@app.get("/health")
async def health():
    return {"status": "ok", "employees_enrolled": database.count()}
