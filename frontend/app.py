"""
BurnScan Full-Stack Clinical API Server
Hệ thống Backend FastAPI tích hợp:
- Cơ sở dữ liệu SQLite & SQLAlchemy (Bệnh nhân, Phiên khám, Lịch sử phân tích)
- Bộ phân tích thị giác U-Net ResNet-34 & ResNet-34 Classification
- XAI Grad-CAM Heatmap
- Phác đồ bù dịch Parkland & Medical Chatbot RAG
- Xuất PDF Bệnh án lâm sàng
- Giao diện người dùng Apple HIG (Stitch UI 4-Step Pipeline)
"""

import os
import sys
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
from config import SEGMENTATION_MODELS, CLASSIFICATION_MODELS, DEFAULT_SEG_MODEL, DEFAULT_CLS_MODEL

def sanitize_json(data):
    """Đảm bảo mọi kiểu dữ liệu numpy (np.bool_, np.float64, ...) được chuyển về kiểu native Python để FastAPI JSON encoder không bị crash"""
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(v) for v in data]
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return sanitize_json(data.tolist())
    return data
from src.database.db import init_db, get_db, Patient, AnalysisSession
from src.vision.burn_analyzer import BurnAnalyzer
from src.utils.report_generator import generate_burn_report_pdf

app = FastAPI(title="BurnScan Clinical System API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "outputs/web_uploads")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs/reports")
STATIC_DIR = os.path.join(PROJECT_ROOT, "frontend/static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "frontend/templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

analyzer = None

@app.on_event("startup")
def startup_event():
    global analyzer
    print("[INFO] Khởi tạo Cơ sở dữ liệu SQLite...")
    init_db()
    print("[INFO] Đang nạp mô hình U-Net & ResNet-34...")
    analyzer = BurnAnalyzer()
    analyzer.load_models()
    print("[INFO] Backend & AI Models đã sẵn sàng phục vụ!")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        return templates.TemplateResponse(request, "index.html")
    return "<h1>BurnScan Backend Ready</h1>"

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Kiểm tra trạng thái hoạt động của Backend, CSDL SQLite và AI Models"""
    total_sessions = db.query(func.count(AnalysisSession.id)).scalar() or 0
    return {
        "status": "ok",
        "models_loaded": analyzer is not None,
        "database_connected": True,
        "total_sessions": total_sessions
    }

@app.get("/api/stats")
async def get_clinical_stats(db: Session = Depends(get_db)):
    """Thống kê tổng hợp các ca khám trong CSDL SQLite"""
    total_sessions = db.query(func.count(AnalysisSession.id)).scalar() or 0
    total_patients = db.query(func.count(Patient.id)).scalar() or 0
    urgent_cases = db.query(func.count(AnalysisSession.id)).filter(AnalysisSession.urgent_triage == True).scalar() or 0
    degree_rows = (
        db.query(AnalysisSession.predicted_degree, func.count(AnalysisSession.id))
        .group_by(AnalysisSession.predicted_degree)
        .all()
    )
    return {
        "total_sessions": total_sessions,
        "total_patients": total_patients,
        "urgent_cases": urgent_cases,
        "by_degree": {deg or "Chưa rõ": cnt for deg, cnt in degree_rows}
    }

def _read_metrics_file(path: str):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@app.get("/api/models")
async def list_models():
    """Liệt kê tất cả trọng số đã huấn luyện trong outputs/checkpoints và mô hình đang kích hoạt"""
    seg_list = []
    for key, cfg in SEGMENTATION_MODELS.items():
        seg_list.append({
            "key": key,
            "name": cfg["name"],
            "available": (
                (bool(cfg.get("checkpoints")) and all(os.path.exists(p) for p in cfg["checkpoints"]))
                or os.path.exists(cfg["checkpoint"])
            ),
            "active": analyzer is not None and analyzer.seg_model_key == key,
            "metrics": _read_metrics_file(cfg.get("metrics_file")),
        })

    cls_list = []
    for key, cfg in CLASSIFICATION_MODELS.items():
        cls_list.append({
            "key": key,
            "name": cfg["name"],
            "available": (
                (bool(cfg.get("checkpoints")) and all(os.path.exists(p) for p in cfg["checkpoints"]))
                or os.path.exists(cfg["checkpoint"])
            ),
            "active": analyzer is not None and analyzer.cls_model_key == key,
            "metrics": _read_metrics_file(cfg.get("metrics_file")),
        })

    return {
        "active_segmentation": analyzer.seg_model_key if analyzer else DEFAULT_SEG_MODEL,
        "active_classification": analyzer.cls_model_key if analyzer else DEFAULT_CLS_MODEL,
        "segmentation_models": seg_list,
        "classification_models": cls_list,
    }

class ModelSwitchRequest(BaseModel):
    seg_model_key: Optional[str] = None
    cls_model_key: Optional[str] = None

@app.post("/api/models/switch")
async def switch_models(payload: ModelSwitchRequest):
    """Chuyển đổi cặp trọng số Segmentation / Classification đang chạy"""
    global analyzer
    seg_key = payload.seg_model_key or (analyzer.seg_model_key if analyzer else DEFAULT_SEG_MODEL)
    cls_key = payload.cls_model_key or (analyzer.cls_model_key if analyzer else DEFAULT_CLS_MODEL)

    if seg_key not in SEGMENTATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Model phân đoạn không hợp lệ: {seg_key}")
    if cls_key not in CLASSIFICATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Model phân loại không hợp lệ: {cls_key}")

    new_analyzer = BurnAnalyzer(seg_model_key=seg_key, cls_model_key=cls_key)
    new_analyzer.load_models()
    analyzer = new_analyzer
    return {
        "status": "success",
        "active_segmentation": analyzer.seg_model_key,
        "active_classification": analyzer.cls_model_key,
    }

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Kiểm tra trạng thái hoạt động của Backend, CSDL SQLite và AI Models"""
    total_sessions = db.query(func.count(AnalysisSession.id)).scalar() or 0
    return {
        "status": "ok",
        "models_loaded": analyzer is not None,
        "database_connected": True,
        "total_sessions": total_sessions
    }

@app.get("/api/stats")
async def get_clinical_stats(db: Session = Depends(get_db)):
    """Thống kê tổng hợp các ca khám trong CSDL SQLite"""
    total_sessions = db.query(func.count(AnalysisSession.id)).scalar() or 0
    total_patients = db.query(func.count(Patient.id)).scalar() or 0
    urgent_cases = db.query(func.count(AnalysisSession.id)).filter(AnalysisSession.urgent_triage == True).scalar() or 0
    degree_rows = (
        db.query(AnalysisSession.predicted_degree, func.count(AnalysisSession.id))
        .group_by(AnalysisSession.predicted_degree)
        .all()
    )
    return {
        "total_sessions": total_sessions,
        "total_patients": total_patients,
        "urgent_cases": urgent_cases,
        "by_degree": {deg or "Chưa rõ": cnt for deg, cnt in degree_rows}
    }

def _read_metrics_file(path: str):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@app.get("/api/models")
async def list_models():
    """Liệt kê tất cả trọng số đã huấn luyện trong outputs/checkpoints và mô hình đang kích hoạt"""
    seg_list = []
    for key, cfg in SEGMENTATION_MODELS.items():
        seg_list.append({
            "key": key,
            "name": cfg["name"],
            "available": (
                (bool(cfg.get("checkpoints")) and all(os.path.exists(p) for p in cfg["checkpoints"]))
                or os.path.exists(cfg["checkpoint"])
            ),
            "active": analyzer is not None and analyzer.seg_model_key == key,
            "metrics": _read_metrics_file(cfg.get("metrics_file")),
        })

    cls_list = []
    for key, cfg in CLASSIFICATION_MODELS.items():
        cls_list.append({
            "key": key,
            "name": cfg["name"],
            "available": (
                (bool(cfg.get("checkpoints")) and all(os.path.exists(p) for p in cfg["checkpoints"]))
                or os.path.exists(cfg["checkpoint"])
            ),
            "active": analyzer is not None and analyzer.cls_model_key == key,
            "metrics": _read_metrics_file(cfg.get("metrics_file")),
        })

    return {
        "active_segmentation": analyzer.seg_model_key if analyzer else DEFAULT_SEG_MODEL,
        "active_classification": analyzer.cls_model_key if analyzer else DEFAULT_CLS_MODEL,
        "segmentation_models": seg_list,
        "classification_models": cls_list,
    }

class ModelSwitchRequest(BaseModel):
    seg_model_key: Optional[str] = None
    cls_model_key: Optional[str] = None

@app.post("/api/models/switch")
async def switch_models(payload: ModelSwitchRequest):
    """Chuyển đổi cặp trọng số Segmentation / Classification đang chạy"""
    global analyzer
    seg_key = payload.seg_model_key or (analyzer.seg_model_key if analyzer else DEFAULT_SEG_MODEL)
    cls_key = payload.cls_model_key or (analyzer.cls_model_key if analyzer else DEFAULT_CLS_MODEL)

    if seg_key not in SEGMENTATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Model phân đoạn không hợp lệ: {seg_key}")
    if cls_key not in CLASSIFICATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Model phân loại không hợp lệ: {cls_key}")

    new_analyzer = BurnAnalyzer(seg_model_key=seg_key, cls_model_key=cls_key)
    new_analyzer.load_models()
    analyzer = new_analyzer
    return {
        "status": "success",
        "active_segmentation": analyzer.seg_model_key,
        "active_classification": analyzer.cls_model_key,
    }

@app.post("/api/analyze")
async def analyze_burn(
    file: UploadFile = File(...),
    pixels_per_cm: Optional[float] = Form(None),
    body_part: str = Form("chest_abdomen"),
    weight_kg: float = Form(65.0),
    burn_time_hours: float = Form(1.0),
    patient_name: str = Form("Bệnh nhân vãng lai"),
    patient_age: Optional[str] = Form(None),
    patient_gender: str = Form("Chưa rõ"),
    db: Session = Depends(get_db)
):
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa sẵn sàng.")

    parsed_age: Optional[int] = None
    if patient_age is not None and str(patient_age).strip():
        try:
            parsed_age = int(str(patient_age).strip())
        except ValueError:
            parsed_age = None

    session_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    original_filename = f"{session_id}_original{ext}"
    original_path = os.path.join(UPLOAD_DIR, original_filename)

    contents = await file.read()
    with open(original_path, "wb") as f:
        f.write(contents)

    image_bgr = cv2.imread(original_path)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Không thể giải mã file ảnh.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Chạy AI Pipeline
    results = analyzer.analyze(
        image_input=image_rgb,
        pixels_per_cm=pixels_per_cm,
        body_part=body_part,
        weight_kg=weight_kg
    )

    overlay_filename = f"{session_id}_overlay.png"
    contours_filename = f"{session_id}_contours.png"
    roi_filename = f"{session_id}_roi.png"
    gradcam_filename = f"{session_id}_gradcam.png"

    overlay_path = os.path.join(UPLOAD_DIR, overlay_filename)
    contours_path = os.path.join(UPLOAD_DIR, contours_filename)
    roi_path = os.path.join(UPLOAD_DIR, roi_filename)
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)

    cv2.imwrite(overlay_path, cv2.cvtColor(results["visualizations"]["overlay"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(contours_path, cv2.cvtColor(results["visualizations"]["contours"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(roi_path, cv2.cvtColor(results["classification"]["roi_image"], cv2.COLOR_RGB2BGR))

    has_gradcam = False
    if results["classification"].get("gradcam_image") is not None:
        cv2.imwrite(gradcam_path, cv2.cvtColor(results["classification"]["gradcam_image"], cv2.COLOR_RGB2BGR))
        has_gradcam = True

    # Lưu thông tin vào CSDL SQLite
    patient = Patient(
        patient_code=f"BN-{int(datetime.utcnow().timestamp()) % 100000}",
        name=patient_name,
        age=parsed_age,
        gender=patient_gender,
        weight_kg=weight_kg
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    m = results["metrics"]
    c = results["classification"]
    p = m.get("parkland_plan")

    db_session = AnalysisSession(
        session_uuid=session_id,
        patient_id=patient.id,
        body_part=body_part,
        burn_time_hours=burn_time_hours,
        pixels_per_cm=pixels_per_cm,
        original_image_path=f"/uploads/{original_filename}",
        overlay_image_path=f"/uploads/{overlay_filename}",
        contours_image_path=f"/uploads/{contours_filename}",
        roi_image_path=f"/uploads/{roi_filename}",
        gradcam_image_path=f"/uploads/{gradcam_filename}" if has_gradcam else None,
        total_pixels=m.get("total_pixels", 0),
        burn_pixels=m.get("burn_pixels", 0),
        pct_frame_area=m.get("pct_frame_area", 0.0),
        area_cm2=m.get("area_cm2"),
        estimated_tbsa_pct=m.get("estimated_tbsa_pct", 0.0),
        predicted_degree=c.get("predicted_degree", "Độ 1"),
        confidence=c.get("confidence", 0.0),
        prob_deg_1=c["probabilities"].get("Độ 1 (Superficial)", 0.0),
        prob_deg_2=c["probabilities"].get("Độ 2 (Partial Thickness)", 0.0),
        prob_deg_3=c["probabilities"].get("Độ 3 (Full Thickness)", 0.0),
        parkland_total_ml=p["total_fluid_24h_ml"] if p else None,
        parkland_first_8h_ml=p["first_8h_ml"] if p else None,
        parkland_rate_8h_ml_h=p["rate_first_8h_ml_h"] if p else None,
        parkland_rate_16h_ml_h=p["rate_next_16h_ml_h"] if p else None,
        urgent_triage=p["urgent_triage"] if p else False
    )
    db.add(db_session)
    db.commit()

    clean_classification = {
        "predicted_degree": c.get("predicted_degree", "Độ 1"),
        "confidence": c.get("confidence", 0.0),
        "is_abstaining": c.get("is_abstaining", False),
        "ethical_status": c.get("ethical_status", "ACCEPT"),
        "abstain_message": c.get("abstain_message", ""),
        "probabilities": c.get("probabilities", {}),
    }

    return sanitize_json({
        "status": "success",
        "session_id": session_id,
        "patient": {
            "name": patient.name,
            "patient_code": patient.patient_code,
            "age": patient.age,
            "gender": patient.gender,
            "weight_kg": patient.weight_kg
        },
        "metrics": m,
        "classification": clean_classification,
        "images": {
            "original": f"/uploads/{original_filename}",
            "overlay": f"/uploads/{overlay_filename}",
            "contours": f"/uploads/{contours_filename}",
            "roi": f"/uploads/{roi_filename}",
            "gradcam": f"/uploads/{gradcam_filename}" if has_gradcam else None
        },
        "created_at": db_session.created_at.strftime("%d/%m/%Y %H:%M")
    })

@app.get("/api/sessions")
async def get_history(q: Optional[str] = None, db: Session = Depends(get_db)):
    """API lấy lịch sử các ca khám đã lưu trong CSDL kèm tìm kiếm"""
    query = db.query(AnalysisSession).join(Patient, AnalysisSession.patient_id == Patient.id, isouter=True)
    if q and q.strip():
        search = f"%{q.strip()}%"
        query = query.filter((Patient.name.ilike(search)) | (Patient.patient_code.ilike(search)) | (AnalysisSession.session_uuid.ilike(search)))
    
    sessions = query.order_by(AnalysisSession.created_at.desc()).limit(30).all()
    history = []
    for s in sessions:
        history.append({
            "session_id": s.session_uuid,
            "patient_code": s.patient.patient_code if s.patient else "N/A",
            "patient_name": s.patient.name if s.patient else "N/A",
            "patient_age": s.patient.age if s.patient else None,
            "patient_gender": s.patient.gender if s.patient else "Chưa rõ",
            "patient_weight": s.patient.weight_kg if s.patient else 65.0,
            "body_part": s.body_part,
            "pct_frame_area": round(s.pct_frame_area, 1) if s.pct_frame_area else 0.0,
            "area_cm2": round(s.area_cm2, 1) if s.area_cm2 else None,
            "estimated_tbsa_pct": round(s.estimated_tbsa_pct, 3) if s.estimated_tbsa_pct else 0.0,
            "predicted_degree": s.predicted_degree,
            "confidence": round(s.confidence, 1) if s.confidence else 0.0,
            "urgent_triage": s.urgent_triage,
            "thumbnail": s.overlay_image_path or s.original_image_path,
            "created_at": s.created_at.strftime("%d/%m/%Y %H:%M")
        })
    return {"history": history, "total": len(history)}

@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, db: Session = Depends(get_db)):
    """API lấy chi tiết đầy đủ một ca khám theo UUID để hiển thị lại kết quả"""
    s = db.query(AnalysisSession).filter(AnalysisSession.session_uuid == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca khám.")

    parkland = None
    if s.parkland_total_ml:
        parkland = {
            "total_fluid_24h_ml": round(s.parkland_total_ml, 1),
            "first_8h_ml": round(s.parkland_first_8h_ml, 1) if s.parkland_first_8h_ml else round(s.parkland_total_ml / 2, 1),
            "rate_first_8h_ml_h": round(s.parkland_rate_8h_ml_h, 1) if s.parkland_rate_8h_ml_h else None,
            "rate_next_16h_ml_h": round(s.parkland_rate_16h_ml_h, 1) if s.parkland_rate_16h_ml_h else None,
            "urgent_triage": bool(s.urgent_triage)
        }

    return sanitize_json({
        "status": "success",
        "session_id": s.session_uuid,
        "patient": {
            "name": s.patient.name if s.patient else "Bệnh nhân vãng lai",
            "patient_code": s.patient.patient_code if s.patient else f"BN-{s.session_uuid}",
            "age": s.patient.age if s.patient else None,
            "gender": s.patient.gender if s.patient else "Chưa rõ",
            "weight_kg": s.patient.weight_kg if s.patient else 65.0
        },
        "metrics": {
            "total_pixels": s.total_pixels,
            "burn_pixels": s.burn_pixels,
            "pct_frame_area": round(s.pct_frame_area, 1) if s.pct_frame_area else 0.0,
            "area_cm2": round(s.area_cm2, 1) if s.area_cm2 else None,
            "body_part": s.body_part,
            "burn_time_hours": s.burn_time_hours or 1.0,
            "pixels_per_cm": s.pixels_per_cm,
            "estimated_tbsa_pct": round(s.estimated_tbsa_pct, 3) if s.estimated_tbsa_pct else 0.0,
            "parkland_plan": parkland
        },
        "classification": {
            "predicted_degree": s.predicted_degree,
            "confidence": round(s.confidence, 1) if s.confidence else 0.0,
            "probabilities": {
                "Độ 1 (Superficial)": round(s.prob_deg_1, 1) if s.prob_deg_1 else 0.0,
                "Độ 2 (Partial Thickness)": round(s.prob_deg_2, 1) if s.prob_deg_2 else 0.0,
                "Độ 3 (Full Thickness)": round(s.prob_deg_3, 1) if s.prob_deg_3 else 0.0
            }
        },
        "images": {
            "original": s.original_image_path,
            "overlay": s.overlay_image_path or s.original_image_path,
            "contours": s.contours_image_path or s.original_image_path,
            "roi": s.roi_image_path or s.original_image_path,
            "gradcam": s.gradcam_image_path
        },
        "created_at": s.created_at.strftime("%d/%m/%Y %H:%M")
    })

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """API xóa ca khám khỏi cơ sở dữ liệu"""
    s = db.query(AnalysisSession).filter(AnalysisSession.session_uuid == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca khám.")
    db.delete(s)
    db.commit()
    return {"status": "success", "message": f"Đã xóa ca khám {session_id}"}

@app.post("/api/export-pdf")
async def export_pdf(data: dict, db: Session = Depends(get_db)):
    session_id = data.get("session_id", str(uuid.uuid4())[:8])
    pdf_filename = f"BurnScan_Report_{session_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    orig_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(UPLOAD_DIR, f"{session_id}_original{ext}")
        if os.path.exists(candidate):
            orig_path = candidate
            break
    if orig_path is None:
        orig_path = os.path.join(UPLOAD_DIR, f"{session_id}_original.jpg")

    overlay_path = os.path.join(UPLOAD_DIR, f"{session_id}_overlay.png")
    gradcam_path = os.path.join(UPLOAD_DIR, f"{session_id}_gradcam.png")

    generate_burn_report_pdf(
        output_pdf_path=pdf_path,
        data=data,
        original_img_path=orig_path,
        overlay_img_path=overlay_path,
        gradcam_img_path=gradcam_path if os.path.exists(gradcam_path) else None
    )

    db_session = db.query(AnalysisSession).filter(AnalysisSession.session_uuid == session_id).first()
    if db_session:
        db_session.pdf_report_path = f"/api/download-pdf/{pdf_filename}"
        db.commit()

    return {"pdf_url": f"/api/download-pdf/{pdf_filename}"}

@app.get("/api/download-pdf/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="File không tồn tại.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("frontend.app:app", host="127.0.0.1", port=8000, reload=True)
