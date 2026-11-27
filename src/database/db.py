"""
Module: src/database/db.py
Mục đích: Thiết lập cơ sở dữ liệu SQLite & SQLAlchemy lưu trữ bệnh án, phiên phân tích và kết quả chẩn đoán y khoa.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH = os.path.join(PROJECT_ROOT, "burnscan.db")

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Patient(Base):
    """Bảng lưu thông tin bệnh nhân"""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(50), unique=True, index=True) # Mã BN: BN-10023
    name = Column(String(100), default="Khách vãng lai")
    age = Column(Integer, nullable=True)
    gender = Column(String(20), default="Chưa rõ") # Nam / Nữ
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, default=65.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("AnalysisSession", back_populates="patient", cascade="all, delete-orphan")


class AnalysisSession(Base):
    """Bảng lưu trữ từng lần khám / tải ảnh phân tích vết bỏng"""
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(String(64), unique=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    
    # Thông tin ca bỏng & hiệu chuẩn giải phẫu
    body_part = Column(String(50), default="chest_abdomen")
    region_coverage = Column(Float, default=1.0)
    calibration_mode = Column(String(30), default="anatomical")
    burn_time_hours = Column(Float, default=1.0) # Thời gian từ khi bị bỏng
    pixels_per_cm = Column(Float, nullable=True) # Hiệu chuẩn thước đo

    # Đường dẫn hình ảnh
    original_image_path = Column(String(255))
    overlay_image_path = Column(String(255), nullable=True)
    contours_image_path = Column(String(255), nullable=True)
    roi_image_path = Column(String(255), nullable=True)
    gradcam_image_path = Column(String(255), nullable=True)
    pdf_report_path = Column(String(255), nullable=True)

    # Kết quả phân đoạn & diện tích
    total_pixels = Column(Integer, default=0)
    burn_pixels = Column(Integer, default=0)
    pct_frame_area = Column(Float, default=0.0)
    pct_skin_in_frame = Column(Float, default=100.0)
    pct_burn_of_skin = Column(Float, default=0.0)
    area_cm2 = Column(Float, nullable=True)
    estimated_tbsa_pct = Column(Float, default=0.0)

    # Kết quả phân loại cấp độ bỏng
    predicted_degree = Column(String(50), default="Độ 1")
    confidence = Column(Float, default=0.0)
    prob_deg_1 = Column(Float, default=0.0)
    prob_deg_2 = Column(Float, default=0.0)
    prob_deg_3 = Column(Float, default=0.0)

    # Phác đồ bù dịch Parkland
    parkland_total_ml = Column(Float, nullable=True)
    parkland_first_8h_ml = Column(Float, nullable=True)
    parkland_rate_8h_ml_h = Column(Float, nullable=True)
    parkland_rate_16h_ml_h = Column(Float, nullable=True)
    urgent_triage = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="sessions")


def init_db():
    Base.metadata.create_all(bind=engine)
    # Tự động nâng cấp lược đồ (Auto-migrate) cho các file burnscan.db cũ không mất dữ liệu
    new_columns = {
        "region_coverage": "FLOAT DEFAULT 1.0",
        "calibration_mode": "VARCHAR(30) DEFAULT 'anatomical'",
        "pct_skin_in_frame": "FLOAT DEFAULT 100.0",
        "pct_burn_of_skin": "FLOAT DEFAULT 0.0",
    }
    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(analysis_sessions)")).fetchall()
        }
        for col_name, col_def in new_columns.items():
            if existing and col_name not in existing:
                conn.execute(text(f"ALTER TABLE analysis_sessions ADD COLUMN {col_name} {col_def}"))
        existing_pat = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(patients)")).fetchall()
        }
        if existing_pat and "height_cm" not in existing_pat:
            conn.execute(text("ALTER TABLE patients ADD COLUMN height_cm FLOAT"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

