/**
 * BurnScan - Global Application State & Constants
 */

const appState = {
  currentStep: 1,
  selectedFile: null,
  selectedDataUrl: null,
  calibrationMode: 'anatomical', // 'anatomical' (mặc định chuẩn y khoa không cần thước) | 'ruler' (có vật chuẩn)
  calibDistPx: 250,
  calibNaturalDistPx: 250,
  calibRealCm: 8.56,
  calibPxPerCm: 29.2,
  bodyPart: 'back_buttocks',
  regionCoverage: 1.0,
  activeSessionData: null,
  isDraggingKnob: null,
  knob1Pos: { x: 120, y: 240 },
  knob2Pos: { x: 380, y: 240 }
};

// Wallace Rule of Nines + Palmar Rule (1%) + Clinical Sub-regions
const BODY_PARTS = {
  'palm_hand': { name: 'Bàn tay / Lòng bàn tay (Quy tắc 1%)', pct: 1.0 },
  'forearm': { name: 'Cẳng tay (1 bên)', pct: 4.0 },
  'upper_arm': { name: 'Cánh tay trên (1 bên)', pct: 4.5 },
  'left_arm': { name: 'Toàn bộ cánh tay trái', pct: 9.0 },
  'right_arm': { name: 'Toàn bộ cánh tay phải', pct: 9.0 },
  'face_only': { name: 'Vùng Mặt', pct: 4.5 },
  'head_neck': { name: 'Đầu - Mặt - Cổ', pct: 9.0 },
  'chest_only': { name: 'Nửa thân trước - Ngực', pct: 9.0 },
  'abdomen_only': { name: 'Nửa thân trước - Bụng', pct: 9.0 },
  'chest_abdomen': { name: 'Toàn bộ Ngực & Bụng - Thân trước', pct: 18.0 },
  'back_only': { name: 'Vùng Lưng (không gồm Mông)', pct: 13.0 },
  'upper_back': { name: 'Nửa lưng trên', pct: 9.0 },
  'back_buttocks': { name: 'Toàn bộ Lưng & Thân sau', pct: 18.0 },
  'thigh': { name: 'Đùi (1 bên)', pct: 9.0 },
  'lower_leg': { name: 'Cẳng chân (1 bên)', pct: 7.0 },
  'foot': { name: 'Bàn chân (1 bên)', pct: 2.0 },
  'left_leg': { name: 'Toàn bộ chân trái', pct: 18.0 },
  'right_leg': { name: 'Toàn bộ chân phải', pct: 18.0 },
  'perineum': { name: 'Bộ phận sinh dục / Tầng sinh môn', pct: 1.0 }
};

/**
 * Toast Notification Helper
 */
function showToast(message, icon = 'check_circle') {
  const toast = document.getElementById('toast');
  const msgEl = document.getElementById('toast-message');
  const iconEl = document.getElementById('toast-icon');
  if (toast && msgEl && iconEl) {
    msgEl.innerText = message;
    iconEl.innerText = icon;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3500);
  }
}
