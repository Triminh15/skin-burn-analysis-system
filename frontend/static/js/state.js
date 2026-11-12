/**
 * BurnScan - Global Application State & Constants
 */

const appState = {
  currentStep: 1,
  selectedFile: null,
  selectedDataUrl: null,
  calibDistPx: 250,
  calibRealCm: 8.56,
  calibPxPerCm: 29.2,
  bodyPart: 'chest_abdomen',
  activeSessionData: null,
  isDraggingKnob: null,
  knob1Pos: { x: 120, y: 240 },
  knob2Pos: { x: 380, y: 240 }
};

// Wallace Rule of Nines (Luật số 9 Wallace - Phân vùng giải phẫu)
const BODY_PARTS = {
  'chest_abdomen': { name: 'Ngực & Bụng trước', pct: 18.0 },
  'back_buttocks': { name: 'Lưng & Mông sau', pct: 18.0 },
  'left_arm': { name: 'Cánh tay trái', pct: 9.0 },
  'right_arm': { name: 'Cánh tay phải', pct: 9.0 },
  'left_leg': { name: 'Đùi & Cẳng chân trái', pct: 18.0 },
  'right_leg': { name: 'Đùi & Cẳng chân phải', pct: 18.0 },
  'head_neck': { name: 'Đầu - Mặt - Cổ', pct: 9.0 },
  'perineum': { name: 'Bộ phận sinh dục', pct: 1.0 }
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

